# -*- coding: utf-8 -*-
"""Experimento VAE-GAN (raio-X): VAE com discriminador GAN + perda perceptual.

run(): importar -> tratar -> carregar -> treinar -> avaliar -> salvar.
Loss G: l1 + l_perc*perceptual + beta*KL + l_adv*adversarial.
beta sobe por warmup (beta_warm_epochs); o GAN entra na época disc_start_epoch.
Checkpoint a cada ckpt_every épocas (pesos + optimizers + RNG + epoch/step);
com resume=true retoma de outputs/vae_gan_ckpt.pt se existir e o config bater.

Logging W&B:
- escalares (l1, perc, kl, adv, d, beta) -> toda época;
- visuais (recon + interpolação) e gate (colapso) -> a cada 10 épocas e no fim.
Devolve o caminho do modelo (o chassi sobe como artefato).
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import wandb

from experiments.base import Experiment
from experiments.vae_gan.data import importar, tratar, carregar, _ImagensXRay
from experiments.vae_gan.models import Encoder, Decoder, PatchDiscriminator, VGGPerceptual, reparam
from experiments.vae_gan.losses import kl_div, d_hinge, g_hinge


def _unwrap(m):
    """Tira o wrapper do DataParallel pra salvar/carregar state_dict limpo."""
    return m.module if isinstance(m, nn.DataParallel) else m


# Hiperparâmetros que precisam bater entre o checkpoint e o config no resume:
# os 4 primeiros mudam a arquitetura (shape mismatch); os demais mudariam a
# receita no meio do treino de forma silenciosa.
_HPARAMS_RESUME = ["channels", "img_size", "base", "zdim", "batch_size", "seed",
                   "beta", "lr_g", "lr_d", "l_perc", "l_adv",
                   "beta_warm_epochs", "disc_start_epoch"]


def slerp(z_a, z_b, t):
    """Interpolação esférica entre z_a e z_b (t em [0,1]).

    Segue o arco do grande-círculo em vez da corda reta. Numa gaussiana de dim alta
    quase toda amostra mora numa casca de raio ~sqrt(d); a corda reta corta o interior
    vazio (norma menor) -> meio do caminho fora da distribuição -> recon borrada. O slerp
    mantém a norma ~constante, então todo z intermediário fica na casca que o decoder viu.
    Quando o ângulo -> 0 o limite vira a interpolação linear naturalmente.
    """
    a = z_a / z_a.norm(dim=1, keepdim=True)
    b = z_b / z_b.norm(dim=1, keepdim=True)
    omega = torch.acos((a * b).sum(1, keepdim=True).clamp(-1.0, 1.0))
    so = torch.sin(omega).clamp_min(1e-6)
    return torch.sin((1 - t) * omega) / so * z_a + torch.sin(t * omega) / so * z_b


class VaeGan(Experiment):
    """Treina um VAE com discriminador GAN (PatchGAN) e perda perceptual (VGG)."""

    def run(self) -> str:
        cfg = self.cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Rodando em: {device}")

        # ---- dados ---- (seed = universal, injetado pelo chassi)
        itens = importar(cfg["data_root"])
        train, val, test = tratar(itens, cfg["val_frac"], cfg["test_frac"], cfg["seed"])
        dl_train, dl_val, _ = carregar(train, val, [], cfg["img_size"], cfg["batch_size"],
                                       cfg["channels"], cfg["num_workers"])
        print(f"split -> train {len(train)} | val {len(val)} | test {len(test)} (treina só no train)")

        # conjuntos fixos (saudáveis e doentes) p/ visuais + gate, sempre os mesmos
        xs_h, xs_s = self._fixos(train, cfg, device)

        # ---- modelos ----
        ch = cfg["channels"]
        enc = Encoder(ch, cfg["base"], cfg["zdim"], cfg["img_size"]).to(device)
        dec = Decoder(ch, cfg["base"], cfg["zdim"], cfg["img_size"]).to(device)
        disc = PatchDiscriminator(ch, cfg["base"]).to(device)
        perc = VGGPerceptual().to(device) if cfg["l_perc"] > 0 else None  # None = sem perceptual

        # 2 GPUs (Kaggle T4 x2): divide o lote entre elas. No-op em 1 GPU/CPU.
        if cfg.get("use_dp", False) and device == "cuda" and torch.cuda.device_count() > 1:
            print(f"DataParallel em {torch.cuda.device_count()} GPUs")
            enc, dec, disc = nn.DataParallel(enc), nn.DataParallel(dec), nn.DataParallel(disc)

        optG = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                                lr=cfg["lr_g"], betas=(0.5, 0.9))
        optD = torch.optim.Adam(disc.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.9))

        # AMP: fp16 nos forwards/backwards (tensor cores da T4), um scaler por
        # optimizer (G e D têm backwards independentes). No-op em CPU.
        use_amp = cfg.get("amp", True) and device == "cuda"
        scaler_g = torch.amp.GradScaler(device, enabled=use_amp)
        scaler_d = torch.amp.GradScaler(device, enabled=use_amp)

        # ---- resume ----
        os.makedirs("outputs", exist_ok=True)
        # em sweep, sufixa com o id do run: evita colisão de arquivos entre agentes
        # paralelos. Run normal fica sem sufixo (resume_id do chassi acha o ckpt).
        rid = f"_{wandb.run.id}" if wandb.run and wandb.run.sweep_id else ""
        ckpt_path = os.path.join("outputs", f"vae_gan{rid}_ckpt.pt")
        start_ep, step = 0, 0
        if cfg.get("resume", False) and os.path.exists(ckpt_path):
            # weights_only=False: o ckpt tem estados de RNG (objetos python/numpy),
            # que o default seguro do torch>=2.6 rejeitaria. Arquivo é nosso, ok.
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            dif = [k for k in _HPARAMS_RESUME if ck["hparams"].get(k) != cfg.get(k)]
            if dif:
                raise ValueError(
                    f"checkpoint incompatível com o config atual em {dif}: "
                    f"apague {ckpt_path} ou desligue resume pra treinar do zero")
            _unwrap(enc).load_state_dict(ck["enc"])
            _unwrap(dec).load_state_dict(ck["dec"])
            _unwrap(disc).load_state_dict(ck["disc"])
            optG.load_state_dict(ck["optG"])
            optD.load_state_dict(ck["optD"])
            start_ep, step = ck["epoch"], ck["step"]
            if use_amp and ck.get("scaler_g") is not None:
                scaler_g.load_state_dict(ck["scaler_g"])
                scaler_d.load_state_dict(ck["scaler_d"])
            random.setstate(ck["rng"]["py"])
            np.random.set_state(ck["rng"]["np"])
            torch.set_rng_state(ck["rng"]["torch"])
            if device == "cuda" and ck["rng"]["cuda"] is not None:
                torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
            print(f"resume: retomando da época {start_ep} (step {step})")

        # ---- loop de treino ----
        # schedules em épocas (não em passos): sobrevivem a mudança de batch/dataset
        steps_per_epoch = len(dl_train)
        beta_warm_steps = cfg["beta_warm_epochs"] * steps_per_epoch
        print(f"steps_per_epoch: {steps_per_epoch} | warmup KL até ep {cfg['beta_warm_epochs']} "
              f"| GAN entra na ep {cfg['disc_start_epoch']}")
        wandb.log({"steps_per_epoch": steps_per_epoch})

        beta = 0.0
        eps_nan = 0  # épocas seguidas com l1 NaN (1 batch ruim o scaler resgata; 2 épocas não)
        for ep in range(start_ep, cfg["epochs"]):
            enc.train(); dec.train(); disc.train()
            soma, nb = {}, 0
            kl_dim_soma = torch.zeros(cfg["zdim"], device=device)
            use_gan = ep >= cfg["disc_start_epoch"]
            for x, _ in dl_train:
                x = x.to(device)
                beta = cfg["beta"] * min(1.0, step / max(1, beta_warm_steps))  # warmup do KL

                with torch.amp.autocast(device, enabled=use_amp):
                    mu, lv = enc(x)
                    # clamp: lv.exp() em fp16 estoura (->inf->NaN) se lv passar de ~11;
                    # sigma em [e-5, e5] cobre qualquer posterior útil
                    lv = lv.clamp(-10, 10)
                    rec = dec(reparam(mu, lv))

                # passo do discriminador
                if use_gan:
                    optD.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device, enabled=use_amp):
                        loss_d = d_hinge(disc(x), disc(rec.detach()))
                    scaler_d.scale(loss_d).backward()
                    scaler_d.step(optD)
                    scaler_d.update()
                else:
                    loss_d = torch.zeros((), device=device)

                # passo do gerador (enc + dec)
                optG.zero_grad(set_to_none=True)
                with torch.amp.autocast(device, enabled=use_amp):
                    l1 = F.l1_loss(rec, x)
                    lp = perc(rec, x) if perc is not None else torch.zeros((), device=device)
                    kl = kl_div(mu.float(), lv.float())  # KL em fp32: exp/quadrado fora do fp16
                    ga = g_hinge(disc(rec)) if use_gan else torch.zeros((), device=device)
                    loss_g = l1 + cfg["l_perc"] * lp + beta * kl + cfg["l_adv"] * ga
                scaler_g.scale(loss_g).backward()
                scaler_g.step(optG)
                scaler_g.update()

                for k, v in dict(l1=l1, perc=lp, kl=kl, adv=ga, d=loss_d).items():
                    soma[k] = soma.get(k, 0.0) + float(v.detach())
                # KL por dimensão: sinal direto de colapso (dim "morta" -> kl ~ 0)
                with torch.no_grad():
                    kl_dim_soma += -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean(0).float()
                nb += 1
                step += 1

            nb = max(1, nb)
            l1m = soma["l1"] / nb
            percm = soma["perc"] / nb
            klm = soma["kl"] / nb
            advm = soma["adv"] / nb
            dm = soma["d"] / nb
            # dim "ativa" = KL médio > 0.01 nat (limiar usual na literatura beta-VAE)
            dims_ativas = int((kl_dim_soma / nb > 0.01).sum())

            print(f"ep {ep + 1:03d}/{cfg['epochs']} | l1 {l1m:.3f} perc {percm:.3f} "
                  f"kl {klm:.3f} adv {advm:.3f} d {dm:.3f} beta {beta:.3f} "
                  f"| dims ativas {dims_ativas}/{cfg['zdim']}")

            # NaN é terminal: aborta e libera a GPU pra próxima receita do sweep
            eps_nan = eps_nan + 1 if l1m != l1m else 0
            if eps_nan >= 2:
                raise RuntimeError(f"l1 NaN por {eps_nan} épocas seguidas (ep {ep + 1}): "
                                   "treino divergiu, abortando o run")

            # validação: fidelidade em imagens não vistas — métrica objetivo do sweep
            val_l1 = self._validar(enc, dec, dl_val, device)
            if val_l1 is not None:
                print(f"          val_l1 {val_l1:.3f}")

            # log DIRETO, exatamente como em classification.py
            wandb.log({"l1": l1m, "perc": percm, "kl": klm, "adv": advm,
                       "d": dm, "beta": beta, "dims_ativas": dims_ativas,
                       "val_l1": val_l1, "epoch": ep})

            # visuais + gate: a cada 10 épocas e no fim (call separado)
            if (ep + 1) % 10 == 0 or ep == cfg["epochs"] - 1:
                wandb.log(self._avaliar(enc, dec, xs_h, xs_s, device))

            # checkpoint periódico ("epoch" = próxima época a rodar no resume)
            if cfg.get("ckpt_every", 0) and (ep + 1) % cfg["ckpt_every"] == 0:
                self._salvar_ckpt(ckpt_path, enc, dec, disc, optG, optD,
                                  scaler_g, scaler_d, ep + 1, step, cfg)

        # ---- salva e devolve o caminho (o chassi sobe como artefato) ----
        caminho = os.path.join("outputs", f"vae_gan{rid}.pt")
        torch.save({"enc": _unwrap(enc).state_dict(), "dec": _unwrap(dec).state_dict(),
                    "disc": _unwrap(disc).state_dict(), "hparams": dict(cfg)}, caminho)
        print(f"Modelo salvo em: {caminho}")
        return caminho

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    @staticmethod
    def _salvar_ckpt(path, enc, dec, disc, optG, optD, scaler_g, scaler_d, epoch, step, cfg):
        """Estado completo pro resume: pesos, optimizers, RNG e posição do treino.

        RNG (python/numpy/torch/cuda) garante que o run retomado seja idêntico ao
        ininterrupto (reparam e shuffle do dataloader). wandb_id permite retomar o
        mesmo run nas curvas. O VGG perceptual não entra (congelado, reconstrói).
        """
        torch.save({
            "enc": _unwrap(enc).state_dict(), "dec": _unwrap(dec).state_dict(),
            "disc": _unwrap(disc).state_dict(),
            "optG": optG.state_dict(), "optD": optD.state_dict(),
            "scaler_g": scaler_g.state_dict() if scaler_g.is_enabled() else None,
            "scaler_d": scaler_d.state_dict() if scaler_d.is_enabled() else None,
            "epoch": epoch, "step": step, "hparams": dict(cfg),
            "wandb_id": wandb.run.id if wandb.run else None,
            "rng": {
                "py": random.getstate(), "np": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }, path)
        print(f"checkpoint salvo em {path} (época {epoch})")

    @staticmethod
    @torch.no_grad()
    def _validar(enc, dec, dl_val, device):
        """L1 médio de reconstrução no val set (recon via mu: determinística)."""
        if dl_val is None:
            return None
        enc.eval(); dec.eval()
        soma, n = 0.0, 0
        for x, _ in dl_val:
            x = x.to(device)
            soma += float(F.l1_loss(dec(enc(x)[0]), x)) * x.size(0)
            n += x.size(0)
        enc.train(); dec.train()
        return soma / max(1, n)

    # ------------------------------------------------------------------
    # Avaliação (gate + visuais)
    # ------------------------------------------------------------------
    @staticmethod
    def _fixos(train, cfg, device):
        """Conjuntos fixos de até 8 saudáveis e 8 doentes (sempre os mesmos)."""
        saud = [it for it in train if it["label"] == 0][:8]
        doente = [it for it in train if it["label"] == 1][:8]
        if not saud or not doente:
            return None, None  # falta uma das classes -> sem gate/visuais
        ds = _ImagensXRay(saud + doente, cfg["img_size"], cfg["channels"])
        xs_h = torch.stack([ds[i][0] for i in range(len(saud))]).to(device)
        xs_s = torch.stack([ds[i][0] for i in range(len(saud), len(saud) + len(doente))]).to(device)
        return xs_h, xs_s

    @staticmethod
    @torch.no_grad()
    def _avaliar(enc, dec, xs_h, xs_s, device):
        """Gate (colapso médio sobre pares) + visuais (recon e interpolação)."""
        if xs_h is None or xs_s is None:
            return {}
        enc.eval(); dec.eval()
        out = {}

        # GATE — colapso: L1 médio entre recon(saudável) e recon(doente) sobre os pares.
        # Perto de 0 => decoder ignora z (colapsou). Robustez vem da média nos pares.
        # Pareia até o mínimo (as duas classes podem ter contagens diferentes).
        n = min(xs_h.size(0), xs_s.size(0))
        rec_h = dec(enc(xs_h[:n])[0])   # mu (média do posterior): recon determinística
        rec_s = dec(enc(xs_s[:n])[0])   # gate estável e comparável entre épocas
        out["colapso"] = float(F.l1_loss(rec_h, rec_s))

        # VISUAL — reconstrução: 4 saudáveis + 4 doentes; entrada (cima) vs recon (baixo)
        x = torch.cat([xs_h[:4], xs_s[:4]])
        rec = dec(enc(x)[0])            # mu: reconstrução sem ruído de amostragem
        g = vutils.make_grid(torch.cat([x, rec]), nrow=8, normalize=True, value_range=(-1, 1))
        out["recon"] = wandb.Image(g.permute(1, 2, 0).cpu().numpy())

        # VISUAL — interpolação: 3 pares saudável->doente, 7 passos (o produto da tese)
        n_pares = min(3, xs_h.size(0), xs_s.size(0))
        linhas = []
        for i in range(n_pares):
            za = enc(xs_h[i:i + 1])[0]  # mu das pontas: caminho limpo e reprodutível
            zb = enc(xs_s[i:i + 1])[0]
            for t in torch.linspace(0, 1, 7, device=device):
                linhas.append(dec(slerp(za, zb, t)))  # esférica: fica na casca de raio ~sqrt(d)
        gi = vutils.make_grid(torch.cat(linhas), nrow=7, normalize=True, value_range=(-1, 1))
        out["interp"] = wandb.Image(gi.permute(1, 2, 0).cpu().numpy())

        enc.train(); dec.train()
        return out
