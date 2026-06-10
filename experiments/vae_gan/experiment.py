# -*- coding: utf-8 -*-
"""Experimento VAE-GAN (raio-X): VAE com discriminador GAN + perda perceptual.

run(): importar -> tratar -> carregar -> treinar -> avaliar -> salvar.
Loss G: l1 + l_perc*perceptual + beta*KL + l_adv*adversarial.
beta sobe por warmup; o GAN só entra após disc_start passos. Sem resume.

Logging W&B:
- escalares (l1, perc, kl, adv, d, beta) -> toda época;
- visuais (recon + interpolação) e gate (colapso) -> a cada 10 épocas e no fim.
Devolve o caminho do modelo (o chassi sobe como artefato).
"""
import os

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


class VaeGan(Experiment):
    """Treina um VAE com discriminador GAN (PatchGAN) e perda perceptual (VGG)."""

    def run(self) -> str:
        cfg = self.cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Rodando em: {device}")

        # ---- dados ---- (seed = universal, injetado pelo chassi)
        itens = importar(cfg["data_root"])
        train, val, test = tratar(itens, cfg["val_frac"], cfg["test_frac"], cfg["seed"])
        dl_train, _, _ = carregar(train, [], [], cfg["img_size"], cfg["batch_size"],
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

        # ---- loop de treino ----
        step = 0
        beta = 0.0
        for ep in range(cfg["epochs"]):
            enc.train(); dec.train(); disc.train()
            soma, nb = {}, 0
            for x, _ in dl_train:
                x = x.to(device)
                beta = cfg["beta"] * min(1.0, step / max(1, cfg["beta_warm"]))  # warmup do KL
                use_gan = step >= cfg["disc_start"]

                mu, lv = enc(x)
                rec = dec(reparam(mu, lv))

                # passo do discriminador
                if use_gan:
                    optD.zero_grad(set_to_none=True)
                    loss_d = d_hinge(disc(x), disc(rec.detach()))
                    loss_d.backward()
                    optD.step()
                else:
                    loss_d = torch.zeros((), device=device)

                # passo do gerador (enc + dec)
                optG.zero_grad(set_to_none=True)
                l1 = F.l1_loss(rec, x)
                lp = perc(rec, x) if perc is not None else torch.zeros((), device=device)
                kl = kl_div(mu, lv)
                ga = g_hinge(disc(rec)) if use_gan else torch.zeros((), device=device)
                loss_g = l1 + cfg["l_perc"] * lp + beta * kl + cfg["l_adv"] * ga
                loss_g.backward()
                optG.step()

                for k, v in dict(l1=l1, perc=lp, kl=kl, adv=ga, d=loss_d).items():
                    soma[k] = soma.get(k, 0.0) + float(v.detach())
                nb += 1
                step += 1

            nb = max(1, nb)
            l1m = soma["l1"] / nb
            percm = soma["perc"] / nb
            klm = soma["kl"] / nb
            advm = soma["adv"] / nb
            dm = soma["d"] / nb

            print(f"ep {ep + 1:03d}/{cfg['epochs']} | l1 {l1m:.3f} perc {percm:.3f} "
                  f"kl {klm:.3f} adv {advm:.3f} d {dm:.3f} beta {beta:.2f}")

            # log DIRETO, exatamente como em classification.py
            wandb.log({"l1": l1m, "perc": percm, "kl": klm, "adv": advm,
                       "d": dm, "beta": beta, "epoch": ep})

            # visuais + gate: a cada 10 épocas e no fim (call separado)
            if (ep + 1) % 10 == 0 or ep == cfg["epochs"] - 1:
                wandb.log(self._avaliar(enc, dec, xs_h, xs_s, device))

        # ---- salva e devolve o caminho (o chassi sobe como artefato) ----
        os.makedirs("outputs", exist_ok=True)
        caminho = os.path.join("outputs", "vae_gan.pt")
        torch.save({"enc": _unwrap(enc).state_dict(), "dec": _unwrap(dec).state_dict(),
                    "disc": _unwrap(disc).state_dict(), "hparams": dict(cfg)}, caminho)
        print(f"Modelo salvo em: {caminho}")
        return caminho

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
        rec_h = dec(reparam(*enc(xs_h[:n])))
        rec_s = dec(reparam(*enc(xs_s[:n])))
        out["colapso"] = float(F.l1_loss(rec_h, rec_s))

        # VISUAL — reconstrução: 4 saudáveis + 4 doentes; entrada (cima) vs recon (baixo)
        x = torch.cat([xs_h[:4], xs_s[:4]])
        rec = dec(reparam(*enc(x)))
        g = vutils.make_grid(torch.cat([x, rec]), nrow=8, normalize=True, value_range=(-1, 1))
        out["recon"] = wandb.Image(g.permute(1, 2, 0).cpu().numpy())

        # VISUAL — interpolação: 3 pares saudável->doente, 7 passos (o produto da tese)
        n_pares = min(3, xs_h.size(0), xs_s.size(0))
        linhas = []
        for i in range(n_pares):
            za = reparam(*enc(xs_h[i:i + 1]))
            zb = reparam(*enc(xs_s[i:i + 1]))
            for t in torch.linspace(0, 1, 7, device=device):
                linhas.append(dec((1 - t) * za + t * zb))
        gi = vutils.make_grid(torch.cat(linhas), nrow=7, normalize=True, value_range=(-1, 1))
        out["interp"] = wandb.Image(gi.permute(1, 2, 0).cpu().numpy())

        enc.train(); dec.train()
        return out
