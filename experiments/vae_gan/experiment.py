# -*- coding: utf-8 -*-
"""Experimento VAE-GAN (raio-X): VAE com discriminador GAN + perda perceptual.

run(): importar -> tratar -> carregar -> treinar -> salvar -> devolve o caminho.
Loss G: l1 + l_perc*perceptual + beta*KL + l_adv*adversarial.
beta sobe por warmup; o GAN só entra após disc_start passos. Sem resume.
Loga tudo no fim e devolve o caminho do modelo (o chassi sobe como artefato).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from experiments.base import Experiment
from experiments.vae_gan.data import importar, tratar, carregar
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
        historico = []
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
            m = {k: v / nb for k, v in soma.items()}
            m.update(epoch=ep, beta=beta)
            historico.append(m)
            print(f"ep {ep + 1:03d}/{cfg['epochs']} | l1 {m['l1']:.3f} perc {m['perc']:.3f} "
                  f"kl {m['kl']:.3f} adv {m['adv']:.3f} d {m['d']:.3f} beta {beta:.2f}")

        # ---- loga TUDO no fim (curvas completas) ----
        for m in historico:
            wandb.log(m, step=m["epoch"])

        # ---- salva e devolve o caminho (o chassi sobe como artefato) ----
        os.makedirs("outputs", exist_ok=True)
        caminho = os.path.join("outputs", "vae_gan.pt")
        torch.save({"enc": _unwrap(enc).state_dict(), "dec": _unwrap(dec).state_dict(),
                    "disc": _unwrap(disc).state_dict(), "hparams": dict(cfg)}, caminho)
        print(f"Modelo salvo em: {caminho}")
        return caminho
