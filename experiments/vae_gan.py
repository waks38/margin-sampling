# -*- coding: utf-8 -*-
"""Experimento VAE-GAN (raio-X), atômico: dados + loader + treino num arquivo só.

Fluxo do run(): importar -> tratar -> carregar -> treinar -> salvar.
- Dados: lista imagens de uma pasta e divide por PACIENTE (sem vazamento).
- Modelo: Encoder/Decoder (VAE) + PatchDiscriminator (GAN) + VGGPerceptual.
- Loss G: l1 + l_perc*perceptual + beta*KL + l_adv*adversarial.
- beta sobe por warmup; o GAN só entra após disc_start passos. Sem resume.
- Loga TUDO no fim (curvas completas) e devolve o caminho do modelo salvo.
"""
import os
import re
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import wandb

from experiments.base import Experiment
from models.vae_gan import Encoder, Decoder, PatchDiscriminator, VGGPerceptual, reparam
from losses import kl_div, d_hinge, g_hinge


# ==================================================================
# DADOS — importar (listar) + tratar (dividir sem vazamento)
# ==================================================================
CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}
EXTS = {".jpeg", ".jpg", ".png"}


def _paciente(arquivo: Path, classe: str) -> str:
    """Id do paciente pelo nome. NORMAL não codifica paciente -> imagem única."""
    if classe == "NORMAL":
        return f"N::{arquivo.stem}"
    m = re.search(r"person(\d+)_(bacteria|virus)", arquivo.stem, re.IGNORECASE)
    return f"P::{m.group(1)}" if m else f"P::{arquivo.stem}"


def importar(root) -> list:
    """Lista toda imagem sob `root`, inferindo a classe pela pasta-pai.
    (Única parte que sabe da origem dos dados — mudou a origem? mexe só aqui.)"""
    root = Path(root)
    itens = []
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in EXTS:
            continue
        classe = f.parent.name.upper()
        if classe not in CLASSES:
            continue
        itens.append({"path": str(f), "label": CLASSES[classe],
                      "paciente": _paciente(f, classe)})
    if not itens:
        raise RuntimeError(f"Nenhuma imagem encontrada em {root}")
    return itens


def tratar(itens: list, val_frac=0.15, test_frac=0.15, seed=42):
    """Divide em train/val/test POR PACIENTE (sem vazamento). Determinístico pelo
    seed -> o mesmo split pode ser reproduzido depois (ex.: pelo classificador)."""
    grupos = {}
    for it in itens:
        grupos.setdefault(it["paciente"], []).append(it)

    ids = sorted(grupos)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    fatias = {
        "test":  ids[:n_test],
        "val":   ids[n_test:n_test + n_val],
        "train": ids[n_test + n_val:],
    }
    return tuple([img for pid in fatias[s] for img in grupos[pid]]
                 for s in ("train", "val", "test"))


# ==================================================================
# LOADER — 3 conjuntos -> 3 DataLoaders (particular do experimento)
# ==================================================================
class _ImagensXRay(Dataset):
    """Abre a imagem, redimensiona e normaliza p/ [-1,1] (casa com o tanh do decoder)."""

    def __init__(self, itens, img_size, channels=1):
        self.itens = itens
        self.channels = channels
        passos = []
        if channels == 1:
            passos.append(T.Grayscale(1))
        passos += [
            T.Resize((img_size, img_size)),
            T.ToTensor(),                                      # -> [0,1]
            T.Normalize([0.5] * channels, [0.5] * channels),   # -> [-1,1]
        ]
        self.tf = T.Compose(passos)

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, i):
        it = self.itens[i]
        modo = "L" if self.channels == 1 else "RGB"
        return self.tf(Image.open(it["path"]).convert(modo)), it["label"]


def carregar(train, val, test, img_size=128, batch_size=16, channels=1, num_workers=0):
    """Recebe os 3 conjuntos (de tratar) e devolve 3 DataLoaders. Vazio -> None."""
    def loader(itens, treino):
        if not itens:
            return None
        ds = _ImagensXRay(itens, img_size, channels)
        return DataLoader(ds, batch_size=batch_size, shuffle=treino,
                          num_workers=num_workers, drop_last=treino)

    return loader(train, True), loader(val, False), loader(test, False)


# ==================================================================
# EXPERIMENTO
# ==================================================================
class VaeGan(Experiment):
    """Treina um VAE com discriminador GAN (PatchGAN) e perda perceptual (VGG)."""

    def run(self) -> str:
        cfg = self.cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Rodando em: {device}")

        # ---- dados ----
        itens = importar(cfg["data_root"])
        train, val, test = tratar(itens, cfg["val_frac"], cfg["test_frac"], cfg.get("seed", 42))
        dl_train, _, _ = carregar(train, [], [], cfg["img_size"], cfg["batch_size"],
                                  cfg["channels"], cfg.get("num_workers", 0))
        print(f"split -> train {len(train)} | val {len(val)} | test {len(test)} (treina só no train)")

        # ---- modelos ----
        ch = cfg["channels"]
        enc = Encoder(ch, cfg["base"], cfg["zdim"], cfg["img_size"]).to(device)
        dec = Decoder(ch, cfg["base"], cfg["zdim"], cfg["img_size"]).to(device)
        disc = PatchDiscriminator(ch, cfg["base"]).to(device)
        perc = VGGPerceptual().to(device) if cfg["l_perc"] > 0 else None  # None = sem perceptual

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
        torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(),
                    "disc": disc.state_dict(), "hparams": dict(cfg)}, caminho)
        print(f"Modelo salvo em: {caminho}")
        return caminho
