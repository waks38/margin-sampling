# -*- coding: utf-8 -*-
"""Dados do experimento VAE-GAN (raio-X): importar + tratar + carregar.

  importar(root) -> lista de {path, label, paciente}   (a parte que muda por origem)
  tratar(itens)  -> (train, val, test)                 (divide sem misturar paciente)
  carregar(...)  -> (train_loader, val_loader, test_loader)  (vira lotes pro modelo)
"""
import re
import random
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

# Únicas suposições sobre a ORIGEM ficam aqui. Mudou a origem? Mexe só nisto.
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
    vistos = set()  # o zip do Kaggle tem cópia aninhada chest_xray/chest_xray -> rglob veria 2x
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in EXTS:
            continue
        # lixo de zip do macOS (pasta __MACOSX / arquivos AppleDouble ._*): não são imagens
        if "__MACOSX" in f.parts or f.name.startswith("._"):
            continue
        classe = f.parent.name.upper()
        if classe not in CLASSES:
            continue
        # dedup por (classe, nome, tamanho): mesma imagem em caminhos diferentes conta 1x
        chave = (classe, f.name, f.stat().st_size)
        if chave in vistos:
            continue
        vistos.add(chave)
        itens.append({"path": str(f), "label": CLASSES[classe],
                      "paciente": _paciente(f, classe)})
    if not itens:
        raise RuntimeError(f"Nenhuma imagem encontrada em {root}")
    return itens


def tratar(itens: list, val_frac=0.15, test_frac=0.15, seed=42):
    """Divide em train/val/test POR PACIENTE (sem vazamento). RNG local semeado
    com o seed UNIVERSAL (injetado pelo chassi) -> mesmo split reproduzível depois
    (ex.: pelo classificador)."""
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


class _ImagensXRay(Dataset):
    """Abre a imagem, redimensiona e normaliza p/ [-1,1] (casa com o tanh do decoder)."""

    def __init__(self, itens, img_size, channels=1):
        self.itens = itens
        self.channels = channels
        passos = []
        if channels == 1:
            passos.append(T.Grayscale(1))
        passos += [
            # preserva a proporção do raio-X (lado menor -> img_size) e corta o
            # centro, em vez de esticar pro quadrado e distorcer a anatomia
            T.Resize(img_size),
            T.CenterCrop(img_size),
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
