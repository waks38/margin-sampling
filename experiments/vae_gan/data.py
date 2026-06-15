# -*- coding: utf-8 -*-
"""Dados do experimento VAE-GAN: importar + tratar + carregar.

  importar(root, ...) -> lista de {path, label, grupo}    (a parte que muda por origem)
  tratar(itens)       -> (train, val, test)               (divide sem misturar grupo)
  carregar(...)       -> (train_loader, val_loader, test_loader)  (vira lotes pro modelo)

A ORIGEM dos dados é dirigida por config (não mais hardcoded), pra mesma fábrica
servir vários datasets:
  - classes:    mapa {nome_da_classe: rótulo}        (default: raio-X NORMAL/PNEUMONIA)
  - label_from: "folder" (classe = pasta-pai) | "filename" (classe = prefixo do nome)
  - group_by:   "patient" (agrupa pra não vazar paciente) | "image" (cada foto é única)
"""
import re
import random
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

# Defaults = raio-X, pra o bloco `vae_gan` antigo continuar idêntico sem novos campos.
DEFAULT_CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}
EXTS = {".jpeg", ".jpg", ".png"}


def _classe_de(arquivo: Path, label_from: str) -> str:
    """Nome da classe (UPPER) pela origem escolhida.

    folder   -> pasta-pai (raio-X: NORMAL/PNEUMONIA).
    filename -> prefixo do nome até o 1o ponto (cats/dogs: cat.0.jpg -> CAT).
    """
    if label_from == "filename":
        return arquivo.stem.split(".")[0].upper()
    return arquivo.parent.name.upper()


def _grupo(arquivo: Path, classe: str, group_by: str) -> str:
    """Chave de agrupamento pro split sem vazamento.

    image   -> cada imagem é seu próprio grupo: split fica disjunto POR IMAGEM.
    patient -> id do paciente pelo nome (raio-X). NORMAL não codifica paciente.
    """
    if group_by == "image":
        return str(arquivo)
    if classe == "NORMAL":
        return f"N::{arquivo.stem}"
    m = re.search(r"person(\d+)_(bacteria|virus)", arquivo.stem, re.IGNORECASE)
    return f"P::{m.group(1)}" if m else f"P::{arquivo.stem}"


def importar(root, classes=None, label_from="folder", group_by="patient") -> list:
    """Lista toda imagem sob `root`, inferindo classe e grupo pela config.
    (Única parte que sabe da origem dos dados — dataset novo? mexe só aqui/no config.)"""
    root = Path(root)
    classes = classes or DEFAULT_CLASSES
    classes_norm = {str(k).upper(): v for k, v in classes.items()}
    itens = []
    vistos = set()  # o zip do Kaggle pode ter cópia aninhada -> rglob veria 2x
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in EXTS:
            continue
        # lixo de zip do macOS (pasta __MACOSX / arquivos AppleDouble ._*): não são imagens
        if "__MACOSX" in f.parts or f.name.startswith("._"):
            continue
        classe = _classe_de(f, label_from)
        if classe not in classes_norm:
            continue
        # dedup por (classe, nome, tamanho): mesma imagem em caminhos diferentes conta 1x
        chave = (classe, f.name, f.stat().st_size)
        if chave in vistos:
            continue
        vistos.add(chave)
        itens.append({"path": str(f), "label": classes_norm[classe],
                      "grupo": _grupo(f, classe, group_by)})
    if not itens:
        raise RuntimeError(f"Nenhuma imagem encontrada em {root}")
    return itens


def tratar(itens: list, val_frac=0.15, test_frac=0.15, seed=42):
    """Divide em train/val/test POR GRUPO (sem vazamento). RNG local semeado
    com o seed UNIVERSAL (injetado pelo chassi) -> mesmo split reproduzível depois
    (ex.: pelo classificador). group_by='image' => grupo = imagem => split por imagem."""
    grupos = {}
    for it in itens:
        grupos.setdefault(it["grupo"], []).append(it)

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
    """Abre a imagem, redimensiona e normaliza p/ [-1,1] (casa com o tanh do decoder).
    (nome histórico; serve qualquer dataset de imagem — grayscale ou RGB)."""

    def __init__(self, itens, img_size, channels=1):
        self.itens = itens
        self.channels = channels
        passos = []
        if channels == 1:
            passos.append(T.Grayscale(1))
        passos += [
            # preserva a proporção (lado menor -> img_size) e corta o centro,
            # em vez de esticar pro quadrado e distorcer a imagem
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
