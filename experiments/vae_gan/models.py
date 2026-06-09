# -*- coding: utf-8 -*-
"""Modelos do experimento VAE-GAN: Encoder, Decoder, PatchDiscriminator, VGGPerceptual."""
import torch.nn as nn
import torch
import torch.nn.functional as F


def conv_bn(cin, cout, s=2):
    """Downsample: conv stride-2 + BN + LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, s, 1),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, True),
    )


def deconv_bn(cin, cout):
    """Upsample: conv transposta + BN + ReLU."""
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, 2, 1),
        nn.BatchNorm2d(cout),
        nn.ReLU(True),
    )


def reparam(mu, logvar):
    """Truque da reparametrizacao: z = mu + sigma * eps, eps ~ N(0,1).

    Amostrar z ~ N(mu, sigma^2) nao e' derivavel (gradiente nao passa por
    sorteio). Tiramos a aleatoriedade pra fora: eps fixo, z deterministico
    em funcao de mu e sigma -> gradiente flui.
    """
    sigma = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)
    return mu + sigma * eps




class Encoder(nn.Module):
    """Projeta a imagem em (mu, logvar) de tamanho zdim.

    Args:
        in_ch: canais de entrada (1=grayscale, 3=RGB).
        base: largura base dos filtros (dobra a cada camada).
        zdim: dimensao do espaco latente.
        img_size: resolucao da imagem de entrada (multiplo de 32).
    """

    def __init__(self, in_ch: int, base: int, zdim: int, img_size: int):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(in_ch, base),
            conv_bn(base, base * 2),
            conv_bn(base * 2, base * 4),
            conv_bn(base * 4, base * 8),
            conv_bn(base * 8, base * 8),
        )
        self.spatial = img_size // 32
        self.flat = base * 8 * self.spatial * self.spatial
        self.fc_mu = nn.Linear(self.flat, zdim)
        self.fc_lv = nn.Linear(self.flat, zdim)

    def forward(self, x):
        h = self.body(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)


class Decoder(nn.Module):
    """z -> imagem de saida, terminando em tanh ([-1,1]).

    Args:
        out_ch: canais de saida (deve casar com in_ch do Encoder).
        base: largura base dos filtros.
        zdim: dimensao do espaco latente.
        img_size: resolucao da imagem de saida (multiplo de 32).
    """

    def __init__(self, out_ch: int, base: int, zdim: int, img_size: int):
        super().__init__()
        self.base = base
        self.spatial = img_size // 32
        self.fc = nn.Linear(zdim, base * 8 * self.spatial * self.spatial)
        self.body = nn.Sequential(
            deconv_bn(base * 8, base * 8),
            deconv_bn(base * 8, base * 4),
            deconv_bn(base * 4, base * 2),
            deconv_bn(base * 2, base),
            nn.ConvTranspose2d(base, out_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.body(self.fc(z).view(-1, self.base * 8, self.spatial, self.spatial))


class PatchDiscriminator(nn.Module):
    """PatchGAN: julga o realismo por REGIAO (devolve um mapa de scores, nao um
    unico numero). Bom pra textura local. Sem sigmoid: o hinge usa o logit cru.

    Args:
        in_ch: canais de entrada (deve casar com o decoder).
        base: largura base dos filtros.
    """

    def __init__(self, in_ch: int, base: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, True),                                  # sem BN na 1a camada
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.BatchNorm2d(base * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.BatchNorm2d(base * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 4, base * 8, 4, 1, 1),
            nn.BatchNorm2d(base * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 8, 1, 4, 1, 1),                          # mapa de scores (logits)
        )

    def forward(self, x):
        return self.net(x)


class VGGPerceptual(nn.Module):
    """Perda perceptual: compara ativacoes de uma VGG16 pre-treinada (ImageNet).

    Imagens chegam em [-1,1] (grayscale, no raio-X). A VGG espera 3 canais em
    [0,1] normalizados pelo ImageNet, entao convertemos antes. Pesos congelados.
    Baixa ~500MB de pesos na 1a instanciacao.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        feats = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        # blocos ate relu1_2, relu2_2, relu3_3
        self.blocks = nn.ModuleList([feats[:4], feats[4:9], feats[9:16]])
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, img):
        img = (img + 1) / 2                       # [-1,1] -> [0,1]
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)          # grayscale -> 3 canais
        return (img - self.mean) / self.std       # normalizacao ImageNet

    def forward(self, rec, x):
        rec, x = self._prep(rec), self._prep(x)
        loss = 0.0
        for b in self.blocks:
            rec, x = b(rec), b(x)
            loss = loss + F.l1_loss(rec, x)
        return loss
