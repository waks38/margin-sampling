# -*- coding: utf-8 -*-
"""Losses generativas do VAE-GAN.

São funções (não classes de registry): o KL depende de (mu, logvar) e o hinge
dos scores do discriminador, então não cabem numa interface criterion(out, y).
"""
import torch
import torch.nn.functional as F


def kl_div(mu, logvar):
    """KL entre N(mu, sigma^2) e N(0, I): soma nas dims do latente, media no batch.

    Convencao padrao do VAE — o KL escala com zdim, entao beta no config esta
    na escala da literatura (beta-VAE). Antes era media em todas as dims, o que
    deixava o beta efetivo ~zdim x mais fraco do que o config sugeria.
    """
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def d_hinge(d_real, d_fake):
    """Hinge loss do discriminador: empurra real p/ >+1 e fake p/ <-1."""
    return torch.mean(F.relu(1.0 - d_real)) + torch.mean(F.relu(1.0 + d_fake))


def g_hinge(d_fake):
    """Hinge loss do gerador: quer que o discriminador pontue alto no fake."""
    return -torch.mean(d_fake)
