import torch
import torch.nn as nn
import torch.nn.functional as F

# Mesmo padrão de registry dos modelos, agora para funções de perda.
LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
}


def build_loss(name: str, **kwargs):
    """Recebe o nome vindo do config e devolve a instancia da loss."""
    if name not in LOSSES:
        disponiveis = ", ".join(LOSSES.keys())
        raise ValueError(f"Loss '{name}' nao registrada. Disponiveis: {disponiveis}")
    return LOSSES[name](**kwargs)


# ------------------------------------------------------------------
# Losses generativas (VAE-GAN). Sao funcoes, nao classes do registry:
# o KL depende de (mu, logvar) e o hinge dos scores do discriminador,
# entao nao cabem na interface build_loss(nome) -> criterion(out, y).
# ------------------------------------------------------------------
def kl_div(mu, logvar):
    """KL entre N(mu, sigma^2) e N(0, I), media no batch. Regulariza o latente."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def d_hinge(d_real, d_fake):
    """Hinge loss do discriminador: empurra real p/ >+1 e fake p/ <-1."""
    return torch.mean(F.relu(1.0 - d_real)) + torch.mean(F.relu(1.0 + d_fake))


def g_hinge(d_fake):
    """Hinge loss do gerador: quer que o discriminador pontue alto no fake."""
    return -torch.mean(d_fake)