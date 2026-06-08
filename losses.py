import torch.nn as nn

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