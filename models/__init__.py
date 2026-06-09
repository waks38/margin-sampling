from models.simple_net import SimpleNet

# Registry: mapeia o nome (string do config) -> a classe da arquitetura.
# Para adicionar um modelo novo: importe-o acima e registre aqui embaixo.
MODELS = {
    "simple_net": SimpleNet,
}


def build_model(name: str, **kwargs):
    """Recebe o nome vindo do config e devolve a instância do modelo."""
    if name not in MODELS:
        disponiveis = ", ".join(MODELS.keys())
        raise ValueError(f"Modelo '{name}' nao registrado. Disponiveis: {disponiveis}")
    return MODELS[name](**kwargs)