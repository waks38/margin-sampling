from experiments.classification import ClassificationExperiment

# Registry: mapeia o nome (do config) -> a classe do experimento.
# Para adicionar um experimento novo: importe-o acima e registre aqui.
EXPERIMENTS = {
    "classification": ClassificationExperiment,
}


def build_experiment(name: str, cfg: dict):
    """Recebe o nome do experimento e o cfg completo; entrega a instancia com SEU bloco."""
    if name not in EXPERIMENTS:
        disponiveis = ", ".join(EXPERIMENTS.keys())
        raise ValueError(f"Experimento '{name}' nao registrado. Disponiveis: {disponiveis}")
    bloco = cfg[name]
    return EXPERIMENTS[name](bloco)