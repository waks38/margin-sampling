from experiments.classification import ClassificationExperiment
from experiments.vae_gan import VaeGan

# Registry: mapeia o nome (do config) -> a classe do experimento.
# Para adicionar um experimento novo: importe-o acima e registre aqui.
EXPERIMENTS = {
    "classification": ClassificationExperiment,
    "vae_gan": VaeGan,
}


def build_experiment(name: str, cfg: dict):
    """Recebe o nome do bloco (do config) e o cfg completo; entrega a instancia.

    O nome do BLOCO desamarra do nome da CLASSE: o bloco pode declarar
    `experiment: vae_gan` pra rodar o mesmo experimento em outro dataset
    (ex.: bloco `dogs` -> classe VaeGan). Sem o campo, usa o proprio nome.
    """
    classe = cfg[name].get("experiment", name)
    if classe not in EXPERIMENTS:
        disponiveis = ", ".join(EXPERIMENTS.keys())
        raise ValueError(f"Experimento '{classe}' nao registrado. Disponiveis: {disponiveis}")
    # injeta o seed UNIVERSAL no bloco (fonte única; cada experimento o recebe)
    bloco = {**cfg[name], "seed": cfg["seed"]}
    return EXPERIMENTS[classe](bloco)