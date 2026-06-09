import yaml
import wandb

from seeds import set_seed
from experiments import build_experiment


def load_config(path: str = "config.yaml") -> dict:
    """Lê o arquivo de configuração."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()

    for nome in cfg["run"]:
        print(f"\n=== Rodando experimento: {nome} ===")
        set_seed(cfg["seed"])
        # loga o bloco do experimento + o seed universal (auditabilidade)
        wandb.init(project="margin-sampling", name=nome,
                   config={**cfg[nome], "seed": cfg["seed"]})

        experiment = build_experiment(nome, cfg)
        try:
            caminho = experiment.run()

            # Sobe o artefato gerado pro wandb (persiste na nuvem)
            artifact = wandb.Artifact(name=nome, type="model")
            artifact.add_file(caminho)
            wandb.log_artifact(artifact)
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
