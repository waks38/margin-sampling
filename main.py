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
        wandb.init(project="margin-sampling", name=nome, config=cfg[nome])

        experiment = build_experiment(nome, cfg)
        experiment.run()

        wandb.finish()


if __name__ == "__main__":
    main()