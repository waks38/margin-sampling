import os

import yaml
import wandb

from seeds import set_seed
from experiments import build_experiment


def load_config(path: str = "config.yaml") -> dict:
    """Lê o arquivo de configuração."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def resume_id(nome: str, cfg: dict):
    """Id do run wandb gravado no checkpoint, se o experimento for retomar.

    Permite continuar as curvas no mesmo run em vez de abrir um novo.
    """
    ckpt = os.path.join("outputs", f"{nome}_ckpt.pt")
    if not (cfg.get("resume", False) and os.path.exists(ckpt)):
        return None
    import torch
    # weights_only=False: o ckpt tem estados de RNG (objetos python/numpy)
    return torch.load(ckpt, map_location="cpu", weights_only=False).get("wandb_id")


def nome_artefato(nome: str, c: dict) -> str:
    """Nome do artefato codifica a receita: num sweep cada run gera um modelo
    diferente e um nome fixo sobrescreveria todos na mesma linhagem."""
    chaves = ["zdim", "beta", "l_adv", "l_perc"]
    sufixo = "_".join(f"{k}{c[k]}" for k in chaves if k in c)
    return f"{nome}_{sufixo}" if sufixo else nome


def main() -> None:
    cfg = load_config()

    for nome in cfg["run"]:
        print(f"\n=== Rodando experimento: {nome} ===")
        rid = resume_id(nome, cfg[nome])
        # loga o bloco do experimento + o seed universal (auditabilidade)
        wandb.init(project="margin-sampling", name=nome,
                   config={**cfg[nome], "seed": cfg["seed"]},
                   id=rid, resume="allow" if rid else None)

        # Sweep: o agente injeta overrides em wandb.config por cima do yaml.
        # wandb.config é a fonte da verdade daqui pra frente (run normal: == yaml).
        efetivo = dict(wandb.config)
        cfg["seed"] = efetivo.pop("seed", cfg["seed"])
        cfg[nome] = efetivo
        set_seed(cfg["seed"])

        experiment = build_experiment(nome, cfg)
        try:
            caminho = experiment.run()

            # Sobe o artefato gerado pro wandb (persiste na nuvem)
            artifact = wandb.Artifact(name=nome_artefato(nome, efetivo), type="model")
            artifact.add_file(caminho)
            wandb.log_artifact(artifact)
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
