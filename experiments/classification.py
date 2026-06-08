import os
import torch
import wandb

from experiments.base import Experiment
from models import build_model
from losses import build_loss
from data import make_data

class ClassificationExperiment(Experiment):
    """Experimento de classificação supervisionada."""

    def run(self):
        cfg = self.cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Rodando em: {device}")

        # Monta as peças a partir do config (via registries)
        loader = make_data(cfg)
        model = build_model(
            cfg["model"],
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_classes=cfg["num_classes"],
        ).to(device)
        criterion = build_loss(cfg["loss"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

        # Loop de treino
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X.size(0)
                correct += (outputs.argmax(dim=1) == y).sum().item()
                total += y.size(0)

            avg_loss = total_loss / total
            acc = correct / total
            print(f"Epoch {epoch:2d}/{cfg['epochs']} | loss: {avg_loss:.4f} | acc: {acc:.4f}")
            wandb.log({"loss": avg_loss, "acc": acc, "epoch": epoch})

        # Salva o modelo treinado em disco e devolve o caminho ao chassi
        os.makedirs("outputs", exist_ok=True)
        caminho = os.path.join("outputs", "classification_model.pt")
        torch.save(model.state_dict(), caminho)
        print(f"Modelo salvo em: {caminho}")
        return caminho