import yaml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import wandb
from seeds import set_seed


def load_config(path: str = "config.yaml") -> dict:
    """Lê os hiperparâmetros do arquivo de config."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def make_data(cfg: dict) -> DataLoader:
    """Cria um dataset sintético, porém com um padrão aprendível."""
    X = torch.randn(cfg["num_samples"], cfg["input_size"])
    true_weights = torch.randn(cfg["input_size"], cfg["num_classes"])
    y = (X @ true_weights).argmax(dim=1)   # rótulo depende de X -> dá pra aprender
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)


class SimpleNet(nn.Module):
    """Rede mínima: entrada -> camada escondida -> classes."""
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train(cfg: dict) -> None:
    wandb.init(project='margin-sampling',config=cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando em: {device}")

    loader = make_data(cfg)
    model = SimpleNet(cfg["input_size"], cfg["hidden_size"], cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

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
        print(f"Epoch {epoch:2d}/{cfg['epochs']} | loss: {total_loss/total:.4f} | acc: {correct/total:.4f}")
        wandb.log({"loss": total_loss/total, "acc": correct/total, "epoch": epoch})

def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    train(cfg)


if __name__ == "__main__":
    main()


