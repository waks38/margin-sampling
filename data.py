import torch
from torch.utils.data import TensorDataset, DataLoader


def make_data(cfg: dict) -> DataLoader:
    """Cria um dataset sintético, porém com um padrão aprendível."""
    X = torch.randn(cfg["num_samples"], cfg["input_size"])
    true_weights = torch.randn(cfg["input_size"], cfg["num_classes"])
    y = (X @ true_weights).argmax(dim=1)   # rótulo depende de X -> dá pra aprender
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True) 