import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixa a aleatoriedade em todas as fontes, para reprodutibilidade."""
    random.seed(seed)            # Python puro
    np.random.seed(seed)         # NumPy
    torch.manual_seed(seed)      # PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch (todas as GPUs, se houver)

    # Força o cuDNN a ser determinístico (resultado igual entre runs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
