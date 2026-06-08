from abc import ABC, abstractmethod


class Experiment(ABC):
    """Contrato base: todo experimento recebe um cfg e implementa run()."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def run(self) -> str:
        """Treina o modelo, salva em disco e retorna o caminho do arquivo."""
        ...