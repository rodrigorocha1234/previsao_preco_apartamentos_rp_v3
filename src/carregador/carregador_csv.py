from dataclasses import dataclass
import pandas as pd

from .icarregador import ICarregador


@dataclass
class CarregadorCSV(ICarregador[pd.DataFrame]):

    caminho: str
    atributos: list[str] | None = None

    def carregar(self) -> pd.DataFrame:

        return pd.read_csv(
            self.caminho,
            names=self.atributos
        )