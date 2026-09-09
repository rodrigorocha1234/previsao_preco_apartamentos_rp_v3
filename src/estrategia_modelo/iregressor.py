from typing import Protocol, Self
import numpy as np


class IRegressor(Protocol):
    """Protocolo que declara a interface esperada de um regressor (fit e predict)."""
    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        ...

    def predict(self, x: np.ndarray) -> np.ndarray:
        ...


# Alias para compatibilidade
RegressorProtocol = IRegressor
