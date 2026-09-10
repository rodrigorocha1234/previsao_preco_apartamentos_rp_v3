from typing import Any, NamedTuple, Protocol, runtime_checkable
import numpy as np
import pandas as pd


class DadosProcessados(NamedTuple):
    """Estrutura imutável contendo os dados divididos e pré-processados."""
    x_treino: np.ndarray
    x_teste: np.ndarray
    y_treino: np.ndarray
    y_teste: np.ndarray


@runtime_checkable
class IScaler(Protocol):
    """Protocolo que declara a interface esperada de um transformador/scaler do scikit-learn."""

    def fit_transform(self, X: Any, y: Any = None, *args: Any, **kwargs: Any) -> Any:
        ...

    def transform(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class IPreprocessador(Protocol):
    """Protocolo que define a interface esperada de um pré-processador de dados."""

    base: pd.DataFrame | None

    @property
    def colunas_features(self) -> list[str]:
        ...

    def realizar_preprocessamento(self, base: pd.DataFrame | None = None) -> DadosProcessados:
        ...
