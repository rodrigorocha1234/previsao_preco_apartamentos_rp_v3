from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV

from .iregressor import IRegressor


class EstrategiaModelo(ABC):

    @property
    @abstractmethod
    def modelo(self) -> IRegressor:
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        ...

    def treinar_modelo_simples(self, x_treino: np.ndarray, y_treino: np.ndarray) -> IRegressor:
        return self.modelo.fit(x_treino, y_treino)


    def realizar_grid_search(self, x_completo: np.ndarray, y_completa: np.ndarray) -> tuple[dict[str, Any], float]:
        modelo = self.modelo
        grid = GridSearchCV(modelo, self.params)
        grid.fit(x_completo, y_completa)
        melhores_parametros: dict[str, Any] = grid.best_params_
        melhores_resultados: float = float(grid.best_score_)
        return melhores_parametros, melhores_resultados
