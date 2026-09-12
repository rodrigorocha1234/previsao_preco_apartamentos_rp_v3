from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from .iregressor import IRegressor


@dataclass
class ResultadoGridSearch:
    """Resultado detalhado do processo de Grid Search."""
    melhores_parametros: dict[str, Any]
    melhor_score: float
    melhor_modelo: IRegressor
    tabela_resultados: pd.DataFrame
    grid: GridSearchCV

    def __iter__(self) -> Iterator[Any]:
        """Permite desempacotamento retrocompatível: `params, score = realizar_grid_search(...)`."""
        yield self.melhores_parametros
        yield self.melhor_score


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

    def realizar_grid_search(
        self,
        x_completo: np.ndarray,
        y_completa: np.ndarray,
        cv: int | Any | None = None,
    ) -> ResultadoGridSearch:
        """Executa a busca em grade de hiperparâmetros utilizando GridSearchCV.

        Args:
            x_completo: Matriz de features para ajuste do grid.
            y_completa: Vetor target para ajuste do grid.
            cv: Quantidade de folds ou splitter de validação cruzada do GridSearchCV.

        Returns:
            ResultadoGridSearch contendo melhores parâmetros, melhor score, melhor estimador,
            tabela completa com todas as combinações e a instância do GridSearchCV.
        """
        modelo = self.modelo
        n_amostras = len(x_completo)
        cv_final = cv if cv is not None else (min(5, max(2, n_amostras // 2)) if n_amostras >= 4 else 2)
        grid = GridSearchCV(modelo, self.params, cv=cv_final)
        grid.fit(x_completo, y_completa)
        melhores_parametros: dict[str, Any] = grid.best_params_
        melhores_resultados: float = float(grid.best_score_)
        df_resultados: pd.DataFrame = pd.DataFrame(grid.cv_results_)

        return ResultadoGridSearch(
            melhores_parametros=melhores_parametros,
            melhor_score=melhores_resultados,
            melhor_modelo=grid.best_estimator_,
            tabela_resultados=df_resultados,
            grid=grid,
        )

