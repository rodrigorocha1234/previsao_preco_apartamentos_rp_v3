from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, cross_validate

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


@dataclass
class ResultadoValidacaoCruzada:
    """Resultado detalhado do processo de Validação Cruzada com KFold e cross_validate."""
    scores_cv: dict[str, np.ndarray]
    media_test_r2: float
    std_test_r2: float
    media_test_mae: float
    std_test_mae: float
    media_test_rmse: float
    std_test_rmse: float
    media_train_r2: float
    std_train_r2: float
    n_splits: int
    tabela_folds: pd.DataFrame
    kfold: KFold

    def __iter__(self) -> Iterator[Any]:
        """Permite desempacotamento retrocompatível: `media_r2, std_r2 = realizar_validacao_cruzada(...)`."""
        yield self.media_test_r2
        yield self.std_test_r2


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

    def realizar_validacao_cruzada(
        self,
        x_completo: np.ndarray,
        y_completa: np.ndarray,
        kfold: KFold | int | None = None,
        scoring: dict[str, str] | None = None,
        return_train_score: bool = True,
    ) -> ResultadoValidacaoCruzada:
        """Executa a validação cruzada do modelo utilizando cross_validate e KFold.

        Args:
            x_completo: Matriz de features para a validação cruzada.
            y_completa: Vetor target para a validação cruzada.
            kfold: Instância de KFold ou número inteiro de folds (default: KFold com 5 splits e shuffle=True).
            scoring: Dicionário mapeando nomes para scorers do scikit-learn.
            return_train_score: Se deve computar métricas também no treino.

        Returns:
            ResultadoValidacaoCruzada com scores por fold, médias, desvios e tabela detalhada.
        """
        n_amostras = len(x_completo)
        if isinstance(kfold, KFold):
            kf = kfold
        elif isinstance(kfold, int):
            kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
        else:
            n_splits = min(5, max(2, n_amostras // 2)) if n_amostras >= 4 else 2
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        if scoring is None:
            scoring = {
                "r2": "r2",
                "neg_mae": "neg_mean_absolute_error",
                "neg_rmse": "neg_root_mean_squared_error",
            }

        estimador = clone(self.modelo)

        scores = cross_validate(
            estimator=estimador,
            X=x_completo,
            y=y_completa,
            cv=kf,
            scoring=scoring,
            return_train_score=return_train_score,
            n_jobs=-1,
        )

        test_r2 = np.asarray(scores.get("test_r2", np.array([])), dtype=float)
        train_r2 = np.asarray(scores.get("train_r2", np.array([])), dtype=float)
        test_mae = np.abs(np.asarray(scores.get("test_neg_mae", np.array([])), dtype=float))
        test_rmse = np.abs(np.asarray(scores.get("test_neg_rmse", np.array([])), dtype=float))

        linhas_folds = []
        for fold_idx in range(len(test_r2)):
            d_fold = {
                "fold": fold_idx + 1,
                "test_r2": float(test_r2[fold_idx]),
                "test_mae": float(test_mae[fold_idx]) if len(test_mae) > fold_idx else 0.0,
                "test_rmse": float(test_rmse[fold_idx]) if len(test_rmse) > fold_idx else 0.0,
            }
            if len(train_r2) > fold_idx:
                d_fold["train_r2"] = float(train_r2[fold_idx])
            linhas_folds.append(d_fold)

        df_folds = pd.DataFrame(linhas_folds)

        return ResultadoValidacaoCruzada(
            scores_cv=scores,
            media_test_r2=float(np.mean(test_r2)) if len(test_r2) > 0 else 0.0,
            std_test_r2=float(np.std(test_r2)) if len(test_r2) > 0 else 0.0,
            media_test_mae=float(np.mean(test_mae)) if len(test_mae) > 0 else 0.0,
            std_test_mae=float(np.std(test_mae)) if len(test_mae) > 0 else 0.0,
            media_test_rmse=float(np.mean(test_rmse)) if len(test_rmse) > 0 else 0.0,
            std_test_rmse=float(np.std(test_rmse)) if len(test_rmse) > 0 else 0.0,
            media_train_r2=float(np.mean(train_r2)) if len(train_r2) > 0 else 0.0,
            std_train_r2=float(np.std(train_r2)) if len(train_r2) > 0 else 0.0,
            n_splits=kf.get_n_splits(x_completo),
            tabela_folds=df_folds,
            kfold=kf,
        )

