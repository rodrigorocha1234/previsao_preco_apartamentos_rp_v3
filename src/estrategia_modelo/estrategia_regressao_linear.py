from typing import Any
from sklearn.linear_model import LinearRegression

from .estrategia_modelo import EstrategiaModelo
from .iregressor import IRegressor


class EstrategiaRegressaoLinear(EstrategiaModelo):
    """Estratégia concreta para o modelo de Regressão Linear."""

    def __init__(
        self,
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: int | None = None,
        positive: bool = False,
        params: dict[str, Any] | None = None,
    ) -> None:
        self._modelo: IRegressor = LinearRegression(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
        )
        self._params = params if params is not None else {
            "fit_intercept": [True, False],
            "positive": [True, False],
        }

    @property
    def modelo(self) -> IRegressor:
        return self._modelo

    @property
    def params(self) -> dict[str, Any]:
        return self._params


# Importação para permitir acesso direto a partir deste módulo
from .estrategia_regressao_linear_multipla import EstrategiaRegressaoLinearMultipla  # noqa: E402
