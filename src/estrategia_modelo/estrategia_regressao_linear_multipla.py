from typing import Any
import numpy as np
from sklearn.linear_model import LinearRegression

from .estrategia_modelo import EstrategiaModelo
from .iregressor import IRegressor


class EstrategiaRegressaoLinearMultipla(EstrategiaModelo):
    """Estratégia concreta para o modelo de Regressão Linear Múltipla.

    A Regressão Linear Múltipla modela a relação entre uma variável dependente
    contínua (y) e duas ou mais variáveis explicativas/independentes (X₁, X₂, ..., Xₚ):

        y = β₀ + β₁·X₁ + β₂·X₂ + ... + βₚ·Xₚ + ε

    Onde:
        - β₀: Coeficiente linear (intercepto).
        - β₁, β₂, ..., βₚ: Coeficientes angulares (inclinações parciais / pesos marginais).
        - ε: Termo de erro estocástico / resíduo.

    Parâmetros:
        fit_intercept (bool): Se calcula o intercepto β₀ (padrão: True).
        copy_X (bool): Se copia os dados de entrada X (padrão: True).
        n_jobs (int | None): Quantidade de threads para cálculo paralelo (padrão: None).
        positive (bool): Quando True, restringe os coeficientes a serem não-negativos (padrão: False).
        params (dict[str, Any] | None): Grade de hiperparâmetros para otimização via GridSearchCV.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: int | None = None,
        positive: bool = False,
        params: dict[str, Any] | None = None,
    ) -> None:
        self._fit_intercept = fit_intercept
        self._copy_X = copy_X
        self._n_jobs = n_jobs
        self._positive = positive
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
        """Instância do regressor linear subjacente (scikit-learn LinearRegression)."""
        return self._modelo

    @property
    def params(self) -> dict[str, Any]:
        """Grade de hiperparâmetros configurada para otimização (GridSearchCV)."""
        return self._params

    def treinar_modelo_simples(self, x_treino: np.ndarray, y_treino: np.ndarray) -> IRegressor:
        """Ajusta o regressor linear múltiplo aos dados de treino.

        Verifica se a matriz possui múltiplas variáveis preditoras (p >= 2) e
        executa o ajuste OLS (Ordinary Least Squares) minimizando a soma dos resíduos quadráticos.
        """
        x_matriz = np.asarray(x_treino)
        if x_matriz.ndim == 2 and x_matriz.shape[1] < 2:
            import warnings
            warnings.warn(
                f"EstrategiaRegressaoLinearMultipla recebeu apenas {x_matriz.shape[1]} feature. "
                f"Para uma única feature preditora, o modelo opera como regressão simples.",
                UserWarning,
                stacklevel=2,
            )
        return self.modelo.fit(x_treino, y_treino)
