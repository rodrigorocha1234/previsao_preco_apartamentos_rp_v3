from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IRegressor(Protocol):
    """Protocolo que declara a interface esperada de um regressor (fit e predict).
    
    Compatível 1:1 com os regressores do scikit-learn (LinearRegression, Ridge, Lasso, etc.).
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        ...

    def predict(self, X: Any) -> Any:
        ...


# Alias para compatibilidade
RegressorProtocol = IRegressor
