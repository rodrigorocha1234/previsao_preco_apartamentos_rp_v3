from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IRegressor(Protocol):
    """Protocolo que declara a interface esperada de um regressor (fit e predict).
    
    Utiliza argumentos posicionais (PEP 570) e parâmetros flexíveis para garantir
    compatibilidade estrutural com todos os estimadores e regressores do scikit-learn
    (LinearRegression, Ridge, Lasso, RandomForestRegressor, etc.).
    """

    def fit(self, X: Any, y: Any = None, /, *args: Any, **kwargs: Any) -> Any:
        ...

    def predict(self, X: Any, /, *args: Any, **kwargs: Any) -> Any:
        ...


# Alias para compatibilidade
RegressorProtocol = IRegressor
