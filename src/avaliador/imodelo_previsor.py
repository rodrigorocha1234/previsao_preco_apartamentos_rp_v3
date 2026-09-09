from typing import Any, Protocol


class IModeloPrevisor(Protocol):
    """Protocolo para qualquer modelo com capacidade de predição."""
    def predict(self, X: Any) -> Any:
        ...


# Alias para compatibilidade
ModeloPrevisor = IModeloPrevisor
