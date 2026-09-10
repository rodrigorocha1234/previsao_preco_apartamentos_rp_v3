from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IModeloPrevisor(Protocol):
    """Protocolo para qualquer modelo com capacidade de predição."""

    def predict(self, X: Any, /, *args: Any, **kwargs: Any) -> Any:
        ...


# Alias para compatibilidade
ModeloPrevisor = IModeloPrevisor
