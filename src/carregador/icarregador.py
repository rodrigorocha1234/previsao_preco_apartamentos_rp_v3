from typing import Protocol, TypeVar

T = TypeVar("T", covariant=True)


class ICarregador(Protocol[T]):

    def carregar(self) -> T:
        ...