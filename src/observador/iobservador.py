from typing import Any, Protocol


class IObservadorPipeline(Protocol):
    """Protocolo que define a interface para observadores de eventos do pipeline de ML."""

    def atualizar(self, evento: str, dados: dict[str, Any]) -> None:
        """Recebe notificações de eventos ocorridos no pipeline de treinamento/avaliação.

        Args:
            evento: Nome identificador do evento (ex: 'inicio_treinamento', 'fim_treinamento', 'fim_avaliacao', 'erro').
            dados: Dicionário contendo os dados e metadados associados ao evento.
        """
        ...


class ISujeitoPipeline(Protocol):
    """Protocolo que define a interface de um sujeito observável no pipeline de ML."""

    def adicionar_observador(self, observador: IObservadorPipeline) -> None:
        """Inscreve um novo observador para receber notificações."""
        ...

    def remover_observador(self, observador: IObservadorPipeline) -> None:
        """Remove um observador previamente inscrito."""
        ...

    def notificar(self, evento: str, dados: dict[str, Any]) -> None:
        """Notifica todos os observadores inscritos sobre um evento ocorrido."""
        ...
