from typing import Any

from .avaliador_base import AvaliadorBase
from .avaliador_regressao_linear import AvaliadorRegressaoLinear
from .avaliador_regressao_linear_multipla import AvaliadorRegressaoLinearMultipla


class AvaliadorFactory:
    """Fábrica (Factory Method) para criação de instâncias especializadas de avaliadores de modelos."""

    @staticmethod
    def criar_avaliador(
        estrategia_ou_modelo: Any = None,
        n_features: int | None = None,
    ) -> AvaliadorBase:
        """Cria e retorna a instância especializada de Avaliador de acordo com o modelo ou estratégia.

        Args:
            estrategia_ou_modelo: Instância da estratégia de modelo (ex: EstrategiaRegressaoLinear,
                EstrategiaRegressaoLinearMultipla) ou modelo scikit-learn.
            n_features: Quantidade de features utilizadas no treinamento.

        Returns:
            Instância de AvaliadorRegressaoLinear (univariada) ou AvaliadorRegressaoLinearMultipla (multivariada).
        """
        # Se já for um avaliador instanciado, retorna diretamente
        if isinstance(estrategia_ou_modelo, AvaliadorBase):
            return estrategia_ou_modelo

        nome_classe = ""
        if estrategia_ou_modelo is not None:
            nome_classe = getattr(estrategia_ou_modelo, "__class__", type(estrategia_ou_modelo)).__name__

        # Identificação por estratégia explícita
        if "Multipla" in nome_classe:
            return AvaliadorRegressaoLinearMultipla()

        if nome_classe == "EstrategiaRegressaoLinear":
            return AvaliadorRegressaoLinear()

        # Identificação por quantidade de variáveis explicativas
        if n_features is not None:
            if n_features > 1:
                return AvaliadorRegressaoLinearMultipla()
            return AvaliadorRegressaoLinear()

        # Fallback padrão: Avaliador de Regressão Múltipla (caso padrão da aplicação)
        return AvaliadorRegressaoLinearMultipla()
