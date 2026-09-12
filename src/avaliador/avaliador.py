from typing import Any
import numpy as np
import pandas as pd

from .avaliador_base import AvaliadorBase
from .avaliador_regressao_linear import AvaliadorRegressaoLinear
from .avaliador_regressao_linear_multipla import AvaliadorRegressaoLinearMultipla


class Avaliador(AvaliadorBase):
    """Classe polimórfica e retrocompatível para avaliação estatística, diagnóstica e financeira.

    Atua como fachada e despachante adaptativo: detecta automaticamente se o cenário
    é univariado (Regressão Linear Simples) ou multivariado (Regressão Linear Múltipla)
    e delega para as classes especializadas correspondentes.
    """

    def __init__(self) -> None:
        self._avaliador_simples = AvaliadorRegressaoLinear()
        self._avaliador_multipla = AvaliadorRegressaoLinearMultipla()

    @property
    def nome_modelo(self) -> str:
        return "Regressão Linear"

    def formatar_equacao_reta(
        self,
        modelo: Any,
        colunas_features: list[str],
        scaler: Any = None,
    ) -> str:
        """Delega a formatação da equação conforme a dimensionalidade das features."""
        if len(colunas_features) <= 1:
            return self._avaliador_simples.formatar_equacao_reta(modelo, colunas_features, scaler)
        return self._avaliador_multipla.formatar_equacao_reta(modelo, colunas_features, scaler)

    def gerar_grafico_diagnostico_ajuste(
        self,
        modelo: Any,
        x_treino: np.ndarray,
        y_treino: np.ndarray,
        x_teste: np.ndarray,
        y_teste: np.ndarray,
        caminho_salvar: str | None = None,
    ) -> Any:
        """Gera o gráfico de diagnóstico de ajuste apropriado."""
        x_matriz = np.asarray(x_treino)
        n_features = x_matriz.shape[1] if x_matriz.ndim > 1 else 1
        if n_features <= 1:
            return self._avaliador_simples.gerar_grafico_diagnostico_ajuste(
                modelo, x_treino, y_treino, x_teste, y_teste, caminho_salvar
            )
        return self._avaliador_multipla.gerar_grafico_diagnostico_ajuste(
            modelo, x_treino, y_treino, x_teste, y_teste, caminho_salvar
        )

    def avaliar_importancia_features(
        self,
        modelo: Any,
        x_teste: np.ndarray | pd.DataFrame,
        y_teste: np.ndarray | pd.Series,
        colunas_features: list[str],
        n_repeats: int = 30,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Calcula a importância de features delegando para o avaliador específico."""
        if len(colunas_features) <= 1:
            return self._avaliador_simples.avaliar_importancia_features(
                modelo, x_teste, y_teste, colunas_features, n_repeats, random_state
            )
        return self._avaliador_multipla.avaliar_importancia_features(
            modelo, x_teste, y_teste, colunas_features, n_repeats, random_state
        )

    def gerar_relatorio_importancia_features(self, resultado_importancia: dict[str, Any]) -> str:
        """Gera relatório de importância delegando de acordo com a quantidade de features."""
        n_features = resultado_importancia.get("n_features", len(resultado_importancia.get("ranking_features", [])))
        if n_features <= 1:
            return self._avaliador_simples.gerar_relatorio_importancia_features(resultado_importancia)
        return self._avaliador_multipla.gerar_relatorio_importancia_features(resultado_importancia)

    def gerar_grafico_importancia_features(
        self,
        resultado_importancia: dict[str, Any],
        nome_modelo: str | None = None,
        caminho_salvar: str | None = None,
    ) -> Any:
        """Gera o gráfico de importância delegando para a visualização univariada ou multivariada."""
        n_features = resultado_importancia.get("n_features", len(resultado_importancia.get("ranking_features", [])))
        if n_features <= 1:
            return self._avaliador_simples.gerar_grafico_importancia_features(
                resultado_importancia, nome_modelo or self.nome_modelo, caminho_salvar
            )
        return self._avaliador_multipla.gerar_grafico_importancia_features(
            resultado_importancia, nome_modelo or self.nome_modelo, caminho_salvar
        )
