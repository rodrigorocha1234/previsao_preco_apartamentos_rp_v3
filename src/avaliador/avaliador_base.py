from abc import ABC, abstractmethod
import os
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from .imodelo_previsor import IModeloPrevisor


class AvaliadorBase(ABC):
    """Classe base abstrata para avaliação estatística, diagnóstica e de saúde financeira de modelos imobiliários."""

    @property
    @abstractmethod
    def nome_modelo(self) -> str:
        """Nome identificador do modelo avaliado."""
        ...

    def avaliar_metricas_regressao(
        self,
        y_real: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        n_features: int | None = None,
    ) -> dict[str, float]:
        """Calcula métricas estatísticas clássicas de regressão.

        Args:
            y_real: Valores observados reais.
            y_pred: Valores preditos pelo modelo.
            n_features: Número de features explicativas para cômputo do R² ajustado.

        Returns:
            Dicionário com R², R² ajustado, MAE, MedAE, RMSE, MAPE e Max Error.
        """
        y_r = np.asarray(y_real, dtype=float).ravel()
        y_p = np.asarray(y_pred, dtype=float).ravel()

        r2_val = float(r2_score(y_r, y_p))
        n_samples = len(y_r)

        if n_features is not None and n_samples > n_features + 1:
            r2_ajustado = float(1.0 - ((1.0 - r2_val) * (n_samples - 1) / (n_samples - n_features - 1)))
        else:
            r2_ajustado = r2_val

        return {
            "r2": r2_val,
            "r2_ajustado": r2_ajustado,
            "mae": float(mean_absolute_error(y_r, y_p)),
            "medae": float(median_absolute_error(y_r, y_p)),
            "rmse": float(root_mean_squared_error(y_r, y_p)),
            "mape": float(mean_absolute_percentage_error(y_r, y_p)),
            "max_error": float(max_error(y_r, y_p)),
        }

    def avaliar_saude_financeira(
        self,
        y_real: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        taxa_comissao: float = 0.06,
    ) -> dict[str, Any]:
        """Calcula métricas de negócio orientadas à saúde financeira e comercial da imobiliária.

        Args:
            y_real: Valores reais de venda dos imóveis.
            y_pred: Preços estimados pelo modelo.
            taxa_comissao: Alíquota padrão de comissão da imobiliária (ex: 0.06 para 6%).

        Returns:
            Dicionário com métricas de impacto comercial, riscos e faixas de desconto.
        """
        y_r = np.asarray(y_real, dtype=float).ravel()
        y_p = np.asarray(y_pred, dtype=float).ravel()

        # Resíduo percentual relativo ao valor real: (predito - real) / real
        erros_percentuais = (y_p - y_r) / y_r

        # Faixa de desconto prudente baseada nos resíduos de sub/superestimação
        desconto_min_sugerido = float(np.clip(np.percentile(np.abs(erros_percentuais), 25) * 100, 3.0, 10.0))
        desconto_max_seguro = float(np.clip(np.percentile(np.abs(erros_percentuais), 50) * 100, 5.0, 15.0))

        # Assertividade em faixas de tolerância comercial
        assertividade_5 = float(np.mean(np.abs(erros_percentuais) <= 0.05) * 100)
        assertividade_10 = float(np.mean(np.abs(erros_percentuais) <= 0.10) * 100)
        assertividade_15 = float(np.mean(np.abs(erros_percentuais) <= 0.15) * 100)

        # Riscos operacionais e de liquidez:
        # Superavaliação (> +10%): Risco de encalhe / imóvel parado no estoque / custo de marketing
        risco_superavaliacao = float(np.mean(erros_percentuais > 0.10) * 100)

        # Subavaliação (< -10%): Dinheiro deixado na mesa / perda de comissão / insatisfação do proprietário
        risco_subavaliacao = float(np.mean(erros_percentuais < -0.10) * 100)

        # Impacto monetário na comissão (taxa de corretagem)
        comissao_real = y_r * taxa_comissao
        comissao_prevista = y_p * taxa_comissao
        desvio_medio_comissao = float(np.mean(np.abs(comissao_prevista - comissao_real)))

        return {
            "faixa_desconto_sugerida_min_pct": round(desconto_min_sugerido, 2),
            "faixa_desconto_sugerida_max_pct": round(desconto_max_seguro, 2),
            "mediana_erro_percentual_pct": round(float(np.median(erros_percentuais) * 100), 2),
            "risco_superavaliacao_pct": round(risco_superavaliacao, 2),
            "risco_subavaliacao_pct": round(risco_subavaliacao, 2),
            "assertividade_tolerancia_5pct": round(assertividade_5, 2),
            "assertividade_tolerancia_10pct": round(assertividade_10, 2),
            "assertividade_tolerancia_15pct": round(assertividade_15, 2),
            "desvio_medio_comissao_reais": round(desvio_medio_comissao, 2),
        }

    def gerar_relatorio_financeiro(
        self,
        y_real: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        taxa_comissao: float = 0.06,
    ) -> str:
        """Gera um relatório textual formatado combinando métricas de ML e impacto de negócio."""
        reg = self.avaliar_metricas_regressao(y_real, y_pred)
        fin = self.avaliar_saude_financeira(y_real, y_pred, taxa_comissao)

        linhas = [
            "=" * 65,
            f"   RELATÓRIO DE SAÚDE FINANCEIRA E PERFORMANCE: {self.nome_modelo.upper()}",
            "=" * 65,
            "1. DESEMPENHO ESTATÍSTICO DO MODELO:",
            f"   - R² (Aderência do Modelo)       : {reg['r2'] * 100:.2f}%",
            f"   - MedAE (Erro Típico / Mediano)  : R$ {reg['medae']:,.2f}",
            f"   - MAE (Erro Médio Absoluto)      : R$ {reg['mae']:,.2f}",
            f"   - RMSE (Penalidade Outliers)     : R$ {reg['rmse']:,.2f}",
            f"   - MAPE (Erro Percentual Médio)   : {reg['mape'] * 100:.2f}%",
            f"   - Max Error (Pior Desvio Pontual): R$ {reg['max_error']:,.2f}",
            "",
            "2. FAIXA DE DESCONTO E NEGOCIAÇÃO RECOMENDADA:",
            f"   - Faixa de Desconto Segura       : {fin['faixa_desconto_sugerida_min_pct']:.1f}% a {fin['faixa_desconto_sugerida_max_pct']:.1f}%",
            f"   - Viés do Modelo (Mediana Erro)  : {fin['mediana_erro_percentual_pct']:+.2f}%",
            "",
            "3. MATRIZ DE RISCO OPERACIONAL PARA A IMOBILIÁRIA:",
            f"   - Risco de Superavaliação (>+10%): {fin['risco_superavaliacao_pct']:.2f}% (Risco de Encalhe / Time on Market)",
            f"   - Risco de Subavaliação (< -10%) : {fin['risco_subavaliacao_pct']:.2f}% (Dinheiro na Mesa / Perda de Comissão)",
            "",
            f"4. IMPACTO NA RECEITA DE CORRETAGEM (Comissão de {taxa_comissao * 100:.0f}%):",
            f"   - Desvio Médio de Comissão/Imóvel: R$ {fin['desvio_medio_comissao_reais']:,.2f}",
            "",
            "5. ASSERTIVIDADE COMERCIAL EM FAIXAS DE TOLERÂNCIA:",
            f"   - Dentro de ±5% do Preço Real    : {fin['assertividade_tolerancia_5pct']:.2f}%",
            f"   - Dentro de ±10% do Preço Real   : {fin['assertividade_tolerancia_10pct']:.2f}%",
            f"   - Dentro de ±15% do Preço Real   : {fin['assertividade_tolerancia_15pct']:.2f}%",
            "=" * 65,
        ]
        return "\n".join(linhas)

    def avaliar_acuracia(
        self,
        modelo: IModeloPrevisor,
        x_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
    ) -> float:
        """Mantido para compatibilidade com interfaces anteriores."""
        predicoes = modelo.predict(x_test)
        return float(accuracy_score(y_test, predicoes))

    def avaliar_diagnostico_ajuste(
        self,
        y_treino_real: np.ndarray | pd.Series,
        y_treino_pred: np.ndarray | pd.Series,
        y_teste_real: np.ndarray | pd.Series,
        y_teste_pred: np.ndarray | pd.Series,
    ) -> dict[str, Any]:
        """Compara métricas de treino e teste e gera o diagnóstico técnico de Underfitting vs Overfitting."""
        y_tr = np.asarray(y_treino_real, dtype=float).ravel()
        y_tp = np.asarray(y_treino_pred, dtype=float).ravel()
        y_te = np.asarray(y_teste_real, dtype=float).ravel()
        y_ep = np.asarray(y_teste_pred, dtype=float).ravel()

        r2_tr = float(r2_score(y_tr, y_tp))
        r2_te = float(r2_score(y_te, y_ep))
        mae_tr = float(mean_absolute_error(y_tr, y_tp))
        mae_te = float(mean_absolute_error(y_te, y_ep))
        rmse_tr = float(root_mean_squared_error(y_tr, y_tp))
        rmse_te = float(root_mean_squared_error(y_te, y_ep))

        delta_r2 = r2_tr - r2_te
        razao_mae = mae_te / mae_tr if mae_tr > 0 else 1.0

        if delta_r2 > 0.15 and razao_mae > 1.20:
            status = "Overfitting (Sobreajuste / Alta Variância)"
            detalhes = (
                "O modelo memorizou as particularidades do conjunto de treino e perdeu capacidade "
                f"de generalização no teste (R² caiu de {r2_tr*100:.1f}% para {r2_te*100:.1f}%; "
                f"MAE subiu {((razao_mae-1)*100):.1f}%)."
            )
            possui_overfitting = True
            possui_underfitting = False
        elif r2_tr < 0.50 and r2_te < 0.50:
            status = "Underfitting Severo (Alto Viés)"
            detalhes = (
                "O modelo não conseguiu aprender os padrões fundamentais dos dados em nenhum dos "
                f"conjuntos (R² treino={r2_tr*100:.1f}%, teste={r2_te*100:.1f}%)."
            )
            possui_overfitting = False
            possui_underfitting = True
        else:
            status = "Leve Underfitting (Alto Viés Linear) / Ausência de Overfitting"
            detalhes = (
                "O modelo generaliza perfeitamente para novos dados (Erro de Teste ≤ Treino; "
                f"R² teste={r2_te*100:.2f}% vs treino={r2_tr*100:.2f}%), descartando Overfitting. "
                "Apresenta um leve underfitting estrutural decorrente da hipótese de linearidade da reta, "
                "que limita a modelagem de efeitos não-lineares complexos do mercado imobiliário."
            )
            possui_overfitting = False
            possui_underfitting = True

        return {
            "status": status,
            "detalhes": detalhes,
            "possui_overfitting": possui_overfitting,
            "possui_underfitting": possui_underfitting,
            "treino": {
                "r2": r2_tr,
                "mae": mae_tr,
                "rmse": rmse_tr,
            },
            "teste": {
                "r2": r2_te,
                "mae": mae_te,
                "rmse": rmse_te,
            },
            "delta_r2_pct": round(delta_r2 * 100, 2),
            "razao_mae": round(razao_mae, 3),
        }

    def gerar_relatorio_diagnostico_ajuste(self, diagnostico: dict[str, Any]) -> str:
        """Formata o diagnóstico de underfitting/overfitting em texto legível para terminal e relatórios."""
        tr = diagnostico["treino"]
        te = diagnostico["teste"]
        linhas = [
            "=" * 65,
            f"   DIAGNÓSTICO DE AJUSTE: {self.nome_modelo.upper()}",
            "=" * 65,
            "1. COMPARATIVO DE MÉTRICAS ENTRE TREINO E TESTE:",
            f"   - R² (Aderência)     : Treino {tr['r2']*100:.2f}% | Teste {te['r2']*100:.2f}% (Gap: {diagnostico['delta_r2_pct']:+.2f} p.p.)",
            f"   - MAE (Erro Médio)   : Treino R$ {tr['mae']:,.2f} | Teste R$ {te['mae']:,.2f}",
            f"   - RMSE (Sensibilidade): Treino R$ {tr['rmse']:,.2f} | Teste R$ {te['rmse']:,.2f}",
            "",
            f"2. VEREDITO TÉCNICO: {diagnostico['status']}",
            f"   - Overfitting (Sobreajuste): {'SIM' if diagnostico['possui_overfitting'] else 'NÃO OCORRE'}",
            f"   - Underfitting (Subajuste) : {'SIM (Leve / Estrutural)' if diagnostico['possui_underfitting'] else 'NÃO'}",
            "",
            "3. PARECER ANALÍTICO E RECOMENDAÇÃO:",
            f"   {diagnostico['detalhes']}",
            "=" * 65,
        ]
        return "\n".join(linhas)

    def gerar_relatorio_validacao_cruzada(self, resultado: Any) -> str:
        """Formata o relatório textual completo da Validação Cruzada k-fold para console e MLflow."""
        n_splits = getattr(resultado, "n_splits", 5)
        media_r2 = getattr(resultado, "media_test_r2", 0.0) * 100
        std_r2 = getattr(resultado, "std_test_r2", 0.0) * 100
        media_mae = getattr(resultado, "media_test_mae", 0.0)
        std_mae = getattr(resultado, "std_test_mae", 0.0)
        media_rmse = getattr(resultado, "media_test_rmse", 0.0)
        std_rmse = getattr(resultado, "std_test_rmse", 0.0)
        media_tr_r2 = getattr(resultado, "media_train_r2", 0.0) * 100
        std_tr_r2 = getattr(resultado, "std_train_r2", 0.0) * 100
        df_folds = getattr(resultado, "tabela_folds", pd.DataFrame())

        linhas = [
            "=" * 65,
            f"   RELATÓRIO DE VALIDAÇÃO CRUZADA: {self.nome_modelo.upper()} (KFOLD = {n_splits} SPLITS)",
            "=" * 65,
            "1. DESEMPENHO MÉDIO GERAL (ESTIMATIVA NÃO-ENVIESADA):",
            f"   - Score R² Médio     : {media_r2:.2f}% (±{std_r2:.2f}%)",
            f"   - MAE Médio (R$)     : R$ {media_mae:,.2f} (±R$ {std_mae:,.2f})",
            f"   - RMSE Médio (R$)    : R$ {media_rmse:,.2f} (±R$ {std_rmse:,.2f})",
            f"   - R² Treino Médio    : {media_tr_r2:.2f}% (±{std_tr_r2:.2f}%)",
            "",
            "2. DETALHAMENTO INDIVIDUAL POR FOLD:",
            "   Fold | R² Teste (%) | MAE Teste (R$)    | RMSE Teste (R$)   | R² Treino (%)",
            "   -------------------------------------------------------------------------",
        ]

        if not df_folds.empty:
            for _, row in df_folds.iterrows():
                f_idx = int(row["fold"])
                r2_val = float(row["test_r2"]) * 100
                mae_val = float(row["test_mae"])
                rmse_val = float(row["test_rmse"])
                tr_r2_val = float(row.get("train_r2", 0.0)) * 100
                linhas.append(
                    f"    {f_idx:2d}  | {r2_val:10.2f}% | R$ {mae_val:13,.2f} | R$ {rmse_val:14,.2f} | {tr_r2_val:11.2f}%"
                )

        linhas.extend([
            "=" * 65,
            "3. PARECER ANALÍTICO DE ESTABILIDADE:",
            f"   A dispersão do R² (desvio padrão de {std_r2:.2f} p.p.) demonstra a "
            f"{'alta estabilidade' if std_r2 < 10.0 else 'sensibilidade amostral'} do regressor",
            "   diante de variações nas partições de teste.",
            "=" * 65,
        ])
        return "\n".join(linhas)

    @abstractmethod
    def formatar_equacao_reta(
        self,
        modelo: Any,
        colunas_features: list[str],
        scaler: Any = None,
    ) -> str:
        """Gera a representação textual formatada da equação do modelo."""
        ...

    @abstractmethod
    def gerar_grafico_diagnostico_ajuste(
        self,
        modelo: Any,
        x_treino: np.ndarray,
        y_treino: np.ndarray,
        x_teste: np.ndarray,
        y_teste: np.ndarray,
        caminho_salvar: str | None = None,
    ) -> Any:
        """Gera a figura diagnóstica de aprendizado e generalização."""
        ...

    @abstractmethod
    def avaliar_importancia_features(
        self,
        modelo: Any,
        x_teste: np.ndarray | pd.DataFrame,
        y_teste: np.ndarray | pd.Series,
        colunas_features: list[str],
        n_repeats: int = 30,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Calcula a importância de features para o modelo."""
        ...

    @abstractmethod
    def gerar_relatorio_importancia_features(self, resultado_importancia: dict[str, Any]) -> str:
        """Formata o ranking de importância de features em relatório textual."""
        ...

    @abstractmethod
    def gerar_grafico_importancia_features(
        self,
        resultado_importancia: dict[str, Any],
        nome_modelo: str | None = None,
        caminho_salvar: str | None = None,
    ) -> Any:
        """Gera a figura com os gráficos de importância de features."""
        ...
