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


class Avaliador:
    """Classe responsável pela avaliação estatística e de saúde financeira de modelos imobiliários."""

    def avaliar_metricas_regressao(
        self,
        y_real: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        n_features: int | None = None,
    ) -> dict[str, float]:
        """Calcula métricas estatísticas clássicas de regressão."""
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
        # P25 e P75 dos erros percentuais
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
            "   RELATÓRIO DE SAÚDE FINANCEIRA E PERFORMANCE DO MODELO",
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

    def formatar_equacao_reta(
        self,
        modelo: Any,
        colunas_features: list[str],
        scaler: Any = None,
    ) -> str:
        """Gera a representação textual formatada da equação da reta do modelo de regressão."""
        coefs = getattr(modelo, "coef_", None)
        intercept = getattr(modelo, "intercept_", None)

        if coefs is None or intercept is None:
            return "O modelo especificado não possui coeficientes lineares (coef_ / intercept_)."

        coefs_arr = np.asarray(coefs, dtype=float).ravel()
        intercept_val = float(intercept)

        # Se houver scaler com escala e centro (ex: RobustScaler, StandardScaler)
        tem_scaler = False
        center = getattr(scaler, "center_", getattr(scaler, "mean_", None))
        scale = getattr(scaler, "scale_", None)

        if center is not None and scale is not None and len(scale) == len(coefs_arr):
            tem_scaler = True
            center_arr = np.asarray(center, dtype=float).ravel()
            scale_arr = np.asarray(scale, dtype=float).ravel()
            coefs_orig = coefs_arr / scale_arr
            intercept_orig = intercept_val - float(np.sum(coefs_arr * center_arr / scale_arr))
        else:
            coefs_orig = coefs_arr
            intercept_orig = intercept_val

        linhas = [
            "=" * 65,
            "   EQUAÇÃO DA RETA (MODELO DE REGRESSÃO LINEAR)",
            "=" * 65,
            "Equação em Escala Original (Valores Brutos em R$ e unidades):",
            f"   Preço Estimado = {intercept_orig:+,.2f} (Intercepto β₀)",
        ]

        for col, c in zip(colunas_features, coefs_orig):
            sinal = "+" if c >= 0 else "-"
            linhas.append(f"      {sinal} {abs(c):,.2f} × {col}")

        linhas.append("")
        linhas.append("   * Zona Centro é a categoria base (todas as dummies de Zona = 0)")

        if tem_scaler:
            linhas.append("")
            linhas.append("Equação no Espaço Escalonado (com transformador ativo):")
            linhas.append(f"   y_norm = {intercept_val:+,.2f}")
            for col, c in zip(colunas_features, coefs_arr):
                sinal = "+" if c >= 0 else "-"
                linhas.append(f"      {sinal} {abs(c):,.2f} × {col}_scaled")

        linhas.append("=" * 65)
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

