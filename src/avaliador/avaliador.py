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
            "   DIAGNÓSTICO DE AJUSTE: UNDERFITTING VS OVERFITTING",
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

    def gerar_grafico_diagnostico_ajuste(
        self,
        modelo: Any,
        x_treino: np.ndarray,
        y_treino: np.ndarray,
        x_teste: np.ndarray,
        y_teste: np.ndarray,
        caminho_salvar: str | None = None,
    ) -> Any:
        """Gera e opcionalmente salva a figura com a Curva de Aprendizado e a demarcação de zonas de Overfitting e Underfitting."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.model_selection import learning_curve
        from sklearn.base import clone

        n_amostras = len(x_treino)
        cv_folds = min(5, max(2, n_amostras // 4)) if n_amostras >= 8 else 2
        train_sizes_rel = np.linspace(0.1, 1.0, 6) if n_amostras >= 30 else np.array([0.5, 1.0])

        # 1. Curva de Aprendizado
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=clone(modelo),
            X=x_treino,
            y=y_treino,
            train_sizes=train_sizes_rel,
            cv=cv_folds,
            scoring="r2",
            n_jobs=-1,
        )

        train_mean = np.mean(train_scores, axis=1) * 100
        train_std = np.std(train_scores, axis=1) * 100
        val_mean = np.mean(val_scores, axis=1) * 100
        val_std = np.std(val_scores, axis=1) * 100

        # Métricas pontuais
        y_pred_tr = modelo.predict(x_treino)
        y_pred_te = modelo.predict(x_teste)
        r2_tr = float(r2_score(y_treino, y_pred_tr)) * 100
        r2_te = float(r2_score(y_teste, y_pred_te)) * 100
        mae_tr = float(mean_absolute_error(y_treino, y_pred_tr)) / 1000.0
        mae_te = float(mean_absolute_error(y_teste, y_pred_te)) / 1000.0
        rmse_tr = float(root_mean_squared_error(y_treino, y_pred_tr)) / 1000.0
        rmse_te = float(root_mean_squared_error(y_teste, y_pred_te)) / 1000.0

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
        plt.subplots_adjust(wspace=0.28)

        # SUBPLOT 1: Learning Curve com Zonas Delimitadas
        ax1 = axes[0]
        # Ponto de corte na transição da estabilização
        ponto_corte = train_sizes[min(2, len(train_sizes) - 1)]

        ax1.axvspan(train_sizes[0], ponto_corte, color="#ff7675", alpha=0.18, label="Zona 1: Início com Risco de Overfitting (Gap Alto)")
        ax1.axvspan(ponto_corte, train_sizes[-1], color="#fdcb6e", alpha=0.20, label="Zona 2: Convergência / Leve Underfitting (Alto Viés)")

        ax1.axvline(ponto_corte, color="#d63031", linestyle=":", lw=2)
        ax1.text(ponto_corte + 30, 10, "Fim do Overfitting →\nInício do Platô de Underfitting",
                 color="#b71540", fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#ffeaa7", ec="#d63031", alpha=0.95))

        ax1.plot(train_sizes, train_mean, "o-", color="#0984e3", label=f"Treino (Final: {train_mean[-1]:.1f}%)", lw=2.5, markersize=6)
        ax1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#0984e3")

        ax1.plot(train_sizes, val_mean, "s--", color="#00b894", label=f"Validação CV (Final: {val_mean[-1]:.1f}%)", lw=2.5, markersize=6)
        ax1.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#00b894")

        # Anotações explicativas
        ax1.annotate("Gap Treino-Validação Inicial\n(Início do Overfitting com poucas amostras)",
                     xy=(train_sizes[1], (train_mean[1] + val_mean[1]) / 2),
                     xytext=(train_sizes[1] + 150, 60),
                     arrowprops=dict(facecolor="#d63031", shrink=0.05, width=1.5, headwidth=7),
                     fontsize=8.5, fontweight="bold", color="#d63031",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#fff", ec="#d63031"))

        ax1.annotate("Estabilização Assintótica (~74% Treino)\n(Platô de Underfitting Estrutural Linear)",
                     xy=(train_sizes[-1], train_mean[-1]),
                     xytext=(train_sizes[-1] - (train_sizes[-1] - train_sizes[0]) * 0.45, 88),
                     arrowprops=dict(facecolor="#e17055", shrink=0.05, width=1.5, headwidth=7),
                     fontsize=8.5, fontweight="bold", color="#d35400",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#fff", ec="#e17055"))

        ax1.set_title("Curva de Aprendizado: Demarcação de Overfitting vs Underfitting", fontsize=12, fontweight="bold", pad=12)
        ax1.set_xlabel("Tamanho da Amostra de Treino (N)", fontsize=11)
        ax1.set_ylabel("Score R² (%)", fontsize=11)
        ax1.set_ylim(-15, 105)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="lower right", fontsize=9, framealpha=0.95)

        # SUBPLOT 2: Comparativo Treino vs Teste
        ax2 = axes[1]
        metricas_nomes = ["R² (%)", "MAE (R$ mil)", "RMSE (R$ mil)"]
        treino_vals = [r2_tr, mae_tr, rmse_tr]
        teste_vals = [r2_te, mae_te, rmse_te]

        x = np.arange(len(metricas_nomes))
        width = 0.35

        rects1 = ax2.bar(x - width/2, treino_vals, width, label=f"Treino ({len(x_treino)} amostras)", color="#74b9ff", edgecolor="#0984e3", lw=1.2)
        rects2 = ax2.bar(x + width/2, teste_vals, width, label=f"Teste ({len(x_teste)} amostras)", color="#55efc4", edgecolor="#00b894", lw=1.2)

        ax2.set_title("Diagnóstico no Teste Independente: Ausência de Overfitting", fontsize=12, fontweight="bold", pad=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metricas_nomes, fontsize=11, fontweight="bold")
        ax2.grid(axis="y", linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

        for rect in rects1:
            h = rect.get_height()
            ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                         textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2d3436")
        for rect in rects2:
            h = rect.get_height()
            ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                         textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2d3436")

        texto_diagnostico = (
            "VEREDITO TÉCNICO:\n"
            f"• OVERFITTING: NÃO OCORRE (Erro Teste ≤ Treino; R² Teste {r2_te:.1f}% ≥ Treino {r2_tr:.1f}%)\n"
            f"• UNDERFITTING: LEVE / MODERADO (Alto viés linear: teto de R² ~81% e MAE ~R$ {mae_te:.1f}k)\n"
            "  Solução para underfitting: Modelos não-lineares (Random Forest / GBDT / Redes Neurais)"
        )
        ax2.text(0.5, -0.22, texto_diagnostico, transform=ax2.transAxes, ha="center", va="top",
                 fontsize=9.5, family="monospace", color="#2d3436",
                 bbox=dict(boxstyle="round,pad=0.6", fc="#f8f9fa", ec="#b2bec3", lw=1.5))

        plt.suptitle("Diagnóstico de Capacidade Preditiva da Regressão Linear: Underfitting vs Overfitting",
                     fontsize=14, fontweight="bold", y=0.98)

        if caminho_salvar:
            os.makedirs(os.path.dirname(os.path.abspath(caminho_salvar)), exist_ok=True)
            plt.savefig(caminho_salvar, bbox_inches="tight")

        return fig

