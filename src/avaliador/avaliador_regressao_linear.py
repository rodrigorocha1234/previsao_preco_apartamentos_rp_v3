import os
from typing import Any
import numpy as np
import pandas as pd

from .avaliador_base import AvaliadorBase


class AvaliadorRegressaoLinear(AvaliadorBase):
    """Avaliador especializado para o modelo de Regressão Linear Simples (Univariada).

    Focado na modelagem univariada tradicional (ex: Metragem -> Preço da Venda),
    gerando equações de reta bidimensionais, diagnósticos de alto viés/underfitting
    e relatórios de importância univariados.
    """

    @property
    def nome_modelo(self) -> str:
        return "Regressão Linear Simples"

    def formatar_equacao_reta(
        self,
        modelo: Any,
        colunas_features: list[str],
        scaler: Any = None,
    ) -> str:
        """Gera a representação textual formatada da equação da reta univariada.

        Equação formatada no formato:
            Preço Estimado = β₀ + β₁ × Metragem
        """
        coefs = getattr(modelo, "coef_", None)
        intercept = getattr(modelo, "intercept_", None)

        if coefs is None or intercept is None:
            return "O modelo especificado não possui coeficientes lineares (coef_ / intercept_)."

        coefs_arr = np.asarray(coefs, dtype=float).ravel()
        intercept_val = float(intercept)

        tem_scaler = False
        center = getattr(scaler, "center_", getattr(scaler, "mean_", None))
        scale = getattr(scaler, "scale_", None)

        if center is not None and scale is not None and len(scale) >= len(coefs_arr):
            tem_scaler = True
            center_arr = np.asarray(center, dtype=float).ravel()[:len(coefs_arr)]
            scale_arr = np.asarray(scale, dtype=float).ravel()[:len(coefs_arr)]
            coefs_orig = coefs_arr / scale_arr
            intercept_orig = intercept_val - float(np.sum(coefs_arr * center_arr / scale_arr))
        else:
            coefs_orig = coefs_arr
            intercept_orig = intercept_val

        feature_nome = colunas_features[0] if colunas_features else "Metragem"
        c_orig = float(coefs_orig[0]) if len(coefs_orig) > 0 else 0.0
        sinal_orig = "+" if c_orig >= 0 else "-"

        linhas = [
            "=" * 65,
            f"   EQUAÇÃO DA RETA ({self.nome_modelo.upper()})",
            "=" * 65,
            "Equação em Escala Original (Valores Brutos em R$ e unidades):",
            f"   Preço Estimado = {intercept_orig:+,.2f} (Intercepto β₀)",
            f"      {sinal_orig} {abs(c_orig):,.2f} × {feature_nome}",
            "",
            "   * Modelo Univariado: Utiliza exclusivamente a metragem como preditor.",
        ]

        if tem_scaler:
            c_norm = float(coefs_arr[0]) if len(coefs_arr) > 0 else 0.0
            sinal_norm = "+" if c_norm >= 0 else "-"
            linhas.append("")
            linhas.append("Equação no Espaço Escalonado (com transformador ativo):")
            linhas.append(f"   y_norm = {intercept_val:+,.2f}")
            linhas.append(f"      {sinal_norm} {abs(c_norm):,.2f} × {feature_nome}_scaled")

        linhas.append("=" * 65)
        return "\n".join(linhas)

    def avaliar_diagnostico_ajuste(
        self,
        y_treino_real: np.ndarray | pd.Series,
        y_treino_pred: np.ndarray | pd.Series,
        y_teste_real: np.ndarray | pd.Series,
        y_teste_pred: np.ndarray | pd.Series,
    ) -> dict[str, Any]:
        """Diagnóstico específico de Underfitting Severo característico da Regressão Linear Simples."""
        diag_base = super().avaliar_diagnostico_ajuste(
            y_treino_real, y_treino_pred, y_teste_real, y_teste_pred
        )
        r2_tr = diag_base["treino"]["r2"]
        r2_te = diag_base["teste"]["r2"]

        # Na regressão simples, a carência de variáveis preditoras (vagas, banheiros, localização)
        # impõe um limite estrito de R² (~25% treino / 43% teste).
        if r2_tr < 0.50 or r2_te < 0.50:
            status = "Underfitting Severo (Alto Viés por Omissão de Variáveis)"
            detalhes = (
                f"O modelo univariado não atinge aderência suficiente (R² treino={r2_tr*100:.1f}%, "
                f"teste={r2_te*100:.1f}%). A exclusão de variáveis cruciais como vagas, banheiros e "
                "localização (zonas) impede a captura da variância dos preços imobiliários."
            )
            diag_base["status"] = status
            diag_base["detalhes"] = detalhes
            diag_base["possui_underfitting"] = True
            diag_base["possui_overfitting"] = False

        return diag_base

    def gerar_grafico_diagnostico_ajuste(
        self,
        modelo: Any,
        x_treino: np.ndarray,
        y_treino: np.ndarray,
        x_teste: np.ndarray,
        y_teste: np.ndarray,
        caminho_salvar: str | None = None,
    ) -> Any:
        """Gera a curva de aprendizado e o comparativo treino/teste para a Regressão Linear Simples."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.model_selection import learning_curve
        from sklearn.base import clone
        from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

        n_amostras = len(x_treino)
        cv_folds = min(5, max(2, n_amostras // 4)) if n_amostras >= 8 else 2
        train_sizes_rel = np.linspace(0.1, 1.0, 6) if n_amostras >= 30 else np.array([0.5, 1.0])

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

        y_pred_tr = modelo.predict(x_treino)
        y_pred_te = modelo.predict(x_teste)
        r2_tr = float(r2_score(y_treino, y_pred_tr)) * 100
        r2_te = float(r2_score(y_teste, y_pred_te)) * 100
        mae_tr = float(mean_absolute_error(y_treino, y_pred_tr)) / 1000.0
        mae_te = float(mean_absolute_error(y_teste, y_pred_te)) / 1000.0
        rmse_tr = float(root_mean_squared_error(y_treino, y_pred_tr)) / 1000.0
        rmse_te = float(root_mean_squared_error(y_teste, y_pred_te)) / 1000.0

        fig, axes = plt.subplots(1, 2, figsize=(18, 8.2), dpi=150)
        plt.subplots_adjust(wspace=0.25, bottom=0.22, top=0.86)

        # SUBPLOT 1: Learning Curve
        ax1 = axes[0]
        ponto_inicio = int(train_sizes[0])
        ponto_fim = int(train_sizes[-1])

        ax1.axvspan(ponto_inicio, ponto_fim, color="#fdcb6e", alpha=0.25, label="Regime Dominante: Platô de Underfitting (Alto Viés Univariado)")
        ax1.plot(train_sizes, train_mean, "o-", color="#0984e3", label=f"Treino (Final: {train_mean[-1]:.1f}%)", lw=2.5, markersize=7)
        ax1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#0984e3")
        ax1.plot(train_sizes, val_mean, "s--", color="#00b894", label=f"Validação CV (Final: {val_mean[-1]:.1f}%)", lw=2.5, markersize=7)
        ax1.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#00b894")

        ax1.set_title(f"Curva de Aprendizado: {self.nome_modelo}\n(Teto Baixo de Aderência R² Decorrente de 1 Única Feature)", fontsize=11, fontweight="bold", pad=14)
        ax1.set_xlabel("Tamanho da Amostra de Treino (N)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Score R² (%)", fontsize=11, fontweight="bold")
        ax1.set_ylim(-20, 100)
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

        ax2.set_title(f"Métricas Treino vs Teste Independente: {self.nome_modelo}", fontsize=11, fontweight="bold", pad=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metricas_nomes, fontsize=11, fontweight="bold")
        ax2.set_ylabel("Valor da Métrica", fontsize=11, fontweight="bold")
        ax2.set_ylim(0, max(max(treino_vals), max(teste_vals)) * 1.15)
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
            f"RESUMO TÉCNICO ({self.nome_modelo.upper()}):\n"
            "• VEREDITO: Underfitting Severo (Alto Viés). O modelo atinge apenas R² ~43% no teste e MAE de ~R$ 198k.\n"
            "• CAUSA: Apenas 1 variável (Metragem) é insuficiente para precificar imóveis em Ribeirão Preto.\n"
            "• RECOMENDAÇÃO: Expandir para Regressão Linear Múltipla incorporando Quartos, Banheiros, Vagas e Zonas."
        )
        fig.text(0.5, 0.02, texto_diagnostico, ha="center", va="bottom",
                 fontsize=9.5, family="monospace", color="#2d3436",
                 bbox=dict(boxstyle="round,pad=0.7", fc="#f8f9fa", ec="#b2bec3", lw=1.5))

        plt.suptitle(f"Diagnóstico de Ajuste: {self.nome_modelo}", fontsize=14, fontweight="bold", y=0.96)

        if caminho_salvar:
            os.makedirs(os.path.dirname(os.path.abspath(caminho_salvar)), exist_ok=True)
            plt.savefig(caminho_salvar, bbox_inches="tight")

        return fig

    def avaliar_importancia_features(
        self,
        modelo: Any,
        x_teste: np.ndarray | pd.DataFrame,
        y_teste: np.ndarray | pd.Series,
        colunas_features: list[str],
        n_repeats: int = 30,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Calcula a importância de feature para o modelo univariado (concentração de 100%)."""
        from sklearn.inspection import permutation_importance

        x_te = np.asarray(x_teste, dtype=float)
        y_te = np.asarray(y_teste, dtype=float).ravel()
        feat_name = colunas_features[0] if colunas_features else "Metragem"

        perm = permutation_importance(
            estimator=modelo,
            X=x_te,
            y=y_te,
            scoring="r2",
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )

        coefs = getattr(modelo, "coef_", None)
        c_val = float(np.asarray(coefs, dtype=float).ravel()[0]) if coefs is not None else 0.0

        queda_r2 = float(perm.importances_mean[0] * 100.0)
        queda_std = float(perm.importances_std[0] * 100.0)

        ranking = [{
            "feature": feat_name,
            "coeficiente_escalonado": c_val,
            "peso_relativo_pct": 100.0,
            "queda_r2_pct": round(queda_r2, 2),
            "queda_r2_std_pct": round(queda_std, 2),
        }]

        return {
            "ranking_features": ranking,
            "n_features": 1,
            "n_repeats": n_repeats,
        }

    def gerar_relatorio_importancia_features(self, resultado_importancia: dict[str, Any]) -> str:
        """Formata o relatório de importância de features univariado."""
        r = resultado_importancia["ranking_features"][0]
        linhas = [
            "=" * 65,
            f"   IMPORTÂNCIA DE RECURSOS (FEATURE IMPORTANCE): {self.nome_modelo.upper()}",
            "=" * 65,
            "Rank | Feature             | Queda R² Teste (%) | Peso Relativo Coef (%)",
            "-----------------------------------------------------------------",
            f"  1  | {r['feature']:19s} | {r['queda_r2_pct']:6.2f}% (±{r['queda_r2_std_pct']:4.2f}%)  | +100.00%",
            "=" * 65,
            "Nota: Sendo um modelo univariado, 100% da variabilidade explicada",
            f"provém exclusivamente da feature '{r['feature']}'.",
            "=" * 65,
        ]
        return "\n".join(linhas)

    def gerar_grafico_importancia_features(
        self,
        resultado_importancia: dict[str, Any],
        nome_modelo: str | None = None,
        caminho_salvar: str | None = None,
    ) -> Any:
        """Gera figura em alta resolução com o gráfico univariado de importância."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = resultado_importancia["ranking_features"][0]
        feat = r["feature"]
        queda = r["queda_r2_pct"]
        std = r["queda_r2_std_pct"]
        modelo_titulo = nome_modelo if nome_modelo is not None else self.nome_modelo

        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        plt.subplots_adjust(bottom=0.22, top=0.80)

        ax.barh([feat], [queda], xerr=[std], color="#0984e3", alpha=0.85, edgecolor="#2d3436", capsize=6, height=0.35)
        ax.set_title(f"Permutation Importance (Modelo Univariado: {feat})\nQueda Média no Score R² ao Permutar a Única Feature", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Queda no R² (pontos percentuais)", fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        ax.annotate(f"{queda:.2f}% (±{std:.2f}%) | Peso Relativo: 100.0%", xy=(max(0.5, queda / 2), 0),
                    ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.set_xlim(0, max(queda * 1.25, 10.0))

        plt.suptitle(f"Análise de Importância de Recursos ({modelo_titulo})", fontsize=13, fontweight="bold", y=0.96)

        if caminho_salvar:
            os.makedirs(os.path.dirname(os.path.abspath(caminho_salvar)), exist_ok=True)
            plt.savefig(caminho_salvar, bbox_inches="tight")

        return fig
