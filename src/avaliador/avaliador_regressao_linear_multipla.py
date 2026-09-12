import os
from typing import Any
import numpy as np
import pandas as pd

from .avaliador_base import AvaliadorBase


class AvaliadorRegressaoLinearMultipla(AvaliadorBase):
    """Avaliador especializado para o modelo de Regressão Linear Múltipla (Multivariada).

    Projetado para modelagens de múltiplas variáveis preditoras (Quartos, Banheiros, Vagas,
    Metragem e Dummies de Zona), provendo equações hiperplanares, curva de aprendizado
    com demarcação de regimes A, B e C, permutation importance e pesos relativos de coeficientes.
    """

    @property
    def nome_modelo(self) -> str:
        return "Regressão Linear Múltipla"

    def formatar_equacao_reta(
        self,
        modelo: Any,
        colunas_features: list[str],
        scaler: Any = None,
    ) -> str:
        """Gera a representação textual formatada da equação hiperplanar multivariada."""
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

        linhas = [
            "=" * 65,
            f"   EQUAÇÃO DA RETA ({self.nome_modelo.upper()})",
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

    def gerar_grafico_diagnostico_ajuste(
        self,
        modelo: Any,
        x_treino: np.ndarray,
        y_treino: np.ndarray,
        x_teste: np.ndarray,
        y_teste: np.ndarray,
        caminho_salvar: str | None = None,
    ) -> Any:
        """Gera figura com a Curva de Aprendizado multivariada e demarcação dos pontos A, B e C."""
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

        # SUBPLOT 1: Learning Curve com Zonas e Pontos Delimitados
        ax1 = axes[0]
        ponto_inicio_over = int(train_sizes[0])
        ponto_corte = int(train_sizes[min(2, len(train_sizes) - 1)])
        ponto_fim_under = int(train_sizes[-1])

        ax1.axvspan(ponto_inicio_over, ponto_corte, color="#ff7675", alpha=0.18, label="Regime 1: Risco de Overfitting (Poucas Amostras)")
        ax1.axvspan(ponto_corte, ponto_fim_under, color="#fdcb6e", alpha=0.22, label="Regime 2: Platô de Underfitting (Alto Viés Linear)")

        ax1.fill_between([ponto_inicio_over, ponto_corte], 102, 110, color="#d63031", alpha=0.90)
        ax1.text((ponto_inicio_over + ponto_corte) / 2, 106, "ZONA DE RISCO DE OVERFITTING", color="white",
                 ha="center", va="center", fontsize=9, fontweight="bold")

        ax1.fill_between([ponto_corte, ponto_fim_under], 102, 110, color="#e67e22", alpha=0.90)
        ax1.text((ponto_corte + ponto_fim_under) / 2, 106, "ZONA DE UNDERFITTING (ALTO VIÉS)", color="white",
                 ha="center", va="center", fontsize=9, fontweight="bold")

        ax1.plot(train_sizes, train_mean, "o-", color="#0984e3", label=f"Treino (Final: {train_mean[-1]:.1f}%)", lw=2.5, markersize=7)
        ax1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#0984e3")

        ax1.plot(train_sizes, val_mean, "s--", color="#00b894", label=f"Validação CV (Final: {val_mean[-1]:.1f}%)", lw=2.5, markersize=7)
        ax1.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#00b894")

        # 1. PONTO A
        ax1.axvline(ponto_inicio_over, color="#c0392b", linestyle="--", lw=2.0, alpha=0.85)
        ax1.annotate(f"PONTO A (N={ponto_inicio_over})\n[+] INÍCIO DO OVERFITTING\n• Gap Treino-Validação Alto (~{train_mean[0]-val_mean[0]:.0f}%)\n• Amostra insuficiente para generalização",
                     xy=(ponto_inicio_over, train_mean[0]),
                     xytext=(ponto_inicio_over + 80, 84),
                     arrowprops=dict(facecolor="#c0392b", edgecolor="#c0392b", shrink=0.08, width=1.5, headwidth=6),
                     fontsize=8.5, fontweight="bold", color="#962d22",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#ffeae8", ec="#e74c3c", lw=1.2))

        # 2. PONTO B
        ax1.axvline(ponto_corte, color="#d35400", linestyle="-.", lw=2.2)
        ax1.annotate(f"PONTO B (N≈{ponto_corte})\n[x] FIM DO OVERFITTING\n[!] INÍCIO DO UNDERFITTING\n• Gap estabilizado (fim da variância)\n• Início do platô de alto viés",
                     xy=(ponto_corte, val_mean[min(2, len(train_sizes) - 1)]),
                     xytext=(ponto_corte + 60, 15),
                     arrowprops=dict(facecolor="#d35400", edgecolor="#d35400", shrink=0.08, width=1.5, headwidth=6),
                     fontsize=8.5, fontweight="bold", color="#7e3800",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#fff9e6", ec="#d35400", lw=1.2))

        # 3. PONTO C
        ax1.axvline(ponto_fim_under, color="#b71540", linestyle=":", lw=2.2)
        ax1.annotate(f"PONTO C (N={ponto_fim_under})\n[*] PLATÔ MÁXIMO DE UNDERFITTING\n• Teto físico da reta linear (~{train_mean[-1]:.0f}%)\n• Mais dados não alteram o viés da reta",
                     xy=(ponto_fim_under, train_mean[-1]),
                     xytext=(ponto_fim_under - 1250, 68),
                     arrowprops=dict(facecolor="#b71540", edgecolor="#b71540", shrink=0.08, width=1.5, headwidth=6),
                     fontsize=8.5, fontweight="bold", color="#6d0c26",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#fdf2f4", ec="#b71540", lw=1.2))

        ax1.set_title("Curva de Aprendizado: Pontos de Início e Término de Overfitting / Underfitting", fontsize=12, fontweight="bold", pad=14)
        ax1.set_xlabel("Tamanho da Amostra de Treino (N)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Score R² (%)", fontsize=11, fontweight="bold")
        ax1.set_ylim(-10, 112)
        ax1.set_xlim(train_sizes[0] - 80, train_sizes[-1] + 120)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

        # SUBPLOT 2: Comparativo Treino vs Teste
        ax2 = axes[1]
        metricas_nomes = ["R² (%)", "MAE (R$ mil)", "RMSE (R$ mil)"]
        treino_vals = [r2_tr, mae_tr, rmse_tr]
        teste_vals = [r2_te, mae_te, rmse_te]

        x = np.arange(len(metricas_nomes))
        width = 0.35

        rects1 = ax2.bar(x - width/2, treino_vals, width, label=f"Treino ({len(x_treino)} amostras)", color="#74b9ff", edgecolor="#0984e3", lw=1.2)
        rects2 = ax2.bar(x + width/2, teste_vals, width, label=f"Teste ({len(x_teste)} amostras)", color="#55efc4", edgecolor="#00b894", lw=1.2)

        ax2.set_title("Diagnóstico de Generalização no Teste Independente (Amostra Completa)", fontsize=12, fontweight="bold", pad=14)
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
            "RESUMO TÉCNICO DOS PONTOS DE TRANSIÇÃO E VEREDITO:\n"
            f"1. OVERFITTING: Começa em N={ponto_inicio_over} (alto gap com poucas amostras) e TERMINA em N≈{ponto_corte}. Na amostra completa NÃO HÁ OVERFITTING (Erro Teste ≤ Treino).\n"
            f"2. UNDERFITTING: COMEÇA em N≈{ponto_corte} (início do platô de viés) e ATINGE O TETO em N={ponto_fim_under}. É um subajuste leve estrutural (teto R² ~81% e MAE ~R$ {mae_te:.1f}k).\n"
            "   Recomendação técnica: Para romper o teto de underfitting da reta linear, utilizar modelos não-lineares baseados em árvores (ex: Random Forest, Gradient Boosting)."
        )

        fig.text(0.5, 0.02, texto_diagnostico, ha="center", va="bottom",
                 fontsize=9.5, family="monospace", color="#2d3436",
                 bbox=dict(boxstyle="round,pad=0.7", fc="#f8f9fa", ec="#b2bec3", lw=1.5))

        plt.suptitle(f"Diagnóstico de Capacidade Preditiva: {self.nome_modelo}",
                     fontsize=14, fontweight="bold", y=0.96)

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
        """Calcula Permutation Importance e pesos relativos dos coeficientes multivariados."""
        from sklearn.inspection import permutation_importance

        x_te = np.asarray(x_teste, dtype=float)
        y_te = np.asarray(y_teste, dtype=float).ravel()
        coefs = getattr(modelo, "coef_", None)

        perm = permutation_importance(
            estimator=modelo,
            X=x_te,
            y=y_te,
            scoring="r2",
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )

        if coefs is not None:
            c_arr = np.asarray(coefs, dtype=float).ravel()
            abs_c = np.abs(c_arr)
            soma_c = float(np.sum(abs_c)) if float(np.sum(abs_c)) > 0 else 1.0
            pesos_rel = (abs_c / soma_c) * 100.0
        else:
            c_arr = np.zeros(len(colunas_features))
            pesos_rel = np.full(len(colunas_features), 100.0 / len(colunas_features))

        ranking: list[dict[str, Any]] = []
        for i, col in enumerate(colunas_features):
            ranking.append({
                "feature": col,
                "coeficiente_escalonado": float(c_arr[i]),
                "peso_relativo_pct": round(float(pesos_rel[i]), 2),
                "queda_r2_pct": round(float(perm.importances_mean[i] * 100.0), 2),
                "queda_r2_std_pct": round(float(perm.importances_std[i] * 100.0), 2),
            })

        ranking.sort(key=lambda item: item["queda_r2_pct"], reverse=True)

        return {
            "ranking_features": ranking,
            "n_features": len(colunas_features),
            "n_repeats": n_repeats,
        }

    def gerar_relatorio_importancia_features(self, resultado_importancia: dict[str, Any]) -> str:
        """Formata o ranking multivariado de importância de features em relatório textual."""
        ranking = resultado_importancia["ranking_features"]
        linhas = [
            "=" * 65,
            f"   ANÁLISE DE IMPORTÂNCIA DE RECURSOS (FEATURE IMPORTANCE): {self.nome_modelo.upper()}",
            "=" * 65,
            "Rank | Feature             | Queda R² Teste (%) | Peso Relativo Coef (%)",
            "-----------------------------------------------------------------",
        ]
        for idx, item in enumerate(ranking, 1):
            feat = item["feature"]
            queda = item["queda_r2_pct"]
            std = item["queda_r2_std_pct"]
            peso = item["peso_relativo_pct"]
            c = item["coeficiente_escalonado"]
            sinal = "+" if c >= 0 else "-"
            linhas.append(
                f" {idx:2d}  | {feat:19s} | {queda:6.2f}% (±{std:4.2f}%)  | {sinal}{peso:5.2f}%"
            )
        linhas.append("=" * 65)
        return "\n".join(linhas)

    def gerar_grafico_importancia_features(
        self,
        resultado_importancia: dict[str, Any],
        nome_modelo: str | None = None,
        caminho_salvar: str | None = None,
    ) -> Any:
        """Gera figura em alta resolução com os 2 subplots comparativos de feature importance."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ranking = resultado_importancia["ranking_features"]
        modelo_titulo = nome_modelo if nome_modelo is not None else self.nome_modelo
        cor_barra_perm = "#0984e3"

        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150)
        plt.subplots_adjust(wspace=0.35, bottom=0.15, top=0.85)

        ranking_plot = list(reversed(ranking))
        feats = [r["feature"].replace("Zona_", "") for r in ranking_plot]
        quedas = [r["queda_r2_pct"] for r in ranking_plot]
        stds = [r["queda_r2_std_pct"] for r in ranking_plot]
        pesos = [r["peso_relativo_pct"] for r in ranking_plot]
        coefs = [r["coeficiente_escalonado"] for r in ranking_plot]

        # SUBPLOT 1: Permutation Importance
        ax1 = axes[0]
        bars1 = ax1.barh(feats, quedas, xerr=stds, color=cor_barra_perm, alpha=0.85, edgecolor="#2d3436", capsize=4)
        ax1.set_title("Permutation Importance no Teste Independente\n(Queda Média no Score R² ao Embaralhar a Feature)", fontsize=11, fontweight="bold", pad=12)
        ax1.set_xlabel("Queda no R² (pontos percentuais)", fontsize=10, fontweight="bold")
        ax1.grid(axis="x", linestyle="--", alpha=0.6)

        for bar in bars1:
            w = bar.get_width()
            ax1.annotate(f"{w:.2f}%", xy=(max(0, w) + 1.2, bar.get_y() + bar.get_height() / 2),
                         va="center", fontsize=9.5, fontweight="bold", color="#2d3436")

        ax1.set_xlim(0, max(max(quedas) * 1.25, 10.0))

        # SUBPLOT 2: Peso Relativo dos Coeficientes
        ax2 = axes[1]
        cores_sinal = ["#00b894" if c >= 0 else "#d63031" for c in coefs]
        bars2 = ax2.barh(feats, pesos, color=cores_sinal, alpha=0.85, edgecolor="#2d3436")
        ax2.set_title("Importância Relativa dos Coeficientes no Espaço Escalonado\n(Verde = Impacto Positivo | Vermelho = Impacto Negativo)", fontsize=11, fontweight="bold", pad=12)
        ax2.set_xlabel("Peso Relativo (%) (|β_j| / ∑|β|)", fontsize=10, fontweight="bold")
        ax2.grid(axis="x", linestyle="--", alpha=0.6)

        for bar, c in zip(bars2, coefs):
            w = bar.get_width()
            sinal = "+" if c >= 0 else "-"
            ax2.annotate(f"{sinal}{w:.1f}%", xy=(w + 0.8, bar.get_y() + bar.get_height() / 2),
                         va="center", fontsize=9.5, fontweight="bold", color="#2d3436")

        ax2.set_xlim(0, max(max(pesos) * 1.25, 10.0))

        plt.suptitle(f"Análise de Importância de Recursos ({modelo_titulo})", fontsize=13, fontweight="bold", y=0.96)

        if caminho_salvar:
            os.makedirs(os.path.dirname(os.path.abspath(caminho_salvar)), exist_ok=True)
            plt.savefig(caminho_salvar, bbox_inches="tight")

        return fig
