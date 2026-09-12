import os
from typing import Any

import pandas as pd

from avaliador.avaliador import Avaliador
from carregador.carregador_csv import CarregadorXLSX
from carregador.icarregador import ICarregador
from estrategia_modelo.estrategia_modelo import EstrategiaModelo
from estrategia_modelo.estrategia_regressao_linear import EstrategiaRegressaoLinear
from estrategia_modelo.estrategia_regressao_linear_multipla import EstrategiaRegressaoLinearMultipla
from observador.iobservador import IObservadorPipeline, ISujeitoPipeline
from observador.observador_mlflow import ObservadorMLflow
from processador.ipreprocessador import IPreprocessador
from processador.preprocessador import Preprocessador


class PipelineML(ISujeitoPipeline):
    """Pipeline de Machine Learning que atua como Sujeito (Subject) no padrão de projeto Observer."""

    def __init__(
        self,
        carregador_dados: ICarregador[pd.DataFrame],
        preprocessador: IPreprocessador | None = None,
        estrategia_modelo: EstrategiaModelo | None = None,
        avaliador: Avaliador | None = None,
        observadores: list[IObservadorPipeline] | None = None,
        flag_processamento: bool = True,
    ) -> None:
        self.__carregador = carregador_dados
        self.__flag_processamento = flag_processamento
        self.__preprocessador = preprocessador
        self.__estrategia_modelo = (
            estrategia_modelo if estrategia_modelo is not None else EstrategiaRegressaoLinear()
        )
        self.__avaliador = avaliador if avaliador is not None else Avaliador()
        self.__observadores: list[IObservadorPipeline] = (
            list(observadores) if observadores is not None else []
        )

    def adicionar_observador(self, observador: IObservadorPipeline) -> None:
        """Inscreve um observador para receber eventos do pipeline."""
        if observador not in self.__observadores:
            self.__observadores.append(observador)

    def remover_observador(self, observador: IObservadorPipeline) -> None:
        """Remove um observador da lista de inscritos."""
        if observador in self.__observadores:
            self.__observadores.remove(observador)

    def notificar(self, evento: str, dados: dict[str, Any]) -> None:
        """Notifica todos os observadores inscritos sobre um evento."""
        for observador in self.__observadores:
            observador.atualizar(evento, dados)

    def __carregar_base(self) -> pd.DataFrame:
        base = self.__carregador.carregar()
        return base

    def rodar_treinamento_simples(self):
        base_original = self.__carregar_base()
        if self.__flag_processamento and self.__preprocessador is not None:
            try:
                self.__preprocessador.base = base_original
                dados = self.__preprocessador.realizar_preprocessamento()

                print("Colunas de features:", self.__preprocessador.colunas_features)
                print("Formato dos dados processados:")
                print(f"X_treino: {dados.x_treino.shape}, y_treino: {dados.y_treino.shape}")
                print(f"X_teste: {dados.x_teste.shape}, y_teste: {dados.y_teste.shape}\n")

                # Notificar início do treinamento
                self.notificar("inicio_treinamento", {
                    "parametros_processador": {
                        "tipo_scaler": getattr(self.__preprocessador, "tipo_scaler", "none"),
                        "escalar": getattr(self.__preprocessador, "escalar", False),
                        "drop_first": getattr(self.__preprocessador, "drop_first", False),
                        "tamanho_teste": getattr(self.__preprocessador, "tamanho_teste", 0.2),
                        "random_state": getattr(self.__preprocessador, "random_state", None),
                    },
                    "parametros_modelo": getattr(self.__estrategia_modelo, "params", {}),
                    "modelo_nome": self.__estrategia_modelo.__class__.__name__,
                    "num_features": len(self.__preprocessador.colunas_features),
                    "colunas_features": self.__preprocessador.colunas_features,
                })

                # 1. Treinamento da Regressão Linear
                print("Treinando o modelo de Regressão Linear...")
                modelo_treinado = self.__estrategia_modelo.treinar_modelo_simples(
                    dados.x_treino, dados.y_treino
                )

                # Notificar fim do treinamento
                self.notificar("fim_treinamento", {
                    "modelo": modelo_treinado,
                })

                # 2. Exibição da Equação da Reta
                scaler_utilizado = getattr(self.__preprocessador, "scaler", None)
                colunas = getattr(self.__preprocessador, "colunas_features", [])
                equacao = self.__avaliador.formatar_equacao_reta(
                    modelo=modelo_treinado,
                    colunas_features=colunas,
                    scaler=scaler_utilizado,
                )
                print(equacao)
                print()

                # 3. Predição no treino e no teste
                y_predicoes_treino = modelo_treinado.predict(dados.x_treino)
                y_predicoes = modelo_treinado.predict(dados.x_teste)

                # 4. Diagnóstico Técnico de Underfitting vs Overfitting
                diag_ajuste = self.__avaliador.avaliar_diagnostico_ajuste(
                    y_treino_real=dados.y_treino,
                    y_treino_pred=y_predicoes_treino,
                    y_teste_real=dados.y_teste,
                    y_teste_pred=y_predicoes,
                )
                relatorio_diag = self.__avaliador.gerar_relatorio_diagnostico_ajuste(diag_ajuste)
                print(relatorio_diag)
                print()

                # 5. Geração do Gráfico Diagnóstico em memória (salvo direto no MLflow)
                figura_diagnostico = self.__avaliador.gerar_grafico_diagnostico_ajuste(
                    modelo=modelo_treinado,
                    x_treino=dados.x_treino,
                    y_treino=dados.y_treino,
                    x_teste=dados.x_teste,
                    y_teste=dados.y_teste,
                    caminho_salvar=None,
                )
                print("Gráfico de diagnóstico de ajuste gerado em memória (será enviado direto ao MLflow sem salvar localmente).\n")

                # 6. Avaliação da Saúde Financeira e Comercial da Imobiliária
                metricas_reg = self.__avaliador.avaliar_metricas_regressao(
                    y_real=dados.y_teste,
                    y_pred=y_predicoes,
                    n_features=dados.x_teste.shape[1],
                )
                metricas_fin = self.__avaliador.avaliar_saude_financeira(
                    y_real=dados.y_teste,
                    y_pred=y_predicoes,
                    taxa_comissao=0.06,
                )
                relatorio = self.__avaliador.gerar_relatorio_financeiro(
                    y_real=dados.y_teste,
                    y_pred=y_predicoes,
                    taxa_comissao=0.06,
                )
                print(relatorio)

                # Notificar fim da avaliação com métricas, relatórios e gráfico para o MLflow
                self.notificar("fim_avaliacao", {
                    "metricas_regressao": metricas_reg,
                    "metricas_financeiras": metricas_fin,
                    "metricas_treino": diag_ajuste["treino"],
                    "relatorio_financeiro": relatorio,
                    "equacao_da_reta": equacao,
                    "diagnostico_ajuste_texto": relatorio_diag,
                    "diagnostico_status": diag_ajuste["status"],
                    "figura_diagnostico": figura_diagnostico,
                })

                return dados, modelo_treinado, y_predicoes

            except Exception as e:
                self.notificar("erro", {"erro": str(e)})
                raise e

        print(base_original.head())
        return None

    def rodar_grid_search(self):
        """Executa o Grid Search puro (busca em grade de hiperparâmetros sem validação cruzada).

        Avalia todas as combinações de hiperparâmetros definidas na estratégia do modelo,
        ajustando cada candidato nos dados de treino e avaliando no conjunto de teste.
        Determina os melhores hiperparâmetros, calibra o modelo vencedor, exibe sua equação da reta,
        gera o relatório financeiro e persiste os melhores parâmetros e métricas no MLflow via Observer.
        """
        base_original = self.__carregar_base()
        if self.__flag_processamento and self.__preprocessador is not None:
            try:
                self.__preprocessador.base = base_original
                dados = self.__preprocessador.realizar_preprocessamento()

                print("=================================================================")
                print("   INICIANDO GRID SEARCH DA REGRESSÃO LINEAR (BUSCA EM GRADE)")
                print("=================================================================")
                print(f"Features analisadas: {len(self.__preprocessador.colunas_features)}")
                print(f"Divisão dos dados: Treino={dados.x_treino.shape[0]} amostras, Teste={dados.x_teste.shape[0]} amostras")
                print(f"Grade de parâmetros pesquisada: {getattr(self.__estrategia_modelo, 'params', {})}\n")

                # 1. Execução do Grid Search com GridSearchCV
                resultado_grid = self.__estrategia_modelo.realizar_grid_search(
                    dados.x_treino,
                    dados.y_treino,
                )

                # 2. Exibição da Tabela de Candidatos e Scores
                print("--- RESULTADOS DO GRID SEARCH ---")
                print(f"Melhores Hiperparâmetros: {resultado_grid.melhores_parametros}")
                print(f"Melhor Score R²: {resultado_grid.melhor_score * 100:.2f}%\n")

                df_res = resultado_grid.tabela_resultados
                cols_necessarias = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
                print("Ranking das Combinações de Hiperparâmetros:")
                linhas_tabela_txt: list[str] = [
                    "Rank | Hiperparâmetros | Score R² Médio | Desvio Padrão",
                    "-----------------------------------------------------------------",
                ]
                for _, row in df_res[cols_necessarias].sort_values("rank_test_score").iterrows():
                    rank = int(row["rank_test_score"])
                    params_desc = str(row["params"])
                    r2_m = float(row["mean_test_score"]) * 100
                    r2_s = float(row["std_test_score"]) * 100
                    linha = f"   Rank {rank}: {params_desc} -> Score R²: {r2_m:.2f}% (±{r2_s:.2f}%)"
                    print(linha)
                    linhas_tabela_txt.append(f"{rank:4d} | {params_desc:45s} | {r2_m:7.2f}% | ±{r2_s:5.2f}%")
                print()
                tabela_grid_txt = "\n".join(linhas_tabela_txt)

                # 3. Notificar início registrando os MELHORES PARÂMETROS no MLflow
                self.notificar("inicio_treinamento", {
                    "parametros_processador": {
                        "tipo_scaler": getattr(self.__preprocessador, "tipo_scaler", "none"),
                        "escalar": getattr(self.__preprocessador, "escalar", False),
                        "drop_first": getattr(self.__preprocessador, "drop_first", False),
                        "tamanho_teste": getattr(self.__preprocessador, "tamanho_teste", 0.2),
                        "random_state": getattr(self.__preprocessador, "random_state", None),
                    },
                    "parametros_modelo": resultado_grid.melhores_parametros,
                    "modelo_nome": f"{self.__estrategia_modelo.__class__.__name__}",
                    "num_features": len(self.__preprocessador.colunas_features),
                    "colunas_features": self.__preprocessador.colunas_features,
                    "tipo_busca": "grid_search",
                })

                # 4. Notificar fim do treinamento com o modelo vencedor e melhores parâmetros
                self.notificar("fim_treinamento", {
                    "modelo": resultado_grid.melhor_modelo,
                    "melhores_parametros": resultado_grid.melhores_parametros,
                })

                # 5. Exibição da Equação da Reta do Modelo Vencedor
                scaler_utilizado = getattr(self.__preprocessador, "scaler", None)
                colunas = getattr(self.__preprocessador, "colunas_features", [])
                equacao = self.__avaliador.formatar_equacao_reta(
                    modelo=resultado_grid.melhor_modelo,
                    colunas_features=colunas,
                    scaler=scaler_utilizado,
                )
                print(equacao)
                print()

                # 6. Avaliação do Modelo Vencedor no Conjunto de Teste
                y_predicoes = resultado_grid.melhor_modelo.predict(dados.x_teste)

                metricas_reg = self.__avaliador.avaliar_metricas_regressao(
                    y_real=dados.y_teste,
                    y_pred=y_predicoes,
                    n_features=dados.x_teste.shape[1],
                )
                metricas_fin = self.__avaliador.avaliar_saude_financeira(
                    y_real=dados.y_teste,
                    y_pred=y_predicoes,
                    taxa_comissao=0.06,
                )
                relatorio = self.__avaliador.gerar_relatorio_financeiro(
                    y_real=dados.y_teste,
                    y_pred=y_predicoes,
                    taxa_comissao=0.06,
                )
                print(relatorio)

                # 7. Notificar fim da avaliação com métricas e artefatos (incluindo tabela do grid)
                self.notificar("fim_avaliacao", {
                    "melhor_score": resultado_grid.melhor_score,
                    "melhores_parametros": resultado_grid.melhores_parametros,
                    "metricas_regressao": metricas_reg,
                    "metricas_financeiras": metricas_fin,
                    "relatorio_financeiro": relatorio,
                    "equacao_da_reta": equacao,
                    "tabela_grid": tabela_grid_txt,
                })

                return dados, resultado_grid, y_predicoes

            except Exception as e:
                self.notificar("erro", {"erro": str(e)})
                raise e

        print(base_original.head())
        return None

    def rodar_validacao_cruzada(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        exibir_relatorio: bool = True,
        run_name: str | None = None,
    ):
        """Executa a Validação Cruzada k-fold utilizando cross_validate e KFold.

        Particiona os dados de treino em k folds balanceados e embaralhados via KFold,
        executando cross_validate para estimar métricas de R², MAE e RMSE em cada fold.
        Gera relatório detalhado de estabilidade e persiste todas as métricas e artefatos no MLflow.
        """
        base_original = self.__carregar_base()
        if self.__flag_processamento and self.__preprocessador is not None:
            try:
                self.__preprocessador.base = base_original
                dados = self.__preprocessador.realizar_preprocessamento()

                if exibir_relatorio:
                    print("=================================================================")
                    print(f"   INICIANDO VALIDAÇÃO CRUZADA (KFOLD = {n_splits} SPLITS)")
                    print("=================================================================")
                    print(f"Features analisadas: {len(self.__preprocessador.colunas_features)}")
                    print(f"Amostras para validação cruzada (Treino): {dados.x_treino.shape[0]} amostras")
                    print(f"Estratégia: KFold(n_splits={n_splits}, shuffle={shuffle}, random_state={random_state})\n")

                from sklearn.model_selection import KFold
                kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

                nome_run = run_name if run_name is not None else f"regressao_linear_cv_seed_{random_state}"

                # 1. Notificar início da validação cruzada no MLflow
                self.notificar("inicio_treinamento", {
                    "run_name": nome_run,
                    "parametros_processador": {
                        "tipo_scaler": getattr(self.__preprocessador, "tipo_scaler", "none"),
                        "escalar": getattr(self.__preprocessador, "escalar", False),
                        "drop_first": getattr(self.__preprocessador, "drop_first", False),
                        "tamanho_teste": getattr(self.__preprocessador, "tamanho_teste", 0.2),
                        "random_state": getattr(self.__preprocessador, "random_state", None),
                    },
                    "parametros_modelo": getattr(self.__estrategia_modelo, "params", {}),
                    "modelo_nome": f"{self.__estrategia_modelo.__class__.__name__}",
                    "num_features": len(self.__preprocessador.colunas_features),
                    "colunas_features": self.__preprocessador.colunas_features,
                    "tipo_busca": "validacao_cruzada",
                    "n_splits": n_splits,
                    "kfold_shuffle": shuffle,
                    "kfold_random_state": random_state,
                    "seed": random_state,
                })

                # 2. Executar validação cruzada com cross_validate
                resultado_cv = self.__estrategia_modelo.realizar_validacao_cruzada(
                    x_completo=dados.x_treino,
                    y_completa=dados.y_treino,
                    kfold=kfold,
                )

                # 3. Ajustar modelo final no treino completo para persistência
                modelo_treinado = self.__estrategia_modelo.treinar_modelo_simples(
                    dados.x_treino, dados.y_treino
                )
                self.notificar("fim_treinamento", {
                    "modelo": modelo_treinado,
                })

                # 4. Formatar relatório analítico de validação cruzada
                relatorio_cv = self.__avaliador.gerar_relatorio_validacao_cruzada(resultado_cv)
                if exibir_relatorio:
                    print(relatorio_cv)
                    print()

                # 5. Dicionário de métricas dos folds individuais para o MLflow
                cv_folds_metricas: dict[str, float] = {}
                for _, row in resultado_cv.tabela_folds.iterrows():
                    f = int(row["fold"])
                    cv_folds_metricas[f"cv_fold_{f}_r2"] = float(row["test_r2"])
                    cv_folds_metricas[f"cv_fold_{f}_mae"] = float(row["test_mae"])
                    cv_folds_metricas[f"cv_fold_{f}_rmse"] = float(row["test_rmse"])

                # 6. Avaliação complementar no teste independente
                y_pred_teste = modelo_treinado.predict(dados.x_teste)
                metricas_reg = self.__avaliador.avaliar_metricas_regressao(
                    y_real=dados.y_teste,
                    y_pred=y_pred_teste,
                    n_features=dados.x_teste.shape[1],
                )
                metricas_fin = self.__avaliador.avaliar_saude_financeira(
                    y_real=dados.y_teste,
                    y_pred=y_pred_teste,
                    taxa_comissao=0.06,
                )
                relatorio_fin = self.__avaliador.gerar_relatorio_financeiro(
                    y_real=dados.y_teste,
                    y_pred=y_pred_teste,
                    taxa_comissao=0.06,
                )

                # 7. Notificar fim da avaliação enviando métricas agregadas, métricas por fold e relatório para o MLflow
                self.notificar("fim_avaliacao", {
                    "validacao_cruzada": {
                        "r2_medio": resultado_cv.media_test_r2,
                        "r2_std": resultado_cv.std_test_r2,
                        "mae_medio": resultado_cv.media_test_mae,
                        "mae_std": resultado_cv.std_test_mae,
                        "rmse_medio": resultado_cv.media_test_rmse,
                        "rmse_std": resultado_cv.std_test_rmse,
                        "train_r2_medio": resultado_cv.media_train_r2,
                        "n_splits": float(resultado_cv.n_splits),
                    },
                    "cv_folds_metricas": cv_folds_metricas,
                    "cv_estrategia": "KFold",
                    "cv_n_splits": str(resultado_cv.n_splits),
                    "relatorio_validacao_cruzada": relatorio_cv,
                    "metricas_regressao": metricas_reg,
                    "metricas_financeiras": metricas_fin,
                    "relatorio_financeiro": relatorio_fin,
                })

                return dados, resultado_cv, modelo_treinado

            except Exception as e:
                self.notificar("erro", {"erro": str(e)})
                raise e

        print(base_original.head())
        return None

    def rodar_validacao_cruzada_multiplas_sementes(
        self,
        n_splits: int = 10,
        sementes: Any = range(30),
    ) -> pd.DataFrame:
        """Executa a Validação Cruzada k-Fold repetidas vezes variando o random_state do KFold.

        Args:
            n_splits: Quantidade de partições do KFold (padrão: 5).
            sementes: Iterável de sementes (padrão: range(0, 30), totalizando 30 repetições).

        Returns:
            pd.DataFrame contendo a tabela com todas as 30 repetições e métricas calculadas.
        """
        lista_sementes = list(sementes)
        qtd = len(lista_sementes)
        print("=" * 70)
        print(f"   EXECUÇÃO DE {qtd} VALIDAÇÕES CRUZADAS (KFOLD = {n_splits} SPLITS)")
        print(f"   Intervalo de sementes (random_state): {lista_sementes[0]} até {lista_sementes[-1]}")
        print("=" * 70)

        linhas_resumo = []
        for i, seed in enumerate(lista_sementes, 1):
            dados, res_cv, modelo = self.rodar_validacao_cruzada(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
                exibir_relatorio=False,
                run_name=f"regressao_linear_cv_seed_{seed:02d}",
            )
            r2_m = res_cv.media_test_r2 * 100
            mae_m = res_cv.media_test_mae
            rmse_m = res_cv.media_test_rmse
            tr_r2 = res_cv.media_train_r2 * 100

            linhas_resumo.append({
                "seed": seed,
                "r2_medio_pct": r2_m,
                "r2_std_pct": res_cv.std_test_r2 * 100,
                "mae_medio": mae_m,
                "mae_std": res_cv.std_test_mae,
                "rmse_medio": rmse_m,
                "rmse_std": res_cv.std_test_rmse,
                "train_r2_pct": tr_r2,
            })
            print(f"[{i:2d}/{qtd}] Seed {seed:2d} -> R² Médio: {r2_m:6.2f}% (±{res_cv.std_test_r2*100:5.2f}%) | MAE: R$ {mae_m:10,.2f} | RMSE: R$ {rmse_m:11,.2f}")

        df_consolidado = pd.DataFrame(linhas_resumo)

        media_geral_r2 = float(df_consolidado["r2_medio_pct"].mean())
        std_geral_r2 = float(df_consolidado["r2_medio_pct"].std())
        media_geral_mae = float(df_consolidado["mae_medio"].mean())
        std_geral_mae = float(df_consolidado["mae_medio"].std())
        media_geral_rmse = float(df_consolidado["rmse_medio"].mean())
        std_geral_rmse = float(df_consolidado["rmse_medio"].std())

        print()
        print("=" * 70)
        print(f"   CONSOLIDAÇÃO ESTATÍSTICA DAS {qtd} REPETIÇÕES (30 SEEDS / {qtd*n_splits} FOLDS):")
        print("=" * 70)
        print(f"• R² Médio Global   : {media_geral_r2:6.2f}% (±{std_geral_r2:5.2f}%) [Min: {df_consolidado['r2_medio_pct'].min():.2f}%, Max: {df_consolidado['r2_medio_pct'].max():.2f}%]")
        print(f"• MAE Médio Global  : R$ {media_geral_mae:,.2f} (±R$ {std_geral_mae:,.2f}) [Min: R$ {df_consolidado['mae_medio'].min():,.2f}, Max: R$ {df_consolidado['mae_medio'].max():,.2f}]")
        print(f"• RMSE Médio Global : R$ {media_geral_rmse:,.2f} (±R$ {std_geral_rmse:,.2f})")
        print("=" * 70)
        print()

        # Registrar run consolidada no MLflow com o resumo estatístico das 30 repetições
        tabela_txt = df_consolidado.to_string(index=False)
        self.notificar("inicio_treinamento", {
            "run_name": f"regressao_linear_cv_consolidado_{qtd}_seeds",
            "tipo_busca": "validacao_cruzada_consolidada",
            "total_repeticoes": qtd,
            "n_splits": n_splits,
        })
        self.notificar("fim_avaliacao", {
            "validacao_cruzada": {
                "r2_global_medio": media_geral_r2 / 100.0,
                "r2_global_std": std_geral_r2 / 100.0,
                "mae_global_medio": media_geral_mae,
                "mae_global_std": std_geral_mae,
                "rmse_global_medio": media_geral_rmse,
                "rmse_global_std": std_geral_rmse,
                "total_repeticoes": float(qtd),
            },
            "relatorio_validacao_cruzada": (
                f"CONSOLIDAÇÃO DE {qtd} REPETIÇÕES DE VALIDAÇÃO CRUZADA (SEEDS {lista_sementes[0]} A {lista_sementes[-1]}):\n\n"
                f"Total de Folds Avaliados: {qtd * n_splits}\n"
                f"R² Global Médio : {media_geral_r2:.2f}% (±{std_geral_r2:.2f}%)\n"
                f"MAE Global Médio: R$ {media_geral_mae:,.2f} (±R$ {std_geral_mae:,.2f})\n"
                f"RMSE Global     : R$ {media_geral_rmse:,.2f} (±R$ {std_geral_rmse:,.2f})\n\n"
                f"TABELA DE TODAS AS SEMENTES:\n{tabela_txt}"
            ),
        })

        return df_consolidado



if __name__ == '__main__':
    lista = ['Zona', 'Quartos', 'Banheiros', 'Vagas', 'Metragem', 'Valor_da_Venda']
    caminho_arquivo = os.path.join(os.getcwd(), 'docs', 'bairro_final_v3_engineered_bkp.xlsx')
    carregador = CarregadorXLSX(caminho=caminho_arquivo, atributos=lista)

    # 1. Configurando o pré-processador com RobustScaler:
    preprocessador_modelo = Preprocessador(
        tipo_scaler="robust",
        escalar=True,
        drop_first=True,
        tamanho_teste=0.2,
        random_state=42,
    )

    # 2. Configurando o Observador do MLflow para Validação Cruzada:
    observador_mlflow = ObservadorMLflow(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        experiment_name="previsao_preco_apartamentos_rp",
        run_name="regressao_linear_multipla_validacao_cruzada",
        tags={"algoritmo": "RegressaoLinearMultipla", "fase": "validacao_cruzada", "cv": "KFold", "ambiente": "desenvolvimento"},
    )

    # 3. Configurando a Estratégia de Modelo (Regressão Linear Múltipla com múltiplas features):
    estrategia_linear_multipla = EstrategiaRegressaoLinearMultipla()

    # 4. Injetando no Pipeline de ML:
    pml = PipelineML(
        carregador_dados=carregador,
        preprocessador=preprocessador_modelo,
        estrategia_modelo=estrategia_linear_multipla,
        observadores=[observador_mlflow],
        flag_processamento=True,
    )
    # pml.rodar_treinamento_simples()
    pml.rodar_grid_search()

    # print(f"=== PIPELINE ML: VALIDAÇÃO CRUZADA KFOLD 30 REPETIÇÕES (SEEDS 0 A 29) ===\n")
    # pml.rodar_validacao_cruzada_multiplas_sementes(n_splits=10, sementes=range(30))
    #
