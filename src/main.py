import os
from typing import Any

import pandas as pd

from avaliador.avaliador import Avaliador
from carregador.carregador_csv import CarregadorXLSX
from carregador.icarregador import ICarregador
from estrategia_modelo.estrategia_modelo import EstrategiaModelo
from estrategia_modelo.estrategia_regressao_linear import EstrategiaRegressaoLinear
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

                # 5. Geração do Gráfico Diagnóstico com Zonas Delimitadas de Overfitting/Underfitting
                caminho_grafico = os.path.join(os.getcwd(), "docs", "diagnostico_ajuste.png")
                figura_diagnostico = self.__avaliador.gerar_grafico_diagnostico_ajuste(
                    modelo=modelo_treinado,
                    x_treino=dados.x_treino,
                    y_treino=dados.y_treino,
                    x_teste=dados.x_teste,
                    y_teste=dados.y_teste,
                    caminho_salvar=caminho_grafico,
                )
                print(f"Gráfico de diagnóstico de ajuste gerado e salvo em: {caminho_grafico}\n")

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
                    "caminho_figura_diagnostico": caminho_grafico,
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

    # 2. Configurando o Observador do MLflow:
    observador_mlflow = ObservadorMLflow(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        experiment_name="previsao_preco_apartamentos_rp",
        run_name="regressao_linear_treinamento_simples",
        tags={"algoritmo": "RegressaoLinear", "fase": "treinamento_simples", "ambiente": "desenvolvimento"},
    )

    # 3. Injetando no Pipeline de ML:
    pml = PipelineML(
        carregador_dados=carregador,
        preprocessador=preprocessador_modelo,
        observadores=[observador_mlflow],
        flag_processamento=True,
    )

    print(f"=== PIPELINE ML: TREINAMENTO SIMPLES COM DIAGNÓSTICO DE AJUSTE (SCALER: '{preprocessador_modelo.tipo_scaler.upper()}') ===\n")
    pml.rodar_treinamento_simples()
