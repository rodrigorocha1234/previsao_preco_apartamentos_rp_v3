import logging
import os
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn

from .iobservador import IObservadorPipeline

logger = logging.getLogger(__name__)


class ObservadorMLflow(IObservadorPipeline):
    """Observador concreto do padrão Observer responsável por persistir o ciclo de vida do modelo no MLflow."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "previsao_preco_apartamentos_rp",
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        logar_modelo: bool = True,
        logar_artefatos: bool = True,
    ) -> None:
        """Inicializa o observador do MLflow.

        Args:
            tracking_uri: URI do servidor MLflow (ex: 'http://localhost:5000' ou 'postgresql://...').
                          Se None, utiliza a variável de ambiente MLFLOW_TRACKING_URI ou diretório local.
            experiment_name: Nome do experimento para agrupar as execuções.
            run_name: Nome identificador da execução (run). Se None, o MLflow gera automaticamente.
            tags: Dicionário de tags adicionais para categorização.
            logar_modelo: Se True, registra o artefato do modelo treinado no MLflow.
            logar_artefatos: Se True, registra relatórios textuais e equações como artefatos no MLflow.
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.logar_modelo = logar_modelo
        self.logar_artefatos = logar_artefatos
        self._run_ativa = False

    def atualizar(self, evento: str, dados: dict[str, Any]) -> None:
        """Processa eventos emitidos pelo sujeito (PipelineML).

        Args:
            evento: Nome do evento ('inicio_treinamento', 'fim_treinamento', 'fim_avaliacao', 'erro').
            dados: Dicionário de dados associados.
        """
        if evento == "inicio_treinamento":
            self._ao_iniciar_treinamento(dados)
        elif evento == "fim_treinamento":
            self._ao_finalizar_treinamento(dados)
        elif evento == "fim_avaliacao":
            self._ao_finalizar_avaliacao(dados)
        elif evento == "erro":
            self._ao_ocorrer_erro(dados)

    def _ao_iniciar_treinamento(self, dados: dict[str, Any]) -> None:
        """Inicia a run no MLflow e salva parâmetros e tags iniciais."""
        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            mlflow.set_experiment(self.experiment_name)
            nome_run = str(dados.get("run_name", self.run_name))
            mlflow.start_run(run_name=nome_run)
            self._run_ativa = True

            # Tags padrão e customizadas
            tags_finais = {
                "projeto": "previsao_preco_apartamentos_rp",
                "versao": "v3",
                **self.tags,
            }
            if "modelo_nome" in dados:
                tags_finais["modelo_nome"] = str(dados["modelo_nome"])
            if "kfold_random_state" in dados:
                tags_finais["kfold_random_state"] = str(dados["kfold_random_state"])
            if "seed" in dados:
                tags_finais["seed"] = str(dados["seed"])
            mlflow.set_tags(tags_finais)

            # Log de parâmetros de pré-processamento e hiperparâmetros
            params: dict[str, Any] = {}
            if "kfold_random_state" in dados:
                params["kfold_random_state"] = dados["kfold_random_state"]
            if "parametros_processador" in dados and isinstance(dados["parametros_processador"], dict):
                for k, v in dados["parametros_processador"].items():
                    params[f"prep_{k}"] = str(v) if isinstance(v, (list, dict, tuple)) else v

            if "parametros_modelo" in dados and isinstance(dados["parametros_modelo"], dict):
                for k, v in dados["parametros_modelo"].items():
                    if not isinstance(v, (list, dict, tuple)):
                        params[k] = v
                    params[f"model_{k}"] = str(v) if isinstance(v, (list, dict, tuple)) else v

            if "num_features" in dados:
                params["num_features"] = dados["num_features"]

            if params:
                mlflow.log_params(params)

        except Exception as e:
            logger.warning(f"Falha ao inicializar run do MLflow: {e}")

    def _ao_finalizar_treinamento(self, dados: dict[str, Any]) -> None:
        """Registra o modelo treinado e os melhores parâmetros no MLflow."""
        if not self._run_ativa:
            return

        try:
            # Salvar os melhores parâmetros encontrados pelo Grid Search
            if "melhores_parametros" in dados and isinstance(dados["melhores_parametros"], dict):
                params_otimos: dict[str, Any] = {}
                for k, v in dados["melhores_parametros"].items():
                    params_otimos[k] = v
                    params_otimos[f"melhor_{k}"] = v
                mlflow.log_params(params_otimos)

            modelo = dados.get("modelo")
            if self.logar_modelo and modelo is not None:
                try:
                    mlflow.sklearn.log_model(
                        sk_model=modelo,
                        name="modelo_treinado",
                    )
                except TypeError:
                    mlflow.sklearn.log_model(
                        sk_model=modelo,
                        artifact_path="modelo_treinado",
                    )
        except Exception as e:
            logger.warning(f"Falha ao salvar modelo no MLflow: {e}")

    def _ao_finalizar_avaliacao(self, dados: dict[str, Any]) -> None:
        """Registra métricas estatísticas e de negócio e relatórios textuais, finalizando a run."""
        if not self._run_ativa:
            return

        try:
            # 1. Métricas Estatísticas, Financeiras e de Avaliação do Grid
            metricas: dict[str, float] = {}

            if "melhor_score" in dados and isinstance(dados["melhor_score"], (int, float)):
                metricas["grid_melhor_score_r2"] = float(dados["melhor_score"])

            if "metricas_regressao" in dados and isinstance(dados["metricas_regressao"], dict):
                for k, v in dados["metricas_regressao"].items():
                    if isinstance(v, (int, float)):
                        metricas[f"reg_{k}"] = float(v)

            if "metricas_financeiras" in dados and isinstance(dados["metricas_financeiras"], dict):
                for k, v in dados["metricas_financeiras"].items():
                    if isinstance(v, (int, float)):
                        metricas[f"fin_{k}"] = float(v)

            if "validacao_cruzada" in dados and isinstance(dados["validacao_cruzada"], dict):
                for k, v in dados["validacao_cruzada"].items():
                    if isinstance(v, (int, float)):
                        metricas[f"cv_{k}"] = float(v)

            if "cv_folds_metricas" in dados and isinstance(dados["cv_folds_metricas"], dict):
                for k, v in dados["cv_folds_metricas"].items():
                    if isinstance(v, (int, float)):
                        metricas[k] = float(v)

            # Métricas de Feature Importance
            if "resultado_importancia" in dados and isinstance(dados["resultado_importancia"], dict):
                ranking_items = dados["resultado_importancia"].get("ranking_features", [])
                if isinstance(ranking_items, list):
                    for item in ranking_items:
                        feat_clean = item["feature"].replace(" ", "_").replace("/", "_")
                        metricas[f"feat_imp_r2_drop_{feat_clean}"] = float(item["queda_r2_pct"])
                        metricas[f"feat_imp_peso_rel_{feat_clean}"] = float(item["peso_relativo_pct"])

            if metricas:
                mlflow.log_metrics(metricas)

            if "diagnostico_status" in dados:
                mlflow.set_tag("diagnostico_ajuste", str(dados["diagnostico_status"]))

            if "cv_estrategia" in dados:
                mlflow.set_tag("cv_estrategia", str(dados["cv_estrategia"]))

            if "cv_n_splits" in dados:
                mlflow.set_tag("cv_n_splits", str(dados["cv_n_splits"]))

            # 2. Artefatos Textuais e Gráficos
            if self.logar_artefatos:
                # Log do Gráfico de Diagnóstico (Curva de Aprendizado e Comparativo Treino vs Teste)
                figura = dados.get("figura_diagnostico")
                if figura is not None:
                    try:
                        mlflow.log_figure(figura, "graficos/diagnostico_ajuste.png")
                    except Exception as ef:
                        logger.warning(f"Falha ao salvar figura no MLflow: {ef}")
                    finally:
                        try:
                            import matplotlib.pyplot as plt
                            plt.close(figura)
                        except Exception:
                            pass

                # Log do Gráfico de Importância de Features
                figura_imp = dados.get("figura_importancia")
                if figura_imp is not None:
                    try:
                        mlflow.log_figure(figura_imp, "graficos/importancia_features.png")
                    except Exception as efi:
                        logger.warning(f"Falha ao salvar figura de importância de features no MLflow: {efi}")
                    finally:
                        try:
                            import matplotlib.pyplot as plt
                            plt.close(figura_imp)
                        except Exception:
                            pass

                caminho_figura = dados.get("caminho_figura_diagnostico")
                if caminho_figura and os.path.exists(str(caminho_figura)) and figura is None:
                    try:
                        mlflow.log_artifact(str(caminho_figura), "graficos")
                    except Exception as ea:
                        logger.warning(f"Falha ao salvar arquivo de imagem no MLflow: {ea}")

                diagnostico_txt = dados.get("diagnostico_ajuste_texto")
                if isinstance(diagnostico_txt, str) and diagnostico_txt.strip():
                    mlflow.log_text(diagnostico_txt, "diagnostico_underfitting_overfitting.txt")

                relatorio_imp = dados.get("relatorio_importancia")
                if isinstance(relatorio_imp, str) and relatorio_imp.strip():
                    mlflow.log_text(relatorio_imp, "importancia_features.txt")

                tabela_grid = dados.get("tabela_grid")
                if isinstance(tabela_grid, str) and tabela_grid.strip():
                    mlflow.log_text(tabela_grid, "tabela_grid_search.txt")

                relatorio_cv = dados.get("relatorio_validacao_cruzada")
                if isinstance(relatorio_cv, str) and relatorio_cv.strip():
                    mlflow.log_text(relatorio_cv, "resultado_validacao_cruzada.txt")

                relatorio = dados.get("relatorio_financeiro")
                if isinstance(relatorio, str) and relatorio.strip():
                    mlflow.log_text(relatorio, "relatorio_saude_financeira.txt")

                equacao = dados.get("equacao_da_reta")
                if isinstance(equacao, str) and equacao.strip():
                    mlflow.log_text(equacao, "equacao_da_reta.txt")

        except Exception as e:
            logger.warning(f"Falha ao salvar métricas ou artefatos no MLflow: {e}")
        finally:
            mlflow.end_run()
            self._run_ativa = False

    def _ao_ocorrer_erro(self, dados: dict[str, Any]) -> None:
        """Encerra a run com status de falha em caso de exceção no pipeline."""
        if self._run_ativa:
            try:
                erro_msg = str(dados.get("erro", "Erro desconhecido no pipeline"))
                mlflow.set_tag("erro_execucao", erro_msg)
                mlflow.end_run(status="FAILED")
            except Exception as e:
                logger.warning(f"Falha ao encerrar run com erro no MLflow: {e}")
            finally:
                self._run_ativa = False
