import unittest
from unittest.mock import MagicMock, patch

from observador.iobservador import IObservadorPipeline
from observador.observador_mlflow import ObservadorMLflow


class ObservadorMock(IObservadorPipeline):
    """Observador espião para validar recebimento de eventos nos testes."""

    def __init__(self) -> None:
        self.eventos_recebidos: list[tuple[str, dict]] = []

    def atualizar(self, evento: str, dados: dict) -> None:
        self.eventos_recebidos.append((evento, dados))


class TestObserverPattern(unittest.TestCase):
    """Testes unitários para o padrão Observer e a classe ObservadorMLflow."""

    def setUp(self) -> None:
        self.observador_mock = ObservadorMock()

    def test_observador_mock_recebe_notificacoes(self) -> None:
        # Simula envio de eventos
        self.observador_mock.atualizar("inicio_treinamento", {"scaler": "robust"})
        self.observador_mock.atualizar("fim_treinamento", {"modelo": "LinearRegression"})

        self.assertEqual(len(self.observador_mock.eventos_recebidos), 2)
        self.assertEqual(self.observador_mock.eventos_recebidos[0][0], "inicio_treinamento")
        self.assertEqual(self.observador_mock.eventos_recebidos[0][1]["scaler"], "robust")
        self.assertEqual(self.observador_mock.eventos_recebidos[1][0], "fim_treinamento")

    @patch("observador.observador_mlflow.mlflow")
    def test_observador_mlflow_fluxo_completo(self, mock_mlflow: MagicMock) -> None:
        observador = ObservadorMLflow(
            tracking_uri="http://localhost:5000",
            experiment_name="teste_experiment",
            run_name="teste_run",
            tags={"ambiente": "teste"},
            logar_modelo=True,
            logar_artefatos=True,
        )

        # 1. Evento de início do treinamento
        dados_inicio = {
            "parametros_processador": {"tipo_scaler": "robust", "escalar": True},
            "parametros_modelo": {"fit_intercept": True},
            "num_features": 8,
            "modelo_nome": "EstrategiaRegressaoLinear",
        }
        observador.atualizar("inicio_treinamento", dados_inicio)

        mock_mlflow.set_tracking_uri.assert_called_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_with("teste_experiment")
        mock_mlflow.start_run.assert_called_with(run_name="teste_run")
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.set_tags.assert_called_once()

        # 2. Evento de fim do treinamento
        modelo_fake = MagicMock()
        dados_treino = {"modelo": modelo_fake}
        observador.atualizar("fim_treinamento", dados_treino)
        mock_mlflow.sklearn.log_model.assert_called_with(
            sk_model=modelo_fake,
            name="modelo_treinado",
        )

        # 3. Evento de fim da avaliação
        dados_avaliacao = {
            "metricas_regressao": {"r2": 0.817, "mae": 111341.69},
            "metricas_financeiras": {"faixa_desconto_sugerida_min_pct": 9.3},
            "relatorio_financeiro": "Texto do relatorio",
            "equacao_da_reta": "Equacao da reta",
        }
        observador.atualizar("fim_avaliacao", dados_avaliacao)

        mock_mlflow.log_metrics.assert_called_once()
        self.assertEqual(mock_mlflow.log_text.call_count, 2)
        mock_mlflow.end_run.assert_called_once()

    @patch("observador.observador_mlflow.mlflow")
    def test_observador_mlflow_evento_erro(self, mock_mlflow: MagicMock) -> None:
        observador = ObservadorMLflow(
            tracking_uri="http://localhost:5000",
            experiment_name="teste_experiment",
        )
        observador._run_ativa = True

        observador.atualizar("erro", {"erro": "Falha na memória"})

        mock_mlflow.set_tag.assert_called_with("erro_execucao", "Falha na memória")
        mock_mlflow.end_run.assert_called_with(status="FAILED")
        self.assertFalse(observador._run_ativa)


if __name__ == "__main__":
    unittest.main()
