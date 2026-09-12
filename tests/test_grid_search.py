import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from estrategia_modelo.estrategia_regressao_linear import EstrategiaRegressaoLinear
from estrategia_modelo.estrategia_modelo import ResultadoGridSearch
from carregador.icarregador import ICarregador
from processador.ipreprocessador import IPreprocessador, DadosProcessados
from observador.iobservador import IObservadorPipeline
from avaliador.avaliador import Avaliador
from main import PipelineML


class ObservadorMock(IObservadorPipeline):
    """Observador espião para validar eventos do Grid Search."""

    def __init__(self) -> None:
        self.eventos: list[tuple[str, dict]] = []

    def atualizar(self, evento: str, dados: dict) -> None:
        self.eventos.append((evento, dados))


class TestGridSearchRegressaoLinear(unittest.TestCase):
    """Testes unitários para o Grid Search da Regressão Linear."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.X = np.random.rand(50, 4)
        # y linearmente dependente com ruído positivo
        self.y = 100000 + 50000 * self.X[:, 0] + 80000 * self.X[:, 1] + np.random.normal(0, 1000, 50)
        self.estrategia = EstrategiaRegressaoLinear()

    def test_realizar_grid_search_retorno_e_desempacotamento(self) -> None:
        resultado = self.estrategia.realizar_grid_search(self.X, self.y)

        self.assertIsInstance(resultado, ResultadoGridSearch)
        self.assertIn("fit_intercept", resultado.melhores_parametros)
        self.assertIn("positive", resultado.melhores_parametros)
        self.assertIsInstance(resultado.melhor_score, float)
        self.assertIsInstance(resultado.melhor_modelo, LinearRegression)
        self.assertIsInstance(resultado.tabela_resultados, pd.DataFrame)
        self.assertEqual(len(resultado.tabela_resultados), 4)

        # Teste do desempacotamento retrocompatível
        params, score = resultado
        self.assertEqual(params, resultado.melhores_parametros)
        self.assertEqual(score, resultado.melhor_score)

        # Teste de compatibilidade e acesso a grid
        self.assertIsNotNone(resultado.grid)
        self.assertEqual(resultado.melhores_parametros, resultado.grid.best_params_)

    def test_grid_search_pipeline_com_observador(self) -> None:
        # Mock do carregador
        carregador_mock = MagicMock(spec=ICarregador)
        carregador_mock.carregar.return_value = pd.DataFrame({
            "Metragem": [50, 60, 70, 80, 90, 100],
            "Quartos": [1, 2, 2, 3, 3, 4],
            "Banheiros": [1, 1, 2, 2, 3, 3],
            "Vagas": [1, 1, 1, 2, 2, 2],
            "Valor_da_Venda": [200000, 250000, 300000, 350000, 420000, 500000],
        })

        # Mock do pré-processador
        preprocessador_mock = MagicMock(spec=IPreprocessador)
        x_fake_treino = np.array([[50, 1], [60, 2], [70, 2], [80, 3]])
        y_fake_treino = np.array([200000, 250000, 300000, 350000])
        x_fake_teste = np.array([[90, 3], [100, 4]])
        y_fake_teste = np.array([420000, 500000])

        dados_proc = DadosProcessados(
            x_treino=x_fake_treino,
            y_treino=y_fake_treino,
            x_teste=x_fake_teste,
            y_teste=y_fake_teste,
        )
        preprocessador_mock.realizar_preprocessamento.return_value = dados_proc
        preprocessador_mock.colunas_features = ["Metragem", "Quartos"]
        preprocessador_mock.scaler = None
        preprocessador_mock.tipo_scaler = "robust"

        observador_mock = ObservadorMock()
        avaliador = Avaliador()

        pipeline = PipelineML(
            carregador_dados=carregador_mock,
            preprocessador=preprocessador_mock,
            estrategia_modelo=self.estrategia,
            avaliador=avaliador,
            observadores=[observador_mock],
            flag_processamento=True,
        )

        dados, res_grid, y_pred = pipeline.rodar_grid_search()

        self.assertIsNotNone(dados)
        self.assertIsInstance(res_grid, ResultadoGridSearch)
        self.assertEqual(len(y_pred), len(y_fake_teste))

        # Valida que todos os eventos do ciclo de vida foram emitidos
        eventos_emitidos = [e[0] for e in observador_mock.eventos]
        self.assertIn("inicio_treinamento", eventos_emitidos)
        self.assertIn("fim_treinamento", eventos_emitidos)
        self.assertIn("fim_avaliacao", eventos_emitidos)

        # Valida dados do evento inicio_treinamento e fim_avaliacao
        dados_inicio = next(d for ev, d in observador_mock.eventos if ev == "inicio_treinamento")
        self.assertIn("parametros_modelo", dados_inicio)
        self.assertEqual(dados_inicio["tipo_busca"], "grid_search")

        dados_avaliacao = next(d for ev, d in observador_mock.eventos if ev == "fim_avaliacao")
        self.assertIn("melhor_score", dados_avaliacao)
        self.assertIn("melhores_parametros", dados_avaliacao)
        self.assertIn("metricas_financeiras", dados_avaliacao)
        self.assertIn("tabela_grid", dados_avaliacao)


if __name__ == "__main__":
    unittest.main()
