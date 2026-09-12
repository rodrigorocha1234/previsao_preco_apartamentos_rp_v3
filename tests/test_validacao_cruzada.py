import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from estrategia_modelo.estrategia_regressao_linear import EstrategiaRegressaoLinear
from estrategia_modelo.estrategia_modelo import ResultadoValidacaoCruzada
from avaliador.avaliador import Avaliador
from observador.iobservador import IObservadorPipeline
from carregador.icarregador import ICarregador
from processador.ipreprocessador import IPreprocessador, DadosProcessados
from main import PipelineML


class CarregadorMock(ICarregador[pd.DataFrame]):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def carregar(self) -> pd.DataFrame:
        return self.df


class PreprocessadorMock(IPreprocessador):
    def __init__(self, dados: DadosProcessados) -> None:
        self.dados = dados
        self.base = pd.DataFrame()
        self._colunas_features = ["F1", "F2"]
        self.tipo_scaler = "robust"
        self.escalar = True
        self.drop_first = True
        self.tamanho_teste = 0.2
        self.random_state = 42

    @property
    def colunas_features(self) -> list[str]:
        return self._colunas_features

    def realizar_preprocessamento(self, base: pd.DataFrame | None = None) -> DadosProcessados:
        return self.dados


class ObservadorMock(IObservadorPipeline):
    def __init__(self) -> None:
        self.eventos: list[tuple[str, dict]] = []

    def atualizar(self, evento: str, dados: dict) -> None:
        self.eventos.append((evento, dados))


class TestValidacaoCruzada(unittest.TestCase):

    def setUp(self) -> None:
        self.estrategia = EstrategiaRegressaoLinear()
        self.avaliador = Avaliador()

        np.random.seed(42)
        # Criar dataset sintético com relação linear clara: y = 2*X1 + 3*X2 + 10
        self.X = np.random.randn(50, 2)
        self.y = 2.0 * self.X[:, 0] + 3.0 * self.X[:, 1] + 10.0 + np.random.randn(50) * 0.1

    def test_realizar_validacao_cruzada_com_kfold(self) -> None:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        resultado = self.estrategia.realizar_validacao_cruzada(
            x_completo=self.X,
            y_completa=self.y,
            kfold=kf,
        )

        self.assertIsInstance(resultado, ResultadoValidacaoCruzada)
        self.assertEqual(resultado.n_splits, 5)
        self.assertGreater(resultado.media_test_r2, 0.95)
        self.assertGreater(resultado.media_train_r2, 0.95)
        self.assertGreater(resultado.media_test_mae, 0.0)
        self.assertGreater(resultado.media_test_rmse, 0.0)
        self.assertEqual(len(resultado.tabela_folds), 5)

        # Testar desempacotamento retrocompatível
        r2_m, r2_s = resultado
        self.assertEqual(r2_m, resultado.media_test_r2)
        self.assertEqual(r2_s, resultado.std_test_r2)

    def test_gerar_relatorio_validacao_cruzada(self) -> None:
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        resultado = self.estrategia.realizar_validacao_cruzada(
            x_completo=self.X,
            y_completa=self.y,
            kfold=kf,
        )

        relatorio = self.avaliador.gerar_relatorio_validacao_cruzada(resultado)
        self.assertIn("RELATÓRIO DE VALIDAÇÃO CRUZADA", relatorio)
        self.assertIn("Score R² Médio", relatorio)
        self.assertIn("DETALHAMENTO INDIVIDUAL POR FOLD", relatorio)
        self.assertIn("Fold | R² Teste", relatorio)
        self.assertIn("PARECER ANALÍTICO DE ESTABILIDADE", relatorio)

    def test_pipeline_rodar_validacao_cruzada(self) -> None:
        dados = DadosProcessados(
            x_treino=self.X[:40],
            x_teste=self.X[40:],
            y_treino=self.y[:40],
            y_teste=self.y[40:],
        )
        carregador = CarregadorMock(pd.DataFrame(self.X))
        preprocessador = PreprocessadorMock(dados)
        observador = ObservadorMock()

        pipeline = PipelineML(
            carregador_dados=carregador,
            preprocessador=preprocessador,
            estrategia_modelo=self.estrategia,
            avaliador=self.avaliador,
            observadores=[observador],
            flag_processamento=True,
        )

        dados_proc, res_cv, modelo = pipeline.rodar_validacao_cruzada(n_splits=4)

        self.assertIsInstance(res_cv, ResultadoValidacaoCruzada)
        self.assertEqual(res_cv.n_splits, 4)

        # Verificar se os eventos foram emitidos para o observador
        nomes_eventos = [e[0] for e in observador.eventos]
        self.assertIn("inicio_treinamento", nomes_eventos)
        self.assertIn("fim_treinamento", nomes_eventos)
        self.assertIn("fim_avaliacao", nomes_eventos)

        # Verificar conteúdo dos dados de fim_avaliacao
        dados_fim = next(e[1] for e in observador.eventos if e[0] == "fim_avaliacao")
        self.assertIn("validacao_cruzada", dados_fim)
        self.assertIn("r2_medio", dados_fim["validacao_cruzada"])
        self.assertIn("relatorio_validacao_cruzada", dados_fim)
        self.assertIn("cv_folds_metricas", dados_fim)
        self.assertEqual(dados_fim["cv_estrategia"], "KFold")


if __name__ == "__main__":
    unittest.main()
