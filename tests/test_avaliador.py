import unittest
import numpy as np

from avaliador.avaliador import Avaliador


class TestAvaliador(unittest.TestCase):

    def setUp(self) -> None:
        self.avaliador = Avaliador()
        self.y_real = np.array([300000.0, 500000.0, 700000.0, 400000.0, 600000.0])
        # Predições com pequenos desvios controlados
        self.y_pred = np.array([310000.0, 480000.0, 720000.0, 390000.0, 610000.0])

    def test_avaliar_metricas_regressao(self) -> None:
        metricas = self.avaliador.avaliar_metricas_regressao(self.y_real, self.y_pred)
        self.assertIn("r2", metricas)
        self.assertIn("mae", metricas)
        self.assertIn("rmse", metricas)
        self.assertIn("mape", metricas)
        self.assertGreater(metricas["r2"], 0.90)
        self.assertLess(metricas["mae"], 25000.0)

    def test_avaliar_saude_financeira(self) -> None:
        fin = self.avaliador.avaliar_saude_financeira(self.y_real, self.y_pred, taxa_comissao=0.06)
        self.assertIn("faixa_desconto_sugerida_min_pct", fin)
        self.assertIn("faixa_desconto_sugerida_max_pct", fin)
        self.assertIn("risco_superavaliacao_pct", fin)
        self.assertIn("risco_subavaliacao_pct", fin)
        self.assertIn("desvio_medio_comissao_reais", fin)
        self.assertIn("assertividade_tolerancia_5pct", fin)

    def test_gerar_relatorio_financeiro(self) -> None:
        relatorio = self.avaliador.gerar_relatorio_financeiro(self.y_real, self.y_pred)
        self.assertIsInstance(relatorio, str)
        self.assertIn("RELATÓRIO DE SAÚDE FINANCEIRA", relatorio)
        self.assertIn("DESEMPENHO ESTATÍSTICO", relatorio)
        self.assertIn("FAIXA DE DESCONTO", relatorio)

    def test_formatar_equacao_reta_sucesso(self) -> None:
        class ModeloMock:
            coef_ = np.array([1000.0, 2000.0])
            intercept_ = 50000.0

        equacao = self.avaliador.formatar_equacao_reta(
            modelo=ModeloMock(),
            colunas_features=["Quartos", "Banheiros"],
        )
        self.assertIn("EQUAÇÃO DA RETA", equacao)
        self.assertIn("Quartos", equacao)
        self.assertIn("Banheiros", equacao)
        self.assertIn("50,000.00", equacao)

    def test_formatar_equacao_reta_com_scaler(self) -> None:
        class ModeloMock:
            coef_ = np.array([1000.0, 2000.0])
            intercept_ = 50000.0

        class ScalerMock:
            center_ = np.array([2.0, 1.0])
            scale_ = np.array([1.0, 2.0])

        equacao = self.avaliador.formatar_equacao_reta(
            modelo=ModeloMock(),
            colunas_features=["Quartos", "Banheiros"],
            scaler=ScalerMock(),
        )
        self.assertIn("EQUAÇÃO DA RETA", equacao)
        self.assertIn("Equação no Espaço Escalonado", equacao)

    def test_formatar_equacao_reta_modelo_sem_coeficientes(self) -> None:
        class ModeloSemCoef:
            pass

        msg = self.avaliador.formatar_equacao_reta(
            modelo=ModeloSemCoef(),
            colunas_features=["Quartos"],
        )
        self.assertIn("não possui coeficientes lineares", msg)


if __name__ == "__main__":
    unittest.main()

