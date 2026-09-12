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

    def test_avaliar_diagnostico_ajuste(self) -> None:
        y_tr_real = np.array([100.0, 200.0, 300.0, 400.0])
        y_tr_pred = np.array([102.0, 198.0, 301.0, 399.0])
        y_te_real = np.array([150.0, 250.0, 350.0, 450.0])
        y_te_pred = np.array([151.0, 249.0, 352.0, 448.0])

        diag = self.avaliador.avaliar_diagnostico_ajuste(
            y_treino_real=y_tr_real,
            y_treino_pred=y_tr_pred,
            y_teste_real=y_te_real,
            y_teste_pred=y_te_pred,
        )
        self.assertIn("status", diag)
        self.assertIn("treino", diag)
        self.assertIn("teste", diag)
        self.assertFalse(diag["possui_overfitting"])
        self.assertIn("r2", diag["treino"])
        self.assertIn("r2", diag["teste"])

        relatorio = self.avaliador.gerar_relatorio_diagnostico_ajuste(diag)
        self.assertIn("DIAGNÓSTICO DE AJUSTE", relatorio)
        self.assertIn("VEREDITO TÉCNICO", relatorio)

    def test_gerar_grafico_diagnostico_ajuste(self) -> None:
        from sklearn.linear_model import LinearRegression
        import matplotlib.figure

        X_tr = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
                         [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0]])
        y_tr = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        X_te = np.array([[9.0, 10.0], [10.0, 11.0]])
        y_te = np.array([90.0, 100.0])

        modelo = LinearRegression()
        modelo.fit(X_tr, y_tr)

        fig = self.avaliador.gerar_grafico_diagnostico_ajuste(
            modelo=modelo,
            x_treino=X_tr,
            y_treino=y_tr,
            x_teste=X_te,
            y_teste=y_te,
            caminho_salvar=None,
        )
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_avaliar_e_gerar_grafico_importancia_features(self) -> None:
        from sklearn.linear_model import LinearRegression
        import matplotlib.figure

        X_tr = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
                         [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0]])
        y_tr = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        X_te = np.array([[9.0, 10.0], [10.0, 11.0], [11.0, 12.0], [12.0, 13.0]])
        y_te = np.array([90.0, 100.0, 110.0, 120.0])

        modelo = LinearRegression()
        modelo.fit(X_tr, y_tr)

        # 1. Avaliar Importância
        res_imp = self.avaliador.avaliar_importancia_features(
            modelo=modelo,
            x_teste=X_te,
            y_teste=y_te,
            colunas_features=["FeatureA", "FeatureB"],
            n_repeats=5,
        )
        self.assertIn("ranking_features", res_imp)
        self.assertEqual(len(res_imp["ranking_features"]), 2)
        self.assertIn("queda_r2_pct", res_imp["ranking_features"][0])
        self.assertIn("peso_relativo_pct", res_imp["ranking_features"][0])

        # 2. Relatório
        relatorio = self.avaliador.gerar_relatorio_importancia_features(res_imp)
        self.assertIn("FEATURE IMPORTANCE", relatorio)
        self.assertIn("FeatureA", relatorio)

        # 3. Gráfico (Múltiplas Features)
        fig_mult = self.avaliador.gerar_grafico_importancia_features(
            resultado_importancia=res_imp,
            nome_modelo="Regressão Linear Múltipla",
        )
        self.assertIsInstance(fig_mult, matplotlib.figure.Figure)

        # 4. Gráfico (Feature Única)
        res_imp_single = {
            "ranking_features": [res_imp["ranking_features"][0]],
            "n_features": 1,
            "n_repeats": 5,
        }
        fig_single = self.avaliador.gerar_grafico_importancia_features(
            resultado_importancia=res_imp_single,
            nome_modelo="Regressão Linear Simples",
        )
        self.assertIsInstance(fig_single, matplotlib.figure.Figure)



class TestAvaliadorRegressaoLinear(unittest.TestCase):
    """Testes específicos do AvaliadorRegressaoLinear (Regressão Simples / Univariada)."""

    def setUp(self) -> None:
        from avaliador import AvaliadorRegressaoLinear
        self.avaliador = AvaliadorRegressaoLinear()

    def test_nome_modelo(self) -> None:
        self.assertEqual(self.avaliador.nome_modelo, "Regressão Linear Simples")

    def test_formatar_equacao_reta_univariada(self) -> None:
        class ModeloSimplesMock:
            coef_ = np.array([2500.0])
            intercept_ = 100000.0

        equacao = self.avaliador.formatar_equacao_reta(
            modelo=ModeloSimplesMock(),
            colunas_features=["Metragem"],
        )
        self.assertIn("REGRESSÃO LINEAR SIMPLES", equacao)
        self.assertIn("100,000.00", equacao)
        self.assertIn("2,500.00 × Metragem", equacao)
        self.assertIn("Modelo Univariado", equacao)

    def test_diagnostico_ajuste_underfitting_severo(self) -> None:
        y_tr_real = np.array([100.0, 200.0, 300.0, 400.0])
        y_tr_pred = np.array([210.0, 220.0, 260.0, 270.0])
        y_te_real = np.array([150.0, 250.0, 350.0, 450.0])
        y_te_pred = np.array([230.0, 240.0, 290.0, 300.0])

        diag = self.avaliador.avaliar_diagnostico_ajuste(
            y_treino_real=y_tr_real,
            y_treino_pred=y_tr_pred,
            y_teste_real=y_te_real,
            y_teste_pred=y_te_pred,
        )
        self.assertTrue(diag["possui_underfitting"])
        self.assertFalse(diag["possui_overfitting"])
        self.assertIn("Underfitting Severo", diag["status"])
        self.assertIn("REGRESSÃO LINEAR SIMPLES", self.avaliador.gerar_relatorio_financeiro(y_te_real, y_te_pred))

    def test_importancia_features_univariada(self) -> None:
        from sklearn.linear_model import LinearRegression
        import matplotlib.figure

        X_te = np.array([[50.0], [70.0], [90.0], [110.0]])
        y_te = np.array([200000.0, 280000.0, 360000.0, 440000.0])

        modelo = LinearRegression()
        modelo.fit(X_te, y_te)

        res_imp = self.avaliador.avaliar_importancia_features(
            modelo=modelo,
            x_teste=X_te,
            y_teste=y_te,
            colunas_features=["Metragem"],
            n_repeats=5,
        )
        self.assertEqual(res_imp["n_features"], 1)
        self.assertEqual(res_imp["ranking_features"][0]["peso_relativo_pct"], 100.0)

        relatorio = self.avaliador.gerar_relatorio_importancia_features(res_imp)
        self.assertIn("Metragem", relatorio)
        self.assertIn("100.00%", relatorio)

        fig = self.avaliador.gerar_grafico_importancia_features(res_imp)
        self.assertIsInstance(fig, matplotlib.figure.Figure)


class TestAvaliadorRegressaoLinearMultipla(unittest.TestCase):
    """Testes específicos do AvaliadorRegressaoLinearMultipla (Regressão Múltipla)."""

    def setUp(self) -> None:
        from avaliador import AvaliadorRegressaoLinearMultipla
        self.avaliador = AvaliadorRegressaoLinearMultipla()

    def test_nome_modelo(self) -> None:
        self.assertEqual(self.avaliador.nome_modelo, "Regressão Linear Múltipla")

    def test_formatar_equacao_reta_multivariada(self) -> None:
        class ModeloMultiploMock:
            coef_ = np.array([500.0, -10000.0, 50000.0])
            intercept_ = -200000.0

        equacao = self.avaliador.formatar_equacao_reta(
            modelo=ModeloMultiploMock(),
            colunas_features=["Metragem", "Quartos", "Vagas"],
        )
        self.assertIn("REGRESSÃO LINEAR MÚLTIPLA", equacao)
        self.assertIn("-200,000.00", equacao)
        self.assertIn("500.00 × Metragem", equacao)
        self.assertIn("10,000.00 × Quartos", equacao)
        self.assertIn("50,000.00 × Vagas", equacao)
        self.assertIn("Zona Centro é a categoria base", equacao)

    def test_importancia_features_multivariada(self) -> None:
        from sklearn.linear_model import LinearRegression
        import matplotlib.figure

        X_tr = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
                         [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0]])
        y_tr = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        X_te = np.array([[9.0, 10.0], [10.0, 11.0], [11.0, 12.0], [12.0, 13.0]])
        y_te = np.array([90.0, 100.0, 110.0, 120.0])

        modelo = LinearRegression()
        modelo.fit(X_tr, y_tr)

        res_imp = self.avaliador.avaliar_importancia_features(
            modelo=modelo,
            x_teste=X_te,
            y_teste=y_te,
            colunas_features=["Quartos", "Banheiros"],
            n_repeats=5,
        )
        self.assertEqual(res_imp["n_features"], 2)
        self.assertEqual(len(res_imp["ranking_features"]), 2)

        relatorio = self.avaliador.gerar_relatorio_importancia_features(res_imp)
        self.assertIn("ANÁLISE DE IMPORTÂNCIA DE RECURSOS", relatorio)
        self.assertIn("Quartos", relatorio)
        self.assertIn("Banheiros", relatorio)

        fig = self.avaliador.gerar_grafico_importancia_features(res_imp)
        self.assertIsInstance(fig, matplotlib.figure.Figure)


class TestAvaliadorFactory(unittest.TestCase):
    """Testes para o padrão Factory Method (AvaliadorFactory)."""

    def test_criar_avaliador_por_estrategia(self) -> None:
        from avaliador import AvaliadorFactory, AvaliadorRegressaoLinear, AvaliadorRegressaoLinearMultipla
        from estrategia_modelo.estrategia_regressao_linear import EstrategiaRegressaoLinear
        from estrategia_modelo.estrategia_regressao_linear_multipla import EstrategiaRegressaoLinearMultipla

        av_simples = AvaliadorFactory.criar_avaliador(EstrategiaRegressaoLinear())
        self.assertIsInstance(av_simples, AvaliadorRegressaoLinear)

        av_multipla = AvaliadorFactory.criar_avaliador(EstrategiaRegressaoLinearMultipla())
        self.assertIsInstance(av_multipla, AvaliadorRegressaoLinearMultipla)

    def test_criar_avaliador_por_quantidade_features(self) -> None:
        from avaliador import AvaliadorFactory, AvaliadorRegressaoLinear, AvaliadorRegressaoLinearMultipla

        av_1 = AvaliadorFactory.criar_avaliador(n_features=1)
        self.assertIsInstance(av_1, AvaliadorRegressaoLinear)

        av_8 = AvaliadorFactory.criar_avaliador(n_features=8)
        self.assertIsInstance(av_8, AvaliadorRegressaoLinearMultipla)

    def test_preservar_instancia_existente(self) -> None:
        from avaliador import AvaliadorFactory, AvaliadorRegressaoLinear

        instancia = AvaliadorRegressaoLinear()
        retorno = AvaliadorFactory.criar_avaliador(instancia)
        self.assertIs(retorno, instancia)


if __name__ == "__main__":
    unittest.main()


