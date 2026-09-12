import unittest
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from estrategia_modelo.estrategia_regressao_linear_multipla import EstrategiaRegressaoLinearMultipla
from estrategia_modelo.estrategia_modelo import (
    EstrategiaModelo,
    ResultadoGridSearch,
    ResultadoValidacaoCruzada,
)


class TestEstrategiaRegressaoLinearMultipla(unittest.TestCase):
    """Testes unitários para a EstrategiaRegressaoLinearMultipla."""

    def setUp(self) -> None:
        np.random.seed(42)
        # Cria um dataset sintético de regressão linear múltipla com 4 features:
        # y = 100 + 10*X1 + 5*X2 - 3*X3 + 2*X4 + ruído
        self.n_samples = 100
        self.p_features = 4
        self.x_treino = np.random.randn(self.n_samples, self.p_features)
        coefs_verdadeiros = np.array([10.0, 5.0, -3.0, 2.0])
        ruido = np.random.normal(0, 0.1, size=self.n_samples)
        self.y_treino = 100.0 + self.x_treino @ coefs_verdadeiros + ruido
        self.colunas = ["X1", "X2", "X3", "X4"]

    def test_instanciacao_padrao(self) -> None:
        """Verifica se a estratégia é instanciada com os padrões corretos."""
        estrategia = EstrategiaRegressaoLinearMultipla()
        self.assertIsInstance(estrategia, EstrategiaModelo)
        modelo = estrategia.modelo
        self.assertIsInstance(modelo, LinearRegression)
        assert isinstance(modelo, LinearRegression)
        self.assertTrue(modelo.fit_intercept)
        self.assertFalse(modelo.positive)
        self.assertIn("fit_intercept", estrategia.params)
        self.assertIn("positive", estrategia.params)

    def test_instanciacao_customizada(self) -> None:
        """Verifica se parâmetros customizados são repassados ao LinearRegression."""
        params_custom = {"fit_intercept": [True], "positive": [True]}
        estrategia = EstrategiaRegressaoLinearMultipla(
            fit_intercept=False,
            positive=True,
            params=params_custom,
        )
        modelo = estrategia.modelo
        self.assertIsInstance(modelo, LinearRegression)
        assert isinstance(modelo, LinearRegression)
        self.assertFalse(modelo.fit_intercept)
        self.assertTrue(modelo.positive)
        self.assertEqual(estrategia.params, params_custom)

    def test_treinar_modelo_multiplo(self) -> None:
        """Verifica o ajuste do modelo com múltiplas features preditoras."""
        estrategia = EstrategiaRegressaoLinearMultipla()
        modelo_ajustado = estrategia.treinar_modelo_simples(self.x_treino, self.y_treino)

        self.assertIsNotNone(modelo_ajustado)
        self.assertIsInstance(modelo_ajustado, LinearRegression)
        assert isinstance(modelo_ajustado, LinearRegression)
        self.assertTrue(hasattr(modelo_ajustado, "coef_"))
        self.assertTrue(hasattr(modelo_ajustado, "intercept_"))

        # O modelo deve possuir exatamente p coeficientes angulares
        self.assertEqual(len(modelo_ajustado.coef_), self.p_features)
        self.assertAlmostEqual(float(modelo_ajustado.intercept_), 100.0, delta=0.5)
        self.assertAlmostEqual(float(modelo_ajustado.coef_[0]), 10.0, delta=0.5)
        self.assertAlmostEqual(float(modelo_ajustado.coef_[1]), 5.0, delta=0.5)
        self.assertAlmostEqual(float(modelo_ajustado.coef_[2]), -3.0, delta=0.5)
        self.assertAlmostEqual(float(modelo_ajustado.coef_[3]), 2.0, delta=0.5)

    def test_predicao_multipla(self) -> None:
        """Verifica se as predições geradas com múltiplas features têm alta precisão."""
        estrategia = EstrategiaRegressaoLinearMultipla()
        modelo = estrategia.treinar_modelo_simples(self.x_treino, self.y_treino)
        y_pred = modelo.predict(self.x_treino)

        self.assertEqual(len(y_pred), self.n_samples)
        from sklearn.metrics import r2_score
        r2 = r2_score(self.y_treino, y_pred)
        self.assertGreater(r2, 0.99)

    def test_warning_quando_apenas_uma_feature(self) -> None:
        """Verifica se um aviso amigável é emitido quando fornecida apenas 1 feature."""
        estrategia = EstrategiaRegressaoLinearMultipla()
        x_unica = self.x_treino[:, [0]]  # Apenas 1 coluna

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            estrategia.treinar_modelo_simples(x_unica, self.y_treino)
            self.assertTrue(any(issubclass(item.category, UserWarning) for item in w))
            self.assertTrue(any("regressão simples" in str(item.message) for item in w))

    def test_grid_search_multipla(self) -> None:
        """Verifica a execução de GridSearchCV sobre o modelo linear múltiplo."""
        estrategia = EstrategiaRegressaoLinearMultipla()
        resultado = estrategia.realizar_grid_search(self.x_treino, self.y_treino, cv=3)

        self.assertIsInstance(resultado, ResultadoGridSearch)
        self.assertIn("fit_intercept", resultado.melhores_parametros)
        self.assertIn("positive", resultado.melhores_parametros)
        self.assertGreater(resultado.melhor_score, 0.90)
        self.assertIsInstance(resultado.tabela_resultados, pd.DataFrame)
        self.assertEqual(len(resultado.tabela_resultados), 4)

    def test_validacao_cruzada_multipla(self) -> None:
        """Verifica a execução da validação cruzada k-fold sobre o modelo linear múltiplo."""
        estrategia = EstrategiaRegressaoLinearMultipla()
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        resultado_cv = estrategia.realizar_validacao_cruzada(
            x_completo=self.x_treino,
            y_completa=self.y_treino,
            kfold=kfold,
        )

        self.assertIsInstance(resultado_cv, ResultadoValidacaoCruzada)
        self.assertEqual(resultado_cv.n_splits, 5)
        self.assertGreater(resultado_cv.media_test_r2, 0.95)
        self.assertLess(resultado_cv.media_test_mae, 1.0)
        self.assertLess(resultado_cv.media_test_rmse, 1.0)
        self.assertEqual(len(resultado_cv.tabela_folds), 5)


if __name__ == "__main__":
    unittest.main()
