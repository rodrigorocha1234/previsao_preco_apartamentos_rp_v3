import unittest
import numpy as np
import pandas as pd

from processador.ipreprocessador import DadosProcessados, IPreprocessador
from processador.preprocessador import Preprocessador
from estrategia_modelo.estrategia_regressao_linear import EstrategiaRegressaoLinear


class TestPreprocessador(unittest.TestCase):

    def setUp(self) -> None:
        self.df_exemplo = pd.DataFrame({
            "Código": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "Zona": ["Sul", "Norte", "Sul", "Leste", "Oeste", "Sul", "Norte", "Leste", "Oeste", "Sul"],
            "Quartos": [2, 3, 1, 4, 2, 3, 2, 3, 4, 1],
            "Banheiros": [1, 2, 1, 3, 2, 2, 1, 2, 3, 1],
            "Vagas": [1, 2, 0, 2, 1, 2, 1, 2, 2, 1],
            "Metragem": [55.0, 80.0, 45.0, 120.0, 65.0, 85.0, 60.0, 90.0, 110.0, 50.0],
            "valor_m2": [5000.0, 4500.0, 4800.0, 6000.0, 5200.0, 4600.0, 4900.0, 5100.0, 5800.0, 4700.0],
            "Valor_da_Venda": [275000.0, 360000.0, 216000.0, 720000.0, 338000.0, 391000.0, 294000.0, 459000.0, 638000.0, 235000.0],
        })

    def test_conformidade_protocolo_ipreprocessador(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo)
        self.assertIsInstance(preprocessador, IPreprocessador)

    def test_divisao_treino_teste_dimensoes(self) -> None:
        tamanho_teste = 0.2
        preprocessador = Preprocessador(
            base=self.df_exemplo,
            coluna_alvo="Valor_da_Venda",
            tamanho_teste=tamanho_teste,
            random_state=42,
        )
        dados = preprocessador.realizar_preprocessamento()

        self.assertIsInstance(dados, DadosProcessados)
        # Total de 10 amostras: 8 no treino e 2 no teste
        self.assertEqual(dados.x_treino.shape[0], 8)
        self.assertEqual(dados.x_teste.shape[0], 2)
        self.assertEqual(dados.y_treino.shape[0], 8)
        self.assertEqual(dados.y_teste.shape[0], 2)
        self.assertEqual(dados.x_treino.shape[1], dados.x_teste.shape[1])

    def test_desempacotamento_como_tupla(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo)
        x_tr, x_te, y_tr, y_te = preprocessador.realizar_preprocessamento()

        self.assertIsInstance(x_tr, np.ndarray)
        self.assertIsInstance(x_te, np.ndarray)
        self.assertIsInstance(y_tr, np.ndarray)
        self.assertIsInstance(y_te, np.ndarray)

    def test_remocao_colunas_vazamento_e_identificadores(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo)
        preprocessador.realizar_preprocessamento()

        features = preprocessador.colunas_features
        # Colunas "Código" e "valor_m2" não devem estar presentes nas features
        self.assertNotIn("Código", features)
        self.assertNotIn("valor_m2", features)
        self.assertNotIn("Valor_da_Venda", features)

    def test_codificacao_one_hot_drop_first(self) -> None:
        # A coluna 'Zona' tem 4 categorias únicas: ['Sul', 'Norte', 'Leste', 'Oeste']
        # Com drop_first=True, devem ser geradas exatamente 3 colunas dummies para Zona
        preprocessador = Preprocessador(self.df_exemplo, drop_first=True)
        preprocessador.realizar_preprocessamento()

        colunas_zona = [col for col in preprocessador.colunas_features if col.startswith("Zona_")]
        self.assertEqual(len(colunas_zona), 3)

    def test_escalonamento_standard_scaler(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo, escalar=True)
        dados = preprocessador.realizar_preprocessamento()

        self.assertIsNotNone(preprocessador.scaler)
        # No conjunto de treino, cada feature deve ter média próxima de 0 e desvio padrão próximo de 1
        medias_treino = np.mean(dados.x_treino, axis=0)
        desvios_treino = np.std(dados.x_treino, axis=0)

        np.testing.assert_allclose(medias_treino, np.zeros_like(medias_treino), atol=1e-7)
        np.testing.assert_allclose(desvios_treino, np.ones_like(desvios_treino), atol=1e-7)

    def test_sem_escalonamento(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo, escalar=False)
        dados = preprocessador.realizar_preprocessamento()

        self.assertIsNone(preprocessador.scaler)
        # Metragens sem escala não devem ter média zero
        self.assertFalse(np.allclose(np.mean(dados.x_treino, axis=0), 0.0, atol=1.0))

    def test_imputacao_valores_ausentes(self) -> None:
        df_com_nulos = self.df_exemplo.copy()
        df_com_nulos.loc[0, "Metragem"] = np.nan
        df_com_nulos.loc[1, "Zona"] = np.nan

        preprocessador = Preprocessador(df_com_nulos)
        dados = preprocessador.realizar_preprocessamento()

        self.assertFalse(np.isnan(dados.x_treino).any())
        self.assertFalse(np.isnan(dados.x_teste).any())

    def test_validacao_dataframe_vazio(self) -> None:
        with self.assertRaises(ValueError):
            Preprocessador(pd.DataFrame())

    def test_validacao_coluna_alvo_inexistente(self) -> None:
        with self.assertRaises(ValueError):
            Preprocessador(self.df_exemplo, coluna_alvo="coluna_que_nao_existe")

    def test_treinamento_com_estrategia_regressao_linear(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo)
        dados = preprocessador.realizar_preprocessamento()

        estrategia = EstrategiaRegressaoLinear()
        modelo_treinado = estrategia.treinar_modelo_simples(dados.x_treino, dados.y_treino)

        predicoes = modelo_treinado.predict(dados.x_teste)
        self.assertEqual(predicoes.shape, (dados.x_teste.shape[0],))
        self.assertFalse(np.isnan(predicoes).any())

    def test_preprocessador_sem_base_inicial_com_setter(self) -> None:
        preprocessador = Preprocessador()
        self.assertIsNone(preprocessador.base)
        preprocessador.base = self.df_exemplo
        self.assertIsNotNone(preprocessador.base)
        dados = preprocessador.realizar_preprocessamento()
        self.assertEqual(dados.x_treino.shape[0], 8)

    def test_preprocessador_passagem_direta_no_metodo(self) -> None:
        preprocessador = Preprocessador()
        dados = preprocessador.realizar_preprocessamento(base=self.df_exemplo)
        self.assertEqual(dados.x_treino.shape[0], 8)

    def test_preprocessador_sem_base_lanca_excecao(self) -> None:
        preprocessador = Preprocessador()
        with self.assertRaises(ValueError):
            preprocessador.realizar_preprocessamento()

    def test_metodos_granulares_etapa_a_etapa(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo)

        # 1. Remoção de colunas desnecessárias
        df_limpo = preprocessador.remover_colunas_desnecessarias()
        self.assertNotIn("Código", df_limpo.columns)
        self.assertNotIn("valor_m2", df_limpo.columns)

        # 2. Imputação de nulos
        df_nulos = df_limpo.copy()
        df_nulos.loc[0, "Metragem"] = np.nan
        df_imputado = preprocessador.tratar_valores_ausentes(df_nulos)
        self.assertFalse(df_imputado.isnull().any().any())

        # 3. Separação de X e y
        x_df, y = preprocessador.separar_features_e_alvo(df_limpo)
        self.assertEqual(len(y), df_limpo.shape[0])
        self.assertNotIn("Valor_da_Venda", x_df.columns)

        # 4. Codificação de categóricas
        x_cod = preprocessador.codificar_categoricas(x_df)
        self.assertTrue(all(np.issubdtype(dtype, np.number) for dtype in x_cod.dtypes))

        # 5. Divisão treino/teste
        x_tr, x_te, y_tr, y_te = preprocessador.dividir_treino_teste(x_cod, y)
        self.assertEqual(x_tr.shape[0], 8)
        self.assertEqual(x_te.shape[0], 2)

        # 6. Escalonamento
        x_tr_sc, x_te_sc = preprocessador.escalonar_dados(x_tr, x_te)
        np.testing.assert_allclose(np.mean(x_tr_sc, axis=0), 0.0, atol=1e-7)

    def test_demonstrar_passo_a_passo(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo)
        resultado = preprocessador.demonstrar_passo_a_passo()
        self.assertIn("base_limpa", resultado)
        self.assertIn("base_imputada", resultado)
        self.assertIn("features_x", resultado)
        self.assertIn("alvo_y", resultado)
        self.assertIn("x_codificado", resultado)
        self.assertIn("dados_processados", resultado)
        self.assertIsInstance(resultado["dados_processados"], DadosProcessados)

    def test_demonstrar_modos_de_uso_estaticos(self) -> None:
        d1 = Preprocessador.demonstrar_uso_base_no_construtor(self.df_exemplo)
        d2 = Preprocessador.demonstrar_uso_atribuicao_tardia(self.df_exemplo)
        d3 = Preprocessador.demonstrar_uso_passagem_direta(self.df_exemplo)

        self.assertIsInstance(d1, DadosProcessados)
        self.assertIsInstance(d2, DadosProcessados)
        self.assertIsInstance(d3, DadosProcessados)
        self.assertEqual(d1.x_treino.shape, d2.x_treino.shape)
        self.assertEqual(d2.x_treino.shape, d3.x_treino.shape)

    def test_minmax_scaler(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo, tipo_scaler="minmax")
        dados = preprocessador.realizar_preprocessamento()
        self.assertEqual(preprocessador.scaler.__class__.__name__, "MinMaxScaler")
        self.assertGreaterEqual(dados.x_treino.min(), -1e-7)
        self.assertLessEqual(dados.x_treino.max(), 1.0 + 1e-7)

    def test_robust_scaler(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo, tipo_scaler="robust")
        dados = preprocessador.realizar_preprocessamento()
        self.assertEqual(preprocessador.scaler.__class__.__name__, "RobustScaler")
        self.assertEqual(dados.x_treino.shape[0], 8)

    def test_maxabs_scaler(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo, tipo_scaler="maxabs")
        dados = preprocessador.realizar_preprocessamento()
        self.assertEqual(preprocessador.scaler.__class__.__name__, "MaxAbsScaler")
        self.assertLessEqual(np.max(np.abs(dados.x_treino)), 1.0 + 1e-7)

    def test_metodos_especificos_de_escalonamento(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo)
        x_tr = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        x_te = np.array([[2.0, 15.0]])

        # StandardScaler
        tr_std, _ = preprocessador.escalonar_com_standard_scaler(x_tr, x_te)
        np.testing.assert_allclose(tr_std.mean(axis=0), [0.0, 0.0], atol=1e-7)

        # MinMaxScaler
        tr_mm, _ = preprocessador.escalonar_com_minmax_scaler(x_tr, x_te)
        self.assertAlmostEqual(float(tr_mm.min()), 0.0)
        self.assertAlmostEqual(float(tr_mm.max()), 1.0)

        # RobustScaler
        tr_rb, _ = preprocessador.escalonar_com_robust_scaler(x_tr, x_te)
        self.assertEqual(tr_rb.shape, x_tr.shape)

        # MaxAbsScaler
        tr_ma, _ = preprocessador.escalonar_com_maxabs_scaler(x_tr, x_te)
        self.assertAlmostEqual(float(tr_ma.max()), 1.0)

    def test_tipo_scaler_invalido_lanca_excecao(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo, tipo_scaler="scaler_inexistente")
        with self.assertRaises(ValueError):
            preprocessador.realizar_preprocessamento()


if __name__ == "__main__":
    unittest.main()



