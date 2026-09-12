import unittest
import numpy as np
import pandas as pd

from processador.ipreprocessador import DadosProcessados, IPreprocessador
from processador.preprocessador import Preprocessador
from estrategia_modelo.estrategia_regressao_linear import EstrategiaRegressaoLinear


class TestPreprocessador(unittest.TestCase):

    def setUp(self) -> None:
        # Base já tratada (apenas as variáveis preditoras e a coluna alvo)
        self.df_exemplo = pd.DataFrame({
            "Zona": ["Sul", "Norte", "Sul", "Leste", "Oeste", "Sul", "Norte", "Leste", "Oeste", "Sul"],
            "Quartos": [2, 3, 1, 4, 2, 3, 2, 3, 4, 1],
            "Banheiros": [1, 2, 1, 3, 2, 2, 1, 2, 3, 1],
            "Vagas": [1, 2, 0, 2, 1, 2, 1, 2, 2, 1],
            "Metragem": [55.0, 80.0, 45.0, 120.0, 65.0, 85.0, 60.0, 90.0, 110.0, 50.0],
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

    def test_codificacao_one_hot_drop_first_true(self) -> None:
        # A coluna 'Zona' tem 4 categorias únicas: ['Sul', 'Norte', 'Leste', 'Oeste']
        # Com drop_first=True, devem ser geradas exatamente 3 colunas dummies para Zona
        preprocessador = Preprocessador(self.df_exemplo, drop_first=True)
        preprocessador.realizar_preprocessamento()

        colunas_zona = [col for col in preprocessador.colunas_features if col.startswith("Zona_")]
        self.assertEqual(len(colunas_zona), 3)

    def test_codificacao_one_hot_drop_first_false(self) -> None:
        # Com drop_first=False, devem ser mantidas todas as 4 categorias de Zona
        preprocessador = Preprocessador(self.df_exemplo, drop_first=False)
        preprocessador.realizar_preprocessamento()

        colunas_zona = [col for col in preprocessador.colunas_features if col.startswith("Zona_")]
        self.assertEqual(len(colunas_zona), 4)

    def test_escalonamento_standard_scaler(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo, tipo_scaler="standard", escalar=True)
        dados = preprocessador.realizar_preprocessamento()

        self.assertIsNotNone(preprocessador.scaler)
        medias_treino = np.mean(dados.x_treino, axis=0)
        desvios_treino = np.std(dados.x_treino, axis=0)

        np.testing.assert_allclose(medias_treino, np.zeros_like(medias_treino), atol=1e-7)
        np.testing.assert_allclose(desvios_treino, np.ones_like(desvios_treino), atol=1e-7)

    def test_escalonamento_minmax_scaler(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo, tipo_scaler="minmax", escalar=True)
        dados = preprocessador.realizar_preprocessamento()

        self.assertEqual(preprocessador.scaler.__class__.__name__, "MinMaxScaler")
        self.assertGreaterEqual(dados.x_treino.min(), -1e-7)
        self.assertLessEqual(dados.x_treino.max(), 1.0 + 1e-7)

    def test_escalonamento_robust_scaler(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo, tipo_scaler="robust", escalar=True)
        dados = preprocessador.realizar_preprocessamento()

        self.assertEqual(preprocessador.scaler.__class__.__name__, "RobustScaler")
        self.assertEqual(dados.x_treino.shape[0], 8)

    def test_escalonamento_maxabs_scaler(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo, tipo_scaler="maxabs", escalar=True)
        dados = preprocessador.realizar_preprocessamento()

        self.assertEqual(preprocessador.scaler.__class__.__name__, "MaxAbsScaler")
        self.assertLessEqual(np.max(np.abs(dados.x_treino)), 1.0 + 1e-7)

    def test_sem_escalonamento(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo, escalar=False)
        dados = preprocessador.realizar_preprocessamento()

        self.assertIsNone(preprocessador.scaler)
        self.assertFalse(np.allclose(np.mean(dados.x_treino, axis=0), 0.0, atol=1.0))

    def test_validacao_dataframe_vazio(self) -> None:
        with self.assertRaises(ValueError):
            Preprocessador(pd.DataFrame())

    def test_validacao_coluna_alvo_inexistente(self) -> None:
        with self.assertRaises(ValueError):
            Preprocessador(self.df_exemplo, coluna_alvo="coluna_inexistente")

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

        # 1. Separação de X e y
        x_df, y = preprocessador.separar_features_e_alvo()
        self.assertEqual(len(y), self.df_exemplo.shape[0])
        self.assertNotIn("Valor_da_Venda", x_df.columns)

        # 2. Codificação de categóricas
        x_cod = preprocessador.codificar_categoricas(x_df)
        self.assertTrue(all(np.issubdtype(dtype, np.number) for dtype in x_cod.dtypes))

        # 3. Divisão treino/teste
        x_tr, x_te, y_tr, y_te = preprocessador.dividir_treino_teste(x_cod, y)
        self.assertEqual(x_tr.shape[0], 8)
        self.assertEqual(x_te.shape[0], 2)

        # 4. Escalonamento
        x_tr_sc, x_te_sc = preprocessador.escalonar_dados(x_tr, x_te)
        np.testing.assert_allclose(np.mean(x_tr_sc, axis=0), 0.0, atol=1e-7)

    def test_tipo_scaler_invalido_lanca_excecao(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo, tipo_scaler="scaler_inexistente")
        with self.assertRaises(ValueError):
            preprocessador.realizar_preprocessamento()

    def test_alteracao_dinamica_de_propriedades(self) -> None:
        preprocessador = Preprocessador(self.df_exemplo)
        preprocessador.tipo_scaler = "robust"
        self.assertEqual(preprocessador.tipo_scaler, "robust")

        preprocessador.escalar = False
        self.assertFalse(preprocessador.escalar)

        preprocessador.drop_first = False
        self.assertFalse(preprocessador.drop_first)

        preprocessador.tamanho_teste = 0.3
        self.assertEqual(preprocessador.tamanho_teste, 0.3)


if __name__ == "__main__":
    unittest.main()
