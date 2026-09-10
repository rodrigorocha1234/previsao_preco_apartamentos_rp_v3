from typing import Any, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from .ipreprocessador import DadosProcessados, IPreprocessador

TipoScaler = Literal["standard", "minmax", "robust", "maxabs"]


class Preprocessador(IPreprocessador):
    """Classe responsável pelo pré-processamento de dados para modelos de regressão linear.
    
    Trata variáveis categóricas (One-Hot com drop_first para evitar Dummy Variable Trap),
    realiza imputação de valores ausentes, remove colunas não preditivas ou com vazamento de dados,
    divide os dados em treino/teste e aplica escalonamento configurável (StandardScaler, MinMaxScaler,
    RobustScaler, MaxAbsScaler ou customizado) sem vazamento de dados.
    """

    COLUNAS_PADRAO_IGNORAR: list[str] = [
        "Código",
        "codigo",
        "Apartamento",
        "apartamento",
        "valor_m2",
    ]

    SCALERS_DISPONIVEIS: dict[str, type] = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "maxabs": MaxAbsScaler,
    }

    def __init__(
        self,
        base: pd.DataFrame | None = None,
        coluna_alvo: str = "Valor_da_Venda",
        tamanho_teste: float = 0.2,
        random_state: int = 42,
        escalar: bool = True,
        tipo_scaler: TipoScaler | Any = "standard",
        drop_first: bool = True,
        colunas_ignorar: list[str] | None = None,
    ) -> None:
        self.__coluna_alvo = coluna_alvo
        self.__tamanho_teste = tamanho_teste
        self.__random_state = random_state
        self.__escalar = escalar
        self.__tipo_scaler = tipo_scaler
        self.__drop_first = drop_first
        self.__colunas_ignorar = (
            colunas_ignorar if colunas_ignorar is not None else self.COLUNAS_PADRAO_IGNORAR
        )

        self.__base: pd.DataFrame | None = None
        if base is not None:
            self.base = base

        self.__scaler: Any | None = None
        self.__colunas_features: list[str] = []

    # -------------------------------------------------------------------------
    # Propriedades
    # -------------------------------------------------------------------------

    @property
    def base(self) -> pd.DataFrame | None:
        """Retorna uma cópia da base de dados atual, ou None se não atribuída."""
        return self.__base.copy() if self.__base is not None else None

    @base.setter
    def base(self, nova_base: pd.DataFrame | None) -> None:
        """Define ou atualiza a base de dados a ser pré-processada."""
        if nova_base is None:
            self.__base = None
            return
        if nova_base.empty:
            raise ValueError("O DataFrame fornecido para o pré-processamento está vazio.")
        if self.__coluna_alvo not in nova_base.columns:
            raise ValueError(f"A coluna alvo '{self.__coluna_alvo}' não está presente no DataFrame.")
        self.__base = nova_base.copy()

    @property
    def coluna_alvo(self) -> str:
        """Retorna o nome da coluna alvo."""
        return self.__coluna_alvo

    @property
    def tipo_scaler(self) -> TipoScaler | Any:
        """Retorna o tipo de scaler configurado."""
        return self.__tipo_scaler

    @tipo_scaler.setter
    def tipo_scaler(self, novo_scaler: TipoScaler | Any) -> None:
        """Configura um novo tipo de scaler ('standard', 'minmax', 'robust', 'maxabs' ou objeto customizado)."""
        self.__tipo_scaler = novo_scaler

    @property
    def scaler(self) -> Any | None:
        """Retorna o scaler ajustado nos dados de treino, se houver escalonamento."""
        return self.__scaler

    @property
    def colunas_features(self) -> list[str]:
        """Retorna os nomes das colunas de features após codificação e transformações."""
        return list(self.__colunas_features)

    @property
    def colunas_ignorar(self) -> list[str]:
        """Retorna a lista de colunas a serem ignoradas/removidas."""
        return list(self.__colunas_ignorar)

    # -------------------------------------------------------------------------
    # Fábrica Interna de Scalers
    # -------------------------------------------------------------------------

    def _criar_instancia_scaler(self, tipo: TipoScaler | Any | None = None) -> Any:
        """Cria e retorna uma instância do scaler especificado."""
        alvo = tipo if tipo is not None else self.__tipo_scaler

        if isinstance(alvo, str):
            nome_normalizado = alvo.lower().strip()
            classe_scaler = self.SCALERS_DISPONIVEIS.get(nome_normalizado)
            if classe_scaler is None:
                opcoes = ", ".join(f"'{k}'" for k in self.SCALERS_DISPONIVEIS)
                raise ValueError(
                    f"Tipo de scaler '{alvo}' inválido. Opções válidas: {opcoes}."
                )
            return classe_scaler()
        elif hasattr(alvo, "fit_transform") and hasattr(alvo, "transform"):
            return alvo
        elif callable(alvo):
            return alvo()

        raise TypeError(f"Scaler inválido: {alvo}. Deve ser string ou objeto transformador.")

    # -------------------------------------------------------------------------
    # Métodos Granulares de Cada Etapa (Uso Individual)
    # -------------------------------------------------------------------------

    def remover_colunas_desnecessarias(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Demonstra e executa a remoção de identificadores e colunas com vazamento de dados."""
        origem = df if df is not None else self.__base
        if origem is None:
            raise ValueError("Nenhum DataFrame disponível para remover colunas.")

        dados = origem.copy()
        colunas_remover = [
            col for col in self.__colunas_ignorar
            if col in dados.columns and col != self.__coluna_alvo
        ]
        if colunas_remover:
            dados = dados.drop(columns=colunas_remover)
        return dados

    def tratar_valores_ausentes(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Demonstra e executa a imputação de dados ausentes (mediana para numéricas, moda para categóricas)."""
        origem = df if df is not None else self.__base
        if origem is None:
            raise ValueError("Nenhum DataFrame disponível para tratar valores ausentes.")

        dados = origem.copy()
        for col in dados.columns:
            if dados[col].isnull().any():
                if pd.api.types.is_numeric_dtype(dados[col]):
                    mediana = dados[col].median()
                    dados[col] = dados[col].fillna(mediana)
                else:
                    moda = dados[col].mode()
                    valor_preenchimento = moda.iloc[0] if not moda.empty else "Desconhecido"
                    dados[col] = dados[col].fillna(valor_preenchimento)
        return dados

    def separar_features_e_alvo(self, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, np.ndarray]:
        """Demonstra e executa a separação entre a matriz de variáveis preditoras (X) e o vetor alvo (y)."""
        origem = df if df is not None else self.__base
        if origem is None:
            raise ValueError("Nenhum DataFrame disponível para separar features e alvo.")
        if self.__coluna_alvo not in origem.columns:
            raise ValueError(f"Coluna alvo '{self.__coluna_alvo}' não encontrada no DataFrame.")

        y = origem[self.__coluna_alvo].to_numpy(dtype=float).ravel()
        x = origem.drop(columns=[self.__coluna_alvo])
        return x, y

    def codificar_categoricas(self, x: pd.DataFrame) -> pd.DataFrame:
        """Demonstra e executa One-Hot Encoding com drop_first=True para evitar a Dummy Variable Trap."""
        colunas_categoricas = [
            col for col in x.columns
            if not pd.api.types.is_numeric_dtype(x[col])
        ]
        if colunas_categoricas:
            x_codificado = pd.get_dummies(
                x,
                columns=colunas_categoricas,
                drop_first=self.__drop_first,
                dtype=float,
            )
        else:
            x_codificado = x.astype(float)

        self.__colunas_features = list(x_codificado.columns)
        return x_codificado

    def dividir_treino_teste(
        self,
        x: pd.DataFrame | np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Demonstra e executa a partição dos dados em conjuntos de treino e teste."""
        x_treino, x_teste, y_treino, y_teste = train_test_split(
            x,
            y,
            test_size=self.__tamanho_teste,
            random_state=self.__random_state,
        )
        x_tr_arr = x_treino.to_numpy(dtype=float) if isinstance(x_treino, pd.DataFrame) else np.asarray(x_treino, dtype=float)
        x_te_arr = x_teste.to_numpy(dtype=float) if isinstance(x_teste, pd.DataFrame) else np.asarray(x_teste, dtype=float)
        y_tr_arr = np.asarray(y_treino, dtype=float).ravel()
        y_te_arr = np.asarray(y_teste, dtype=float).ravel()
        return x_tr_arr, x_te_arr, y_tr_arr, y_te_arr

    def escalonar_dados(
        self,
        x_treino: np.ndarray,
        x_teste: np.ndarray,
        tipo_scaler: TipoScaler | Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Demonstra e executa o escalonamento dos dados ajustado no treino e aplicado no teste.
        
        Suporta os seguintes scalers:
        - 'standard': StandardScaler (z-score, média 0 e variância 1)
        - 'minmax': MinMaxScaler (intervalo [0, 1])
        - 'robust': RobustScaler (baseado em mediana e IQR, robusto a outliers)
        - 'maxabs': MaxAbsScaler (escala pelo valor absoluto máximo)
        - Ou qualquer instância/classe customizada compatível com a API do scikit-learn.
        """
        self.__scaler = self._criar_instancia_scaler(tipo_scaler)
        x_treino_scaled = self.__scaler.fit_transform(x_treino)
        x_teste_scaled = self.__scaler.transform(x_teste)
        return x_treino_scaled, x_teste_scaled

    # Métodos específicos para cada tipo de scaler
    def escalonar_com_standard_scaler(
        self, x_treino: np.ndarray, x_teste: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aplica padronização z-score com StandardScaler (média=0, std=1)."""
        return self.escalonar_dados(x_treino, x_teste, tipo_scaler="standard")

    def escalonar_com_minmax_scaler(
        self, x_treino: np.ndarray, x_teste: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aplica normalização linear com MinMaxScaler para o intervalo [0, 1]."""
        return self.escalonar_dados(x_treino, x_teste, tipo_scaler="minmax")

    def escalonar_com_robust_scaler(
        self, x_treino: np.ndarray, x_teste: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aplica RobustScaler baseado em mediana e IQR, ideal para presença de outliers."""
        return self.escalonar_dados(x_treino, x_teste, tipo_scaler="robust")

    def escalonar_com_maxabs_scaler(
        self, x_treino: np.ndarray, x_teste: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aplica MaxAbsScaler dividindo pelo valor absoluto máximo."""
        return self.escalonar_dados(x_treino, x_teste, tipo_scaler="maxabs")

    # -------------------------------------------------------------------------
    # Pipeline Completo
    # -------------------------------------------------------------------------

    def realizar_preprocessamento(self, base: pd.DataFrame | None = None) -> DadosProcessados:
        """Executa o pipeline completo orquestrando cada etapa sequencialmente.
        
        Args:
            base: DataFrame opcional. Se fornecido, define a base a ser processada.
            
        Returns:
            DadosProcessados: NamedTuple contendo (x_treino, x_teste, y_treino, y_teste).
        """
        if base is not None:
            self.base = base

        if self.__base is None:
            raise ValueError("Nenhuma base de dados foi fornecida para o pré-processamento.")

        # 1. Remover colunas desnecessárias
        dados_limpos = self.remover_colunas_desnecessarias(self.__base)

        # 2. Tratar valores ausentes
        dados_imputados = self.tratar_valores_ausentes(dados_limpos)

        # 3. Separar X e y
        x_df, y = self.separar_features_e_alvo(dados_imputados)

        # 4. Codificar variáveis categóricas
        x_codificado = self.codificar_categoricas(x_df)

        # 5. Divisão em Treino e Teste
        x_treino, x_teste, y_treino, y_teste = self.dividir_treino_teste(x_codificado, y)

        # 6. Escalonamento configurável (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler)
        if self.__escalar:
            x_treino, x_teste = self.escalonar_dados(x_treino, x_teste)

        return DadosProcessados(
            x_treino=x_treino,
            x_teste=x_teste,
            y_treino=y_treino,
            y_teste=y_teste,
        )

    # -------------------------------------------------------------------------
    # Métodos de Demonstração Didática dos Modos de Uso
    # -------------------------------------------------------------------------

    def demonstrar_passo_a_passo(self, base: pd.DataFrame | None = None) -> dict[str, Any]:
        """Demonstra a execução detalhada de cada etapa, inspecionando dimensões e estados intermediários."""
        if base is not None:
            self.base = base
        if self.__base is None:
            raise ValueError("Nenhuma base de dados para demonstrar.")

        print("=== DEMONSTRAÇÃO PASSO A PASSO DO PRÉ-PROCESSAMENTO ===")
        print(f"0. Base original: {self.__base.shape[0]} linhas, {self.__base.shape[1]} colunas")

        # Etapa 1
        etapa1 = self.remover_colunas_desnecessarias(self.__base)
        print(f"1. Remoção de colunas: {etapa1.shape[1]} colunas restantes (Removidas: {set(self.__base.columns) - set(etapa1.columns)})")

        # Etapa 2
        etapa2 = self.tratar_valores_ausentes(etapa1)
        print(f"2. Imputação de nulos: {etapa2.isnull().sum().sum()} valores nulos restantes")

        # Etapa 3
        x, y = self.separar_features_e_alvo(etapa2)
        print(f"3. Separação X/y: X tem {x.shape[1]} features, y tem {y.shape[0]} valores")

        # Etapa 4
        x_cod = self.codificar_categoricas(x)
        print(f"4. One-Hot Encoding (drop_first={self.__drop_first}): expandido para {x_cod.shape[1]} colunas")

        # Etapa 5
        x_tr, x_te, y_tr, y_te = self.dividir_treino_teste(x_cod, y)
        print(f"5. Divisão treino/teste ({100 * (1 - self.__tamanho_teste):.0f}%/{100 * self.__tamanho_teste:.0f}%): Treino={x_tr.shape[0]}, Teste={x_te.shape[0]}")

        # Etapa 6
        if self.__escalar:
            x_tr, x_te = self.escalonar_dados(x_tr, x_te)
            nome_scaler = self.__scaler.__class__.__name__ if self.__scaler else "Desconhecido"
            print(f"6. Escalonamento ({nome_scaler}): min={x_tr.min():.4f}, max={x_tr.max():.4f}, média={x_tr.mean():.4f}, std={x_tr.std():.4f}")

        dados = DadosProcessados(x_treino=x_tr, x_teste=x_te, y_treino=y_tr, y_teste=y_te)
        print("=== FIM DA DEMONSTRAÇÃO ===")

        return {
            "base_limpa": etapa1,
            "base_imputada": etapa2,
            "features_x": x,
            "alvo_y": y,
            "x_codificado": x_cod,
            "dados_processados": dados,
        }

    @staticmethod
    def demonstrar_uso_base_no_construtor(base: pd.DataFrame) -> DadosProcessados:
        """Modo de uso 1: Instanciando com a base diretamente no construtor."""
        preprocessador = Preprocessador(base=base)
        return preprocessador.realizar_preprocessamento()

    @staticmethod
    def demonstrar_uso_atribuicao_tardia(base: pd.DataFrame) -> DadosProcessados:
        """Modo de uso 2: Instanciando vazio e atribuindo a base posteriormente via property setter."""
        preprocessador = Preprocessador()
        preprocessador.base = base
        return preprocessador.realizar_preprocessamento()

    @staticmethod
    def demonstrar_uso_passagem_direta(base: pd.DataFrame) -> DadosProcessados:
        """Modo de uso 3: Instanciando vazio e passando o DataFrame diretamente no método de execução."""
        preprocessador = Preprocessador()
        return preprocessador.realizar_preprocessamento(base=base)
