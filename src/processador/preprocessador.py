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

from .ipreprocessador import DadosProcessados, IPreprocessador, IScaler

TipoScaler = Literal["standard", "minmax", "robust", "maxabs"]


class Preprocessador(IPreprocessador):
    """Classe responsável pelo pré-processamento de dados para modelagem de Machine Learning.
    
    Como a base já é fornecida tratada, esta classe foca exclusivamente na configuração
    do tipo de processamento para os modelos:
    
    - tipo_scaler:
        * 'standard': StandardScaler (z-score: média 0, std 1; padrão para OLS/Ridge/Lasso)
        * 'robust'  : RobustScaler (mediana/IQR: ideal para outliers no mercado imobiliário)
        * 'minmax'  : MinMaxScaler (normaliza entre [0, 1])
        * 'maxabs'  : MaxAbsScaler (normaliza pelo valor absoluto máximo)
    - escalar:
        * True (para modelos lineares, baseados em distância ou redes neurais)
        * False (para modelos baseados em árvore como Random Forest, XGBoost)
    - drop_first:
        * True (evita Dummy Variable Trap / multicolinearidade perfeita em OLS)
        * False (mantém todas as categorias dummy)
    - tamanho_teste:
        * Proporção de divisão para teste (ex: 0.2 para 20% teste e 80% treino)
    - random_state:
        * Semente para reprodutibilidade da partição dos dados
    """

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
        tipo_scaler: TipoScaler | Any = "standard",
        escalar: bool = True,
        drop_first: bool = True,
        tamanho_teste: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.__coluna_alvo = coluna_alvo
        self.__tipo_scaler = tipo_scaler
        self.__escalar = escalar
        self.__drop_first = drop_first
        self.__tamanho_teste = tamanho_teste
        self.__random_state = random_state

        self.__base: pd.DataFrame | None = None
        if base is not None:
            self.base = base

        self.__scaler: IScaler | None = None
        self.__colunas_features: list[str] = []

    # -------------------------------------------------------------------------
    # Propriedades de Configuração
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
        return self.__coluna_alvo

    @coluna_alvo.setter
    def coluna_alvo(self, novo_alvo: str) -> None:
        self.__coluna_alvo = novo_alvo

    @property
    def tipo_scaler(self) -> TipoScaler | Any:
        return self.__tipo_scaler

    @tipo_scaler.setter
    def tipo_scaler(self, novo_scaler: TipoScaler | Any) -> None:
        self.__tipo_scaler = novo_scaler

    @property
    def escalar(self) -> bool:
        return self.__escalar

    @escalar.setter
    def escalar(self, valor: bool) -> None:
        self.__escalar = valor

    @property
    def drop_first(self) -> bool:
        return self.__drop_first

    @drop_first.setter
    def drop_first(self, valor: bool) -> None:
        self.__drop_first = valor

    @property
    def tamanho_teste(self) -> float:
        return self.__tamanho_teste

    @tamanho_teste.setter
    def tamanho_teste(self, valor: float) -> None:
        self.__tamanho_teste = valor

    @property
    def random_state(self) -> int:
        return self.__random_state

    @random_state.setter
    def random_state(self, valor: int) -> None:
        self.__random_state = valor

    @property
    def scaler(self) -> IScaler | None:
        """Retorna o scaler ajustado nos dados de treino, se houver escalonamento."""
        return self.__scaler

    @property
    def colunas_features(self) -> list[str]:
        """Retorna os nomes das colunas de features após codificação."""
        return list(self.__colunas_features)

    # -------------------------------------------------------------------------
    # Fábrica Interna de Scalers
    # -------------------------------------------------------------------------

    def _criar_instancia_scaler(self, tipo: TipoScaler | Any | None = None) -> IScaler:
        """Cria e retorna a instância do scaler configurado."""
        alvo = tipo if tipo is not None else self.__tipo_scaler

        if isinstance(alvo, str):
            nome_normalizado = alvo.lower().strip()
            classe_scaler = self.SCALERS_DISPONIVEIS.get(nome_normalizado)
            if classe_scaler is None:
                opcoes = ", ".join(f"'{k}'" for k in self.SCALERS_DISPONIVEIS)
                raise ValueError(
                    f"Tipo de scaler '{alvo}' inválido. Opções válidas: {opcoes}."
                )
            instancia: IScaler = classe_scaler()
            return instancia
        elif hasattr(alvo, "fit_transform") and hasattr(alvo, "transform"):
            return alvo  # type: ignore[no-any-return]
        elif callable(alvo):
            instancia_callable: IScaler = alvo()
            return instancia_callable

        raise TypeError(f"Scaler inválido: {alvo}. Deve ser string ou objeto transformador.")

    # -------------------------------------------------------------------------
    # Etapas do Processamento
    # -------------------------------------------------------------------------

    def separar_features_e_alvo(self, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, np.ndarray]:
        """Separa a matriz de variáveis preditoras (X) e o vetor alvo (y)."""
        origem = df if df is not None else self.__base
        if origem is None:
            raise ValueError("Nenhum DataFrame disponível para separar features e alvo.")
        if self.__coluna_alvo not in origem.columns:
            raise ValueError(f"Coluna alvo '{self.__coluna_alvo}' não encontrada no DataFrame.")

        y = origem[self.__coluna_alvo].to_numpy(dtype=float).ravel()
        x = origem.drop(columns=[self.__coluna_alvo])
        return x, y

    def codificar_categoricas(self, x: pd.DataFrame) -> pd.DataFrame:
        """Aplica One-Hot Encoding com controle da flag drop_first."""
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
        """Particiona os dados segundo o tamanho_teste e random_state configurados."""
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
        """Aplica o tipo de scaler escolhido ajustado no treino e aplicado no teste."""
        scaler: IScaler = self._criar_instancia_scaler(tipo_scaler)
        x_treino_scaled = scaler.fit_transform(x_treino)
        x_teste_scaled = scaler.transform(x_teste)
        self.__scaler = scaler
        return x_treino_scaled, x_teste_scaled

    # -------------------------------------------------------------------------
    # Pipeline Principal
    # -------------------------------------------------------------------------

    def realizar_preprocessamento(self, base: pd.DataFrame | None = None) -> DadosProcessados:
        """Executa o processamento focado na preparação dos dados para o modelo.
        
        Etapas:
        1. Separação de X (features) e y (alvo)
        2. Codificação One-Hot (com ou sem drop_first)
        3. Particionamento Treino e Teste (conforme tamanho_teste e random_state)
        4. Escalonamento (conforme flag escalar e tipo_scaler)
        """
        if base is not None:
            self.base = base

        if self.__base is None:
            raise ValueError("Nenhuma base de dados foi fornecida para o pré-processamento.")

        # 1. Separar X e y
        x_df, y = self.separar_features_e_alvo(self.__base)

        # 2. Codificar variáveis categóricas
        x_codificado = self.codificar_categoricas(x_df)

        # 3. Divisão em Treino e Teste
        x_treino, x_teste, y_treino, y_teste = self.dividir_treino_teste(x_codificado, y)

        # 4. Escalonamento (se ativado)
        if self.__escalar:
            x_treino, x_teste = self.escalonar_dados(x_treino, x_teste)

        return DadosProcessados(
            x_treino=x_treino,
            x_teste=x_teste,
            y_treino=y_treino,
            y_teste=y_teste,
        )
