import os

import pandas as pd

from carregador.carregador_csv import CarregadorXLSX
from carregador.icarregador import ICarregador
from processador.ipreprocessador import IPreprocessador
from processador.preprocessador import Preprocessador


class PipelineML:

    def __init__(
        self,
        carregador_dados: ICarregador[pd.DataFrame],
        preprocessador: IPreprocessador | None = None,
        flag_processamento: bool = True,
    ) -> None:
        self.__carregador = carregador_dados
        self.__flag_processamento = flag_processamento
        self.__preprocessador = preprocessador

    def __carregar_base(self) -> pd.DataFrame:
        base = self.__carregador.carregar()
        return base

    def rodar_treinamento_simples(self):
        base_original = self.__carregar_base()
        if self.__flag_processamento and self.__preprocessador is not None:
            self.__preprocessador.base = base_original
            dados = self.__preprocessador.realizar_preprocessamento()
            print("Colunas de features:", self.__preprocessador.colunas_features)
            print("Formato dos dados processados:")
            print(f"X_treino: {dados.x_treino.shape}, y_treino: {dados.y_treino.shape}")
            print(f"X_teste: {dados.x_teste.shape}, y_teste: {dados.y_teste.shape}")

            x_treino, x_teste, y_treino, y_teste = dados
            print(x_treino)
            return dados
        print(base_original.head())
        return None


if __name__ == '__main__':
    lista = ['Zona', 'Quartos', 'Banheiros', 'Vagas', 'Metragem', 'Valor_da_Venda']
    caminho_arquivo = os.path.join(os.getcwd(), 'docs', 'bairro_final_v3_engineered_bkp.xlsx')
    preprocessador_modelo = Preprocessador()
    carregador = CarregadorXLSX(caminho=caminho_arquivo, atributos=lista)

    pml = PipelineML(carregador_dados=carregador, preprocessador=preprocessador_modelo)
    pml.rodar_treinamento_simples()
