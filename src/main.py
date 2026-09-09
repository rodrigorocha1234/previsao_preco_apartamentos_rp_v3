import os

import pandas as pd

from carregador.carregador_csv import CarregadorXLSX
from carregador.icarregador import ICarregador


class PipelineML:

    def __init__(self, carregador_dados: ICarregador[pd.DataFrame]):
        self.__carregador = carregador_dados

    def __carregar_base(self) -> pd.DataFrame:
        base = self.__carregador.carregar()
        return base

    def rodar_treinamento_simples(self):
        base_original = self.__carregar_base()
        print(base_original.head())


if __name__ == '__main__':
    lista = ['Bairro', 'Quartos', 'Banheiros', 'Vagas', 'Metragem', 'Valor_da_Venda']
    caminho_arquivo = os.path.join(os.getcwd(), 'docs', 'bairro_final_v3_engineered.xlsx')
    carregador = CarregadorXLSX(caminho=caminho_arquivo, atributos=lista)

    pml = PipelineML(carregador_dados=carregador)
    pml.rodar_treinamento_simples()
