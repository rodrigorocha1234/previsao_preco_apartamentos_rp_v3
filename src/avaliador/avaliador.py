from abc import ABC
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .imodelo_previsor import IModeloPrevisor


class Avaliador(ABC):
    def avaliar_acuracia(
        self,
        modelo: IModeloPrevisor,
        x_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
    ) -> float:
        """ Faz uma predição e avalia o modelo. Poderia parametrizar o tipo de
        avaliação, entre outros.
        """
        predicoes = modelo.predict(x_test)
        return float(accuracy_score(y_test, predicoes))
