from abc import ABC
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .imodelo_previsor import IModeloPrevisor


class Avaliador(ABC):
    def avaliar_acuracia(
        self,
        modelo: IModeloPrevisor,
        X_test: np.ndarray | pd.DataFrame,
        Y_test: np.ndarray | pd.Series,
    ) -> float:
        """ Faz uma predição e avalia o modelo. Poderia parametrizar o tipo de
        avaliação, entre outros.
        """
        predicoes = modelo.predict(X_test)
        return float(accuracy_score(Y_test, predicoes))
