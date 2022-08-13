import pandas as pd
from abc import ABC, abstractmethod
from configs.config import CONFIG


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self) -> None:
        self._config = CONFIG['models']

    # noinspection PyPep8Naming
    @abstractmethod
    def build(
            self,
            X_train: pd.Series,
            loss_function: str,
            hidden_layers: int,
            neuron: int,
            activation_function: str,
            learning_rate: float
    ) -> None:
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def train(
            self,
            X_train: pd.Series,
            y_train: pd.Series,
            X_valid: pd.Series,
            y_valid: pd.Series
    ) -> None:
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def evaluate(self, X_test: pd.Series, y_test: pd.Series, file: str) -> None:
        pass
