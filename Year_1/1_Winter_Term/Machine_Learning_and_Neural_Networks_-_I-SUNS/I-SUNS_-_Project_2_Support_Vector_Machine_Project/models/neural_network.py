import numpy as np
import pandas as pd
from termcolor import colored
from configs.config import CONFIG
from typing import Optional, Union
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class NeuralNetwork:

    def __init__(self, network_type: str = 'best') -> None:
        """Initialize the NeuralNetwork Class.

        Args:
            network_type (str) : Type of Neural Network ('regression'/
                'classification')

        """

        self.__config = CONFIG['models']["NeuralNetwork"]

        # Set type of Neural Network based on argument
        self.__type = network_type

        # Create empty model
        self.__model = Sequential()

    # noinspection PyPep8Naming
    def build(
            self,
            X_train: pd.Series,
            loss_function: str,
            hidden_layers: Optional[int] = None,
            neuron: Optional[int] = None,
            activation_function: Optional[str] = None,
            learning_rate: Optional[float] = None
    ) -> None:
        """Build the Network based on parameters.

        Args:
            
            X_train (pandas.Series): Training Set for Input Layer Dimension
            loss_function (str): Type of loss function
            hidden_layers (int): Number of Hidden Layers
            neuron (int): Number of Neurons in Layer
            activation_function (str): Neuron Activation Function
            learning_rate (float): Learning Rate

        """

        # If no value is passed, read from config file
        learning_rate = learning_rate or self.__config['type'] \
            [self.__type]['train']['learning_rate']

        hidden_layers = hidden_layers or self.__config['type'] \
            [self.__type]['model']['hidden_layers']

        neuron = neuron or self.__config['type'] \
            [self.__type]['model']['neurons_per_layer']

        activation_function = activation_function or self.__config['type'] \
            [self.__type]['model']['activation_function']

        # Add first hidden layer and specify the input layer shape
        self.__model.add(
            Dense(
                neuron,
                input_shape=(X_train.shape[1],),
                activation=activation_function)
        )

        for _ in range(1, hidden_layers):
            self.__model.add(Dense(neuron, activation=activation_function))

        self.__model.add(Dense(1))

        # Create Adam optimizer
        optimizer = Adam(learning_rate=learning_rate)

        # Build the model
        self.__model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=['accuracy']
        )

    # noinspection PyPep8Naming
    def train(
            self,
            X_train: pd.Series,
            y_train: pd.Series,
            batch_size: Optional[int] = None,
            X_valid: Optional[pd.Series] = None,
            y_valid: Optional[pd.Series] = None,
            early_stopping: bool = False,
            monitor: Optional[str] = None,
            patience: Optional[int] = None
    ) -> pd.DataFrame:
        """Train the Network based on parameters.

        Args:
            batch_size:
            X_train (pandas.Series): Training Feature Set
            y_train (pandas.Series): Training Label Set
            X_valid (Optional[pandas.Series]): Validation Feature Set
            y_valid (Optional[pandas.Series]): Validation Label Set
            early_stopping (bool): Need for early stopping option
            monitor (Optional[str]): Value to monitor
            patience (Optional[int]): Number of patience epochs

        Returns:
            pandas.DataFrame : Training History

        """

        # If no value is passed, read from config file
        batch_size = batch_size or self.__config['type'][self.__type]['train'] \
            ['batch_size']

        patience = patience or self.__config['type'][self.__type]['train'] \
            ['patience']

        # By default disable early stopping
        early_stop = None

        # If there is need for early stopping
        if early_stopping:
            # Create EarlyStopping Callback
            early_stop = EarlyStopping(
                monitor=monitor,
                mode='min' if ((monitor is not None) and
                               (monitor == 'val_loss')) else 'max',
                verbose=1,
                patience=patience
            )

        # Train the model
        self.__model.fit(
            x=X_train,
            y=y_train,

            epochs=self.__config['type'][self.__type]['train']['epochs'],

            batch_size=batch_size,

            validation_data=(X_valid,
                             y_valid) if ((X_valid is not None) and
                                          (y_valid is not None)) else None,

            callbacks=[early_stop] if early_stopping else None,
        )

        return pd.DataFrame.from_dict(self.__model.history.history)

    # noinspection PyPep8Naming
    def evaluate(
            self,
            X_test: pd.Series,
            y_test: pd.Series,
    ) -> None:
        """Evaluate the Network.

        Args:
            X_test (pandas.Series): Testing Feature Set
            y_test (pandas.Series): Testing Label Set

        """

        # Predict based on the Testing Data Set
        network_prediction = self.__model.predict(x=X_test)

        # Display Mean Absolute Error (MAE)
        print(
            colored(
                '\nNeural Network - Mean Absolute Error (MAE) :\n',
                'green'
            ),
            mean_absolute_error(
                y_true=y_test,
                y_pred=network_prediction
            )
        )

        # Display Mean Squared Error (MSE)
        print(
            colored(
                '\nNeural Network - Mean Squared Error (MSE) :\n',
                'green'
            ),
            mean_squared_error(
                y_true=y_test,
                y_pred=network_prediction
            )
        )

        # Display Root Mean Square Error (RMSE)
        print(
            colored(
                '\nNeural Network - Root Mean Square Error (RMSE) :\n',
                'green'
            ),
            mean_squared_error(
                y_true=y_test,
                y_pred=network_prediction
            ) ** 0.5
        )

        # Display R2 Score
        print(
            colored(
                '\nNeural Network - R2 Score :\n',
                'green'
            ),
            r2_score(
                y_true=y_test,
                y_pred=network_prediction
            )
        )
