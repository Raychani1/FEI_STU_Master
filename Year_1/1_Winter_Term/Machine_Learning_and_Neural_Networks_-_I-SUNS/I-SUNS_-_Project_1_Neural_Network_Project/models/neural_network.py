import json
import os.path
import pprint
import numpy as np
import pandas as pd
from typing import Optional, Union
from termcolor import colored
from models.base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    explained_variance_score


class NeuralNetwork(BaseModel):

    def __init__(self, network_type: str = 'best') -> None:
        """Initialize the NeuralNetwork Class.

        Args:
            network_type (str) : Type of Neural Network ('regression'/
                'classification')

        """

        super().__init__()

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
            
            X_train (pandas.Series) : Training Set for Input Layer Dimension
            loss_function (str) : Type of loss function
            hidden_layers (int) : Number of Hidden Layers
            neuron (int) : Number of Neurons in Layer
            activation_function (str) : Neuron Activation Function
            learning_rate (float): Learning Rate

        """

        # If no value is passed, read from config file
        learning_rate = learning_rate or self._config['NeuralNetwork']['type'] \
            [self.__type]['train']['learning_rate']

        hidden_layers = hidden_layers or self._config['NeuralNetwork']['type'] \
            [self.__type]['model']['hidden_layers']

        neuron = neuron or self._config['NeuralNetwork']['type'] \
            [self.__type]['model']['neurons_per_layer']

        activation_function = activation_function or self._config \
            ['NeuralNetwork']['type'][self.__type]['model'] \
            ['activation_function']

        # Add first hidden layer and specify the input layer shape
        self.__model.add(
            Dense(
                neuron,
                input_shape=(X_train.shape[1],),
                activation=activation_function)
        )

        for _ in range(1, hidden_layers):
            self.__model.add(Dense(neuron, activation=activation_function))

        self.__model.add(Dense(1, activation='sigmoid'))

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
            X_train (pandas.Series) : Training Feature Set
            y_train (pandas.Series) : Training Label Set
            X_valid (Optional[pandas.Series]) : Validation Feature Set
            y_valid (Optional[pandas.Series]) : Validation Label Set
            early_stopping (bool) : Need for early stopping option
            monitor (Optional[str]) : Value to monitor
            patience (Optional[int]) : Number of patience epochs

        Returns:
            pandas.DataFrame : Training History

        """

        # If no value is passed, read from config file
        batch_size = batch_size or self._config['NeuralNetwork']['type'] \
            [self.__type]['train']['batch_size']

        patience = patience or self._config['NeuralNetwork']['type'] \
            [self.__type]['train']['patience']

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

            epochs=self._config['NeuralNetwork']['type'][self.__type]['train'][
                'epochs'],

            batch_size=batch_size,

            validation_data=(X_valid,
                             y_valid) if ((X_valid is not None) and
                                          (y_valid is not None)) else None,

            callbacks=[early_stop] if early_stopping else None,
        )

        # Return History for Visualization
        history = self.__model.history.history

        # return pd.DataFrame(data=history, columns=history.keys())
        return history

    # noinspection PyPep8Naming
    def evaluate(
            self,
            X_test: pd.Series,
            y_test: pd.Series,
            file: Optional[str] = None
    ) -> Union[None, np.ndarray]:
        """Evaluate the Network.

        Args:
            X_test (pandas.Series) : Testing Feature Set
            y_test (pandas.Series) : Testing Label Set
            file (str): Save File Path

        Returns:
            Union[None, numpy.ndarray]: None if we have a Regression Task,
                numpy.ndarray if we have Classification Task

        """

        # Predict based on the Testing Data Set
        network_prediction = self.__model.predict(x=X_test)

        # Classification task evaluations
        if self.__type == 'classification':
            # Categorize the prediction
            network_prediction = np.round(
                network_prediction
            )

            # Display Confusion Matrix
            conf_matrix = confusion_matrix(
                y_true=y_test,
                y_pred=network_prediction
            )

            print(colored('Confusion Matrix:', 'green'))
            print(conf_matrix, end='\n\n')

            # Display Classification Report
            print(colored('Classification Report:', 'green'))
            print(
                classification_report(
                    y_true=y_test,
                    y_pred=network_prediction
                )
            )

            if file is not None:
                with open(file, 'a') as output_file:
                    output_file.write('Confusion Matrix:\n')
                    output_file.write(str(conf_matrix))

                    output_file.write('\nClassification Report:\n')
                    output_file.write(
                        str(classification_report(
                            y_true=y_test,
                            y_pred=network_prediction
                        ))
                    )

            return conf_matrix

        # Regression task evaluations - Not used in this project
        if self.__type == 'regression':
            # Display Mean Absolute Error (MAE)
            print(f"{colored('Mean Absolute Error (MAE) :', 'green')}\n" +
                  mean_absolute_error(y_true=y_test, y_pred=network_prediction))

            # Display Mean Squared Error (MSE)
            print(f"{colored('Mean Squared Error (MSE) :', 'green')}\n" +
                  mean_squared_error(y_true=y_test, y_pred=network_prediction))

            # Display Root Mean Square Error (RMSE)
            print(f"{colored('Root Mean Square Error (RMSE) :', 'green')}\n" +
                  mean_squared_error(y_true=y_test,
                                     y_pred=network_prediction) ** 0.5)

            # Display Explained Variance Score
            print(f"{colored('Explained Variance Score:', 'green')}\n" +
                  explained_variance_score(y_true=y_test,
                                           y_pred=network_prediction))

    # noinspection PyPep8Naming
    def grid_search(
            self,
            X_train: pd.Series,
            y_train: pd.Series,
            X_valid: pd.Series,
            y_valid: pd.Series,
            X_test: pd.Series,
            y_test: pd.Series,
            loss_function: str,
            early_stopping: bool,
            monitor: str,
            file: str,
            training_history_file: str
    ) -> None:
        """Execute Grid Search.

        Args:
        	X_train (pandas.Series) : Training Feature Set
            y_train (pandas.Series) : Training Label Set
            X_valid (pandas.Series) : Validation Feature Set
            y_valid (pandas.Series) : Validation Label Set
            X_test (pandas.Series) : Testing Feature Set
            y_test (pandas.Series) : Testing Label Set
            loss_function (str) : Loss function
            early_stopping (bool) : Enable Early Stopping
            monitor (str): Value to Monitor
            file (str): Save File Path
            training_history_file (str) : Training History Save File Path
            
        """

        learning_rates = [0.001, 0.0001, 0.00001]
        activation_functions = ['sigmoid', 'relu']
        layers = [2, 3, 4, 5]
        neurons = [16, 32, 64, 128]
        batch_sizes = [512, 1024, len(X_train)]
        patience_list = [25, 40, 55]

        history_dict = dict()

        for learning_rate in learning_rates:
            for activation_function in activation_functions:
                for layer in layers:
                    for neuron in neurons:
                        for batch_size in batch_sizes:
                            for patience in patience_list:
                                with open(file, 'a') as output_file:
                                    output_file.write(
                                        f'Neural Network - '
                                        f'Learning_Rate = {learning_rate}, '
                                        f'Activation_Function = '
                                        f'{activation_function}, '
                                        f'Hidden_Layers = {layer}, '
                                        f'Neurons = {neuron}, '
                                        f'Batch_Size = {batch_size}, '
                                        f'Patience = {patience}\n'
                                    )

                                self.build(
                                    X_train=X_train,
                                    loss_function=loss_function,
                                    hidden_layers=layer,
                                    neuron=neuron,
                                    activation_function=activation_function,
                                    learning_rate=learning_rate
                                )

                                training_history = self.train(
                                    X_train=X_train,
                                    y_train=y_train,
                                    X_valid=X_valid,
                                    y_valid=y_valid,
                                    batch_size=batch_size,
                                    early_stopping=early_stopping,
                                    monitor=monitor,
                                    patience=patience
                                )

                                history_dict[
                                    f'L{learning_rate}_'
                                    f'A{activation_function[0].upper()}_'
                                    f'L{layer}_'
                                    f'N{neuron}_'
                                    f'B{batch_size}_'
                                    f'P{patience}'
                                ] = training_history

                                self.evaluate(
                                    X_test=X_test,
                                    y_test=y_test,
                                    file=file
                                )

                                del self.__model

                                self.__model = Sequential()

        with open(training_history_file, 'w+') as history_file:
            history_file.write(json.dumps(history_dict, indent=2))
