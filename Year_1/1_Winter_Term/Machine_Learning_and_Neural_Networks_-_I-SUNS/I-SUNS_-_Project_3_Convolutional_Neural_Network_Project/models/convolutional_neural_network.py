import os
import numpy as np
import pandas as pd
from datetime import datetime
from termcolor import colored
from keras import regularizers
from ops.plotter import Plotter
from typing import Optional, Union
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation, \
    MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import DirectoryIterator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix


class ConvolutionalNeuralNetwork:

    def __init__(self, target_size: int) -> None:
        """Initialize the NeuralNetwork Class.

        Args:
            target_size (int) : Size of Target Image

        """

        # Create empty model
        self.__model = Sequential()

        self.__plotter = Plotter()

        self.__target_size = target_size

    # noinspection PyPep8Naming
    def build(
            self,
            mode: str,
            loss_function: str,
            learning_rate: float,
            checkpoint_file: Optional[str] = None
    ) -> None:
        # TODO - Docstring
        """Build the Network based on parameters."""

        self.__model.add(
            Conv2D(
                32,
                (5, 5),
                padding='same',
                input_shape=(
                    self.__target_size,
                    self.__target_size,
                    3
                )
            )
        )

        if mode == 'overfit' or mode == 'normal':
            self.__model.add(Activation('relu'))
            self.__model.add(Conv2D(64, (3, 3), padding='same'))
            self.__model.add(Activation('relu'))
            self.__model.add(MaxPooling2D(pool_size=(2, 2)))
            self.__model.add(Conv2D(64, (3, 3), padding='same'))
            self.__model.add(Activation('relu'))
            self.__model.add(Flatten())
            self.__model.add(Dense(
                1024,
                kernel_regularizer=regularizers.l1(0.0001)
            ))
            self.__model.add(Activation('relu'))
            self.__model.add(Dropout(0.5))
        else:
            self.__model.add(Activation('relu'))
            self.__model.add(Conv2D(64, (3, 3), padding='same'))
            self.__model.add(Activation('relu'))
            self.__model.add(MaxPooling2D(pool_size=(2, 2)))
            self.__model.add(Conv2D(64, (3, 3), padding='same'))
            self.__model.add(Activation('relu'))
            self.__model.add(Flatten())
            if mode[:2] == 'l1':
                self.__model.add(
                    Dense(
                        1024,
                        kernel_regularizer=regularizers.l1(
                            float(mode.split('_')[1])
                        )
                    )
                )
            elif mode[:2] == 'l2':
                self.__model.add(
                    Dense(
                        1024,
                        kernel_regularizer=regularizers.l2(
                            float(mode.split('_')[1])
                        )
                    )
                )
            self.__model.add(Activation('relu'))
            self.__model.add(Dropout(0.5))

        self.__model.add(Dense(30, activation='softmax'))

        if checkpoint_file is not None:
            print(f'Loading weights from {checkpoint_file}')
            self.__model.load_weights(checkpoint_file)

        optimizer = Adam(learning_rate=learning_rate)

        # Build the model
        self.__model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=['accuracy']
        )

        print(self.__model.summary())

    # noinspection PyPep8Naming
    def train(
            self,
            training_data_generator: DirectoryIterator,
            validation_data_generator: DirectoryIterator,
            epochs: int,
            batch_size: int,
            mode: str,
            early_stopping: bool = False,
            monitor: Optional[str] = None,
            patience: Optional[int] = None
    ) -> None:
        # TODO - Docstring
        """Train the Network based on parameters.

        Args:

        Returns:
            pandas.DataFrame : Training History

        """

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

        # SOURCE:
        # https://machinelearningmastery.com/check-point-deep-learning-models-keras/

        checkpoint = ModelCheckpoint(
            filepath=os.path.join(
                os.getcwd(),
                'models',
                'checkpoints',
                f'checkpoint_{mode}_best_'
                f'{datetime.now().strftime("%Y_%m_%d_-_%H:%M:%S")}.hdf5'
            ),
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            mode='min' if ((monitor is not None) and
                           (monitor == 'val_loss')) else 'max',
        )

        self.__model.fit(
            x=training_data_generator,
            steps_per_epoch=training_data_generator.n // batch_size,
            epochs=epochs,
            validation_data=validation_data_generator,
            validation_steps=validation_data_generator.n // batch_size,
            callbacks=[early_stop, checkpoint] if early_stopping else None,
        )

        # Display Training History
        self.__plotter.display_training_history(
            history=pd.DataFrame(self.__model.history.history),
            mode=mode
        )

    # noinspection PyPep8Naming
    def evaluate(
            self,
            testing_data_generator: DirectoryIterator,
            batch_size: int,
            mode: str
    ) -> None:
        # TODO - Docstring
        """Evaluate the Network.

        Args:


        Returns:
            Union[None, numpy.ndarray]: None if we have a Regression Task,
                numpy.ndarray if we have Classification Task

        """

        # Predict based on the Testing Data Set
        network_prediction = self.__model.predict(
            x=testing_data_generator,
            steps=testing_data_generator.n // batch_size
        )

        network_prediction = np.argmax(network_prediction, axis=1)

        # Display Confusion Matrix
        conf_matrix = confusion_matrix(
            y_true=testing_data_generator.classes,
            y_pred=network_prediction
        )

        loss, accuracy = self.__model.evaluate(
            testing_data_generator,
            batch_size=batch_size
        )

        print(f'Model evaluation:\nLoss:{loss}\nAccuracy:{accuracy}')

        print(colored('Confusion Matrix:', 'green'))
        print(conf_matrix, end='\n\n')

        # Display Classification Report
        print(colored('Classification Report:', 'green'))
        print(
            classification_report(
                y_true=testing_data_generator.classes,
                y_pred=network_prediction
            )
        )

        self.__plotter.display_confusion_matrix(
            confusion_matrix=conf_matrix,
            labels=list(testing_data_generator.class_indices.keys()),
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'confusion_matrices',
                f'confusion_matrix_{mode}_'
                f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}'
            )
        )
