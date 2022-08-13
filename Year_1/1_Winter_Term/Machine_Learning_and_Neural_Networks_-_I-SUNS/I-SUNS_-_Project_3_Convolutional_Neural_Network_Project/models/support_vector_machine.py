import os
import numpy as np
import pandas as pd
from typing import Any
from sklearn.svm import SVC
from datetime import datetime
from termcolor import colored
from ops.plotter import Plotter
from keras.preprocessing.image import DirectoryIterator
from sklearn.metrics import classification_report, confusion_matrix


class SupportVectorMachine:

    def __init__(self) -> None:
        """Initialize the SupportVectorMachine Class."""

        self.__model = None

        self.__plotter = Plotter()

    def build(
            self,
            kernel: Any = 'rbf',
            degree: Any = 3,
            gamma: Any = "scale",
            coef0: Any = 0.0,
            tol: Any = 1e-3,
            C: Any = 1.0,
            shrinking: Any = True,
            cache_size: Any = 200,
            verbose: Any = True,
            max_iter: Any = -1
    ) -> None:
        """Build the Support Vector Machine Model.
        Args:
            kernel (Any): Kernel type to be used in the algorithm
            degree (Any): Degree of the polynomial kernel function (‘poly’).
            gamma (Any): Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            coef0 (Any): Independent term in kernel function.
            tol (Any): Tolerance for stopping criterion.
            C (Any): Regularization parameter.
            shrinking (Any): Whether to use the shrinking heuristic.
            cache_size (Any): Specify the size of the kernel cache (in MB).
            verbose (Any): Enable verbose output.
            max_iter (Any): Hard limit on iterations within solver, or -1 for no
                limit.
        """

        # Parameters and inspiration for Docstring:
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

        self.__model = SVC(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter
        )

    def train(self, X_train: pd.Series, y_train: pd.Series) -> None:
        """Train the Support Vector Machine Model.
        Args:
            X_train (pandas.Series): Train Data Feature Set
            y_train (pandas.Series): Train Data Label Set
        """

        self.__model.fit(
            X_train,
            y_train
        )

    def evaluate(
            self,
            X_test: pd.Series,
            y_test: pd.Series,
            mode: str
    ) -> None:
        """Evaluate the Support Vector Machine model with Cross Validation and
        on the Predicted Values.
        Args:
            X_test (pandas.Series): Test Data Feature Set used in the Prediction
                Evaluation
            y_test (pandas.Series): Test Data Label Set used in the Prediction
                Evaluation
            testing_data_generator (DirectoryIterator): Image Data Generator
            mode (str): Execution Mode

        """

        # Predict based on the Testing Data Set
        network_prediction = self.__model.predict(
            X=X_test
        )

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

        self.__plotter.display_confusion_matrix(
            confusion_matrix=conf_matrix,
            labels=list(range(0, len(np.unique(y_test)))),
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'confusion_matrices',
                f'confusion_matrix_svm_{mode}_'
                f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}'
            )
        )
