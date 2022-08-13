import os.path
import pandas as pd
from typing import Any
from termcolor import colored
from ops.plotter import Plotter
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class Evaluator:
    def __init__(self) -> None:
        """Initialize the Evaluator Class."""

        # Create an instance of Plotter
        self.__plotter = Plotter()

    def evaluate(
            self,
            X_train: pd.Series,
            y_train: pd.Series,
            X_test: pd.Series,
            y_test: pd.Series,
            mode: str,
            model: Any,
            file_name: str
    ) -> None:
        """Evaluate a Model with Cross Validation and on the Predicted Values.

        Args:
            X_train (pandas.Series): Train Data Feature Set used in the Cross
                Validation
            y_train (pandas.Series): Train Data Label Set used in the Cross
                Validation
            X_test (pandas.Series): Test Data Feature Set used in the Prediction
                Evaluation
            y_test (pandas.Series): Test Data Label Set used in the Prediction
                Evaluation
            mode (str): Mode of Prediction
            model (Any): Regression Model to evaluate
            file_name (str): Save File Name

        """

        # Select (Negative) Mean Squared Error and R2 Score
        scoring = ['neg_mean_squared_error', 'r2']

        # Run Validation
        scores = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            scoring=scoring,
            n_jobs=-1
        )

        # Extract the values from Cross Validation result
        cross_validation_mse = scores['test_neg_mean_squared_error'] * -1
        cross_validation_r2 = scores['test_r2']

        # Display Mean Squared Error (MSE) from Cross Validation
        print(
            colored(
                f'\n{mode} - Mean Squared Error (MSE) from Cross Validation:\n',
                'green'
            ),
            cross_validation_mse
        )

        # Display R2 Score from Cross Validation
        print(
            colored(
                f'\n{mode} - R2 Score from Cross Validation:\n',
                'green'
            ),
            cross_validation_r2
        )

        # Display Residuals Plot
        self.__plotter.display_residuals(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            file_path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'residuals',
                file_name
            )
        )

        # Evaluate model on the Predicted Values

        # Display Info for User
        print(
            colored(f'\n{mode} - Error Values for Predicted Values:', 'green')
        )

        # Make prediction
        network_prediction = model.predict(X_test)

        # Display Mean Absolute Error (MAE) on the Predicted Values
        print(
            colored('\nMean Absolute Error (MAE) :\n', 'green'),
            mean_absolute_error(
                y_true=y_test,
                y_pred=network_prediction
            )
        )

        # Display Mean Squared Error (MSE) on the Predicted Values
        print(
            colored('\nMean Squared Error (MSE) :\n', 'green'),
            mean_squared_error(
                y_true=y_test,
                y_pred=network_prediction
            )
        )

        # Display Root Mean Square Error (RMSE) on the Predicted Values
        print(
            colored('\nRoot Mean Square Error (RMSE) :\n', 'green'),
            mean_squared_error(
                y_true=y_test,
                y_pred=network_prediction
            ) ** 0.5
        )

        # Display R2 Score on the Predicted Values
        print(
            colored('\nR2 Score :\n', 'green'),
            r2_score(
                y_true=y_test,
                y_pred=network_prediction
            )
        )
