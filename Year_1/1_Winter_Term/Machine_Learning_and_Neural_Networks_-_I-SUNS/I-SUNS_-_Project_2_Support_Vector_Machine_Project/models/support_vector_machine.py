import pandas as pd
from typing import Any
from sklearn.svm import SVR
from datetime import datetime
from termcolor import colored
from ops.evaluator import Evaluator
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor


class SupportVectorMachine:

    def __init__(self) -> None:
        """Initialize the SupportVectorMachine Class."""

        self.__model = None
        self.__evaluator = Evaluator()

    def build(
            self,
            kernel: Any = 'rbf',
            degree: Any = 3,
            gamma: Any = "scale",
            coef0: Any = 0.0,
            tol: Any = 1e-3,
            C: Any = 1.0,
            epsilon: Any = 0.1,
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
            epsilon (Any): Epsilon in the epsilon-SVR model.
            shrinking (Any): Whether to use the shrinking heuristic.
            cache_size (Any): Specify the size of the kernel cache (in MB).
            verbose (Any): Enable verbose output.
            max_iter (Any): Hard limit on iterations within solver, or -1 for no
                limit.

        """

        # Parameters and inspiration for Docstring:
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

        self.__model = SVR(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            epsilon=epsilon,
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
            X_train: pd.Series,
            y_train: pd.Series,
            X_test: pd.Series,
            y_test: pd.Series,
            file_name: str,
            model: Any = None,
    ) -> None:
        """Evaluate the Support Vector Machine model with Cross Validation and
        on the Predicted Values.

        Args:
            X_train (pandas.Series): Train Data Feature Set used in the Cross
                Validation
            y_train (pandas.Series): Train Data Label Set used in the Cross
                Validation
            X_test (pandas.Series): Test Data Feature Set used in the Prediction
                Evaluation
            y_test (pandas.Series): Test Data Label Set used in the Prediction
                Evaluation
            file_name (str): Save File Name
            model (Any): Regression Model to evaluate

        """

        # If no model is provided evaluate self
        if model is None:
            model = self.__model

        # Run Evaluation
        self.__evaluator.evaluate(
            model=model,
            mode=f'Support Vector Machine',
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            file_name=file_name
        )

    def grid_search(
            self,
            X_train: pd.Series,
            y_train: pd.Series,
            X_test: pd.Series,
            y_test: pd.Series
    ) -> None:
        """Run Grid Search and Further Evaluation on the Support Vector Machine
        Model.

        Args:
            X_train (pandas.Series): Train Data Feature Set used for training
                and further evaluation
            y_train (pandas.Series): Train Data Label Set used for training
                and further evaluation
            X_test (pandas.Series): Test Data Feature Set used for further
                evaluation
            y_test (pandas.Series): Test Data Label Set used for further
                evaluation

        """

        # Set the parameters for Grid Search
        param_grid = {
            'C': [0.1, 1, 100],
            'gamma': [0.1, 0.01, 'scale'],
            'epsilon': [0.1, 0.01, 0.001]
        }

        # Single new line for a more pleasant view
        print()

        # Start the timer
        start_time = datetime.now()

        # Run Grid Search
        grid = GridSearchCV(
            SVR(),
            param_grid=param_grid,
            verbose=10,
            n_jobs=-1,
            scoring=['neg_mean_squared_error', 'r2'],
            refit='r2'
        )

        # Train the result estimator
        grid.fit(X=X_train, y=y_train)

        # Stop the timer and display the time difference
        end_time = datetime.now()

        print(
            f'\n{colored("Grid Search Completed in: ", "green")}'
            f'{end_time - start_time}'
        )

        # Display the Best Parameters Combination in the Search
        print(
            colored(
                '\nBest Parameters in Grid Search:\n',
                'green'
            ),
            grid.best_params_,
            end='\n\n'
        )

        # Display the Best Estimator in the Search
        print(
            colored(
                'Best Estimator in Grid Search:\n',
                'green'
            ),
            grid.best_estimator_,
            end='\n\n'
        )

        # Send the Best Estimator for Further Evaluation ( Cross Validation and
        # Prediction Evaluation )
        self.evaluate(
            model=grid.best_estimator_,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            file_name='grid_search.png'
        )

    def bagging_regression(
            self,
            X_train: pd.Series,
            y_train: pd.Series,
            X_test: pd.Series,
            y_test: pd.Series
    ) -> None:
        """Run Bagging Regression and Evaluate.

        Args:
            X_train (pandas.Series): Training Feature Set
            y_train (pandas.Series): Training Label Set
            X_test (pandas.Series): Testing Feature Set
            y_test (pandas.Series): Testing Label Set

        """

        print(colored('Bagging Regression\n', 'green'))

        # Start the timer
        start_time = datetime.now()

        bagging = BaggingRegressor(
            base_estimator=self.__model,
            n_jobs=-1
        )

        bagging.fit(X=X_train, y=y_train)

        # Stop the timer and display the time difference
        end_time = datetime.now()

        print(
            f'\n{colored("Bagging Regression Completed in: ", "green")}'
            f'{end_time - start_time}'
        )

        self.evaluate(
            model=bagging,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            file_name='svm_bagging.png'
        )

    def boosting_regression(
            self,
            X_train: pd.Series,
            y_train: pd.Series,
            X_test: pd.Series,
            y_test: pd.Series
    ) -> None:
        """Run a Boosting Regression and Evaluate the result.

        Args:
            X_train (pandas.Series): Training Feature Set
            y_train (pandas.Series): Training Label Set
            X_test (pandas.Series): Testing Feature Set
            y_test (pandas.Series): Testing Label Set

        """

        print(colored('Boosting Regression\n', 'green'))

        # Start the timer
        start_time = datetime.now()

        boosting = AdaBoostRegressor(
            base_estimator=self.__model
        )

        boosting.fit(X=X_train, y=y_train)

        # Stop the timer and display the time difference
        end_time = datetime.now()

        print(
            f'\n{colored("Boosting Regression Completed in: ", "green")}'
            f'{end_time - start_time}'
        )

        self.evaluate(
            model=boosting,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            file_name='svm_boosting.png'
        )
