import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from termcolor import colored
from dataloader.dataloader import DataLoader
from models.neural_network import NeuralNetwork
from ops.plotter import Plotter
from configs.units import UNITS
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from typing import List, Optional
from configs.config import CONFIG
import os
import colorama

from datetime import datetime


class NeuralNetworkProject:

    def __init__(self, mode: str):
        """Initialize the NeuralNetworkProject Class."""

        # Set Project Mode
        self.__mode = mode.split('-')[-1]

        # Load configurations
        self.__config = CONFIG

        # Create an instance of DataLoader
        self.__dataloader = DataLoader()

        # Create an instance of Plotter
        self.__plotter = Plotter()

    def __detailed_nan_report(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            timing: str
    ) -> None:
        """Display detailed NaN Value report both for Training and Testing Data.

        Args:
            train (pandas.DataFrame) : Train Data Set
            test (pandas.DataFrame) : Test Data Set
            timing (str) : Time representation of NaN Value removal ('before'/
                'after')

        """

        # Display detailed information about the NaN Values in the Training Data
        self.__dataloader.display_detailed_nan_values_stats(
            data=train,
            set_name='Training Data Set',
            timing=timing,
            mode=self.__dataloader.mode
        )

        # Display detailed information about the NaN Values in the Testing Data
        self.__dataloader.display_detailed_nan_values_stats(
            data=test,
            set_name='Testing Data Set',
            timing=timing,
            mode='filling'
        )

    def __information(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            valid: Optional[pd.DataFrame] = None,
            titles: List[str] = None
    ) -> None:
        """Display information about Training, Testing Data and Validation Data
        if passed.

        Args:
            train (pandas.DataFrame) : Train Data Set
            test (pandas.DataFrame) : Test Data Set
            valid (Optional[pandas.DataFrame]) : Validation Data Set
            titles (List[str]) : List of titles to display along with the
                information about the DataFrames

        """

        # Display basic information about the Training Data
        self.__dataloader.display_info_about_dataframe(
            data=train,
            title=titles[0]
        )

        if valid is not None:
            # Display basic information about the Validation Data
            self.__dataloader.display_info_about_dataframe(
                data=valid,
                title=titles[1]
            )

        # Display basic information about the Testing Data
        self.__dataloader.display_info_about_dataframe(
            data=test,
            title=titles[-1]
        )

    def __info_and_description(self, data: pd.DataFrame, title: str) -> None:
        """Display information and description of a given DataFrame.

        Args:
            data (pandas.DataFrame) : Data to work with
            title (str) : Additional information to display

        """

        # Display basic information about the Training Data
        self.__dataloader.display_info_about_dataframe(
            data=data,
            title=title
        )

        # Display the description of the Training Data
        self.__dataloader.display_description_about_dataframe(
            data=data,
            title=title
        )

    def __info_and_description_for_every_dataframe(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            valid: pd.DataFrame,
            titles: List[str]
    ) -> None:
        """Display information and description for every DataFrame.

        Args:
            train (pandas.DataFrame) : Train Data Set
            test (pandas.DataFrame) : Test Data Set
            valid (pandas.DataFrame) : Validation Data Set
            titles (List[str]) : List of titles to display along with the
                information and description about the DataFrames

        """

        self.__info_and_description(data=train, title=titles[0])
        self.__info_and_description(data=valid, title=titles[1])
        self.__info_and_description(data=test, title=titles[2])

    @staticmethod
    def __calculate_random_classifier_accuracy(
            data: pd.DataFrame,
            column: str,
            title: str
    ) -> None:
        """Calculate Accuracy for Random Classifier.

        Args:
            data (pandas.DataFrame) : Data to work with
            column (str) : Column to process
            title (str) : Title for information

        """

        # Get the number of rows in the DataFrame
        total_rows = len(data)

        # Get the Unique Values for the given column
        unique_values = data[column].value_counts()

        # Create empty list where the components of the equation will be stored
        unique_values_percentage = list()

        # For each unique value
        for value in unique_values:
            # Calculate the percentage of the given values and square it
            unique_values_percentage.append(round(value / total_rows, 4) ** 2)

        # Display the Final Accuracy with a custom message
        print(f"{colored(f'{title}:', 'green')} "
              f"{sum(unique_values_percentage)}")

    # noinspection PyPep8Naming
    @staticmethod
    def __run_logistic_regression(
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            randoms_tate: int,
            max_iter: int
    ) -> None:
        """Run Logistic Regression Process ( Create, Train, Predict, Evaluate ).

        Args:
            X_train (pandas.DataFrame) : Training Feature Set
            y_train (pandas.Series) : Training Label Set
            X_test (pandas.DataFrame) : Testing Feature Set
            y_test (pandas.Series) : Testing Label Set
            randoms_tate (int) : Random State for Model
            max_iter (int) : Max Number of Iteration for Model

        """

        # Inform the User
        print(colored('Logistic Regression\n', 'green'))

        # Create Logistic Regression Model
        logistic_regression = LogisticRegression(
            random_state=randoms_tate,
            max_iter=max_iter
        )

        # Train the Logistic Regression Model
        logistic_regression.fit(X=X_train, y=y_train)

        # Predict using the Logistic Regression Model
        logistic_regression_prediction = logistic_regression.predict(X=X_test)

        # Display Classification Report
        print(colored('Classification Report:', 'green'))
        print(
            classification_report(
                y_true=y_test,
                y_pred=logistic_regression_prediction
            )
        )

    # noinspection PyPep8Naming
    def run(self):
        """Runs the whole Neural Network Assignment."""

        # Remove plt.figure warning
        plt.rcParams.update({'figure.max_open_warning': 0})

        colorama.init()

        # Read Data
        training_data = self.__dataloader.load_file(
            self.__config['data']['training_data']
        )

        testing_data = self.__dataloader.load_file(
            self.__config['data']['testing_data']
        )

        # Display information about Training and Testing Data
        self.__information(train=training_data, test=testing_data, titles=[
            'Information about the Training Data Set',
            'Information about the Testing Data Set'
        ])

        # Display detailed NaN Value Report
        self.__detailed_nan_report(
            train=training_data,
            test=testing_data,
            timing='before'
        )

        # Remove NaN values from the Data Sets
        self.__dataloader.remove_nan_values(train_data=training_data,
                                            test_data=testing_data)

        # Display information about Training and Testing Data
        self.__information(train=training_data, test=testing_data, titles=[
            f'Information about the Training Data Set after '
            f'{self.__dataloader.mode} NaN Values',
            'Information about the Testing Data Set after filling NaN Values'
        ])

        # Display detailed NaN Value Report
        self.__detailed_nan_report(
            train=training_data,
            test=testing_data,
            timing='after'
        )

        # Drop duplicate values
        self.__dataloader.remove_duplicate_values(training_data)
        self.__dataloader.remove_duplicate_values(testing_data)

        correlation = training_data.corr()['quality']

        # Drop Negative Correlated Columns
        if self.__mode == 'negative':
            print(colored('Dropping Negative Correlated Columns\n', 'green'))

            training_data.drop(
                list(correlation[correlation < 0].index),
                axis=1,
                inplace=True
            )
            testing_data.drop(
                list(correlation[correlation < 0].index),
                axis=1,
                inplace=True
            )

        # Drop Positive Correlated Columns
        if self.__mode == 'positive':
            print(colored('Dropping Positive Correlated Columns\n', 'green'))

            training_data.drop(
                list(correlation[(correlation > 0) & (correlation < 1)].index),
                axis=1,
                inplace=True
            )
            testing_data.drop(
                list(correlation[(correlation > 0) & (correlation < 1)].index),
                axis=1,
                inplace=True
            )

        # Display information about Training and Testing Data
        self.__information(train=training_data, test=testing_data, titles=[
            f'Information about the Training Data Set after '
            f'{self.__dataloader.mode} NaN Values and dropped duplicates',
            f'Information about the Testing Data Set after filling NaN Values '
            f'and dropped duplicates'
        ])

        # Display the distribution of Column value for the Test Set
        self.__plotter.display_unique_column_values(
            data=testing_data,
            column='quality',
            kind='pie',
            title=f"Test Data - {'quality'.capitalize()} Values"
        )

        # Calculate the Accuracy of the Random Classifier on the Test Set
        self.__calculate_random_classifier_accuracy(
            data=testing_data,
            column='quality',
            title='Random Classifier Accuracy on the Testing Data Set'
        )

        # Display Histograms for the Training Data
        self.__plotter.display_histograms(
            data=training_data,
            skip_columns=['type', 'quality'],
            units=UNITS,
            title='Unscaled Training Data Set',
            bins=50,
            hue='type'
        )

        # Display Histograms for the Testing Data
        self.__plotter.display_histograms(
            data=testing_data,
            skip_columns=['type', 'quality'],
            units=UNITS,
            title='Unscaled Testing Data Set',
            bins=50,
            hue='type'
        )

        # Display Correlation for the Quality Column
        print(colored('\nQuality Column Correlation:\n', 'green'))
        print(f"{self.__dataloader.get_correlation(training_data, 'quality')}\n")

        # Display Heatmap of the Training Data
        self.__plotter.display_heatmap(training_data)

        # Split the data ( we won't create a new test set, we just need the
        # validation set - this is just an all in one method)
        X_train, X_validation, X_test, y_train, y_validation, y_test = \
            self.__dataloader.split_data(
                label='quality',
                data=training_data,
                test_data=testing_data,
                validation=True,
                test_random_state=101,
                validation_percentage=0.25,
                validation_random_state=101
            )

        # Display the distribution of Type for the Training Set
        self.__plotter.display_unique_column_values(
            data=X_train,
            column='type',
            kind='pie',
            title='Training Data - Type'
        )

        # Display the distribution of Type for the Validation Set
        self.__plotter.display_unique_column_values(
            data=X_validation,
            column='type',
            kind='pie',
            title='Validation Data - Type'
        )

        # Display the distribution of Type for the Test Set
        self.__plotter.display_unique_column_values(
            data=X_test,
            column='type',
            kind='pie',
            title='Testing Data - Type'
        )

        # Display basic information and description of each DataFrame before
        # scaling
        self.__info_and_description_for_every_dataframe(
            train=X_train,
            valid=X_validation,
            test=X_test,
            titles=[
                'Information about the Training Data Set before Scaling',
                'Information about the Validation Data Set before Scaling',
                'Information about the Testing Data Set before Scaling'
            ]
        )

        # Create an instance of the selected scaler
        scaler = StandardScaler()

        # Scale our Data
        X_train, X_validation, X_test = self.__dataloader.scale_data(
            scaler=scaler,
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test
        )

        # Display Histograms for the Training Data
        self.__plotter.display_histograms(
            data=pd.DataFrame(data=X_train, columns=training_data.columns[:-1]),
            skip_columns=['type'],
            units=UNITS,
            title='Scaled Training Data Set',
            bins=50,
            hue='type'
        )

        # Display Histograms for the Training Data
        self.__plotter.display_histograms(
            data=pd.DataFrame(
                data=X_validation,
                columns=training_data.columns[:-1]
            ),
            skip_columns=['type'],
            units=UNITS,
            title='Scaled Validation Data Set',
            bins=50,
            hue='type'
        )

        # Display Histograms for the Training Data
        self.__plotter.display_histograms(
            data=pd.DataFrame(data=X_test, columns=training_data.columns[:-1]),
            skip_columns=['type'],
            units=UNITS,
            title='Scaled Testing Data Set',
            bins=50,
            hue='type'
        )

        # Run Logistic Regression Model Prediction Process
        self.__run_logistic_regression(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            randoms_tate=101,
            max_iter=1000
        )

        # Display basic information and description of each DataFrame after
        # scaling
        self.__info_and_description_for_every_dataframe(
            train=pd.DataFrame(X_train, columns=training_data.columns[:-1]),
            valid=pd.DataFrame(X_validation,
                               columns=training_data.columns[:-1]),
            test=pd.DataFrame(X_test, columns=training_data.columns[:-1]),
            titles=[
                'Information about the Training Data Set after Scaling',
                'Information about the Validation Data Set after Scaling',
                'Information about the Testing Data Set after Scaling'
            ]
        )

        if self.__mode == 'grid':
            # Create an instance of a Neural Network
            neural_network = NeuralNetwork()
        
            # Start the timer
            start_time = datetime.now()
        
            neural_network.grid_search(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_validation,
                y_valid=y_validation,
                X_test=X_test,
                y_test=y_test,
                loss_function='binary_crossentropy',
                early_stopping=True,
                monitor='val_loss',
                file=os.path.join(
                    os.path.abspath('.'),
                    'output/grid_search_output2.txt'
                ),
                training_history_file=os.path.join(
                    os.path.abspath('.'),
                    'output/history.json'
                )
            )
        
            # Stop the timer and display the time difference
            end_time = datetime.now()
        
            print(
                f'{colored("Grid Search Completed in: ", "green")}'
                f'{end_time - start_time}'
            )
        
        else:
            neural_network = NeuralNetwork()
        
            if self.__mode != 'positive' or self.__mode != 'negative':
                neural_network = NeuralNetwork(network_type=self.__mode)
        
            # Build the Neural Network
            neural_network.build(
                X_train=X_train,
                loss_function='binary_crossentropy',
            )
        
            # Train the Neural Network
            training_history = neural_network.train(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_validation,
                y_valid=y_validation,
                early_stopping=True
                if self.__mode == 'best'
                or self.__mode == 'positive'
                or self.__mode == 'negative'
                else False,
                monitor='val_loss'
            )
        
            self.__plotter.display_training_history(
                history=pd.DataFrame(training_history)
            )
        
            # Evaluate the Model and generate Confusion Matrix
            confusion_matrix = neural_network.evaluate(
                X_test=X_test,
                y_test=y_test,
                file=os.path.join(
                    os.path.abspath('.'),
                    'output/neural_network_evaluation.txt'
                )
            )
        
            # Plot the Confusion Matrix
            self.__plotter.display_confusion_matrix(
                confusion_matrix=confusion_matrix,
                labels=list(range(training_data['quality'].nunique()))
            )
        
        # Make sure the plots are displayed
        plt.show()
