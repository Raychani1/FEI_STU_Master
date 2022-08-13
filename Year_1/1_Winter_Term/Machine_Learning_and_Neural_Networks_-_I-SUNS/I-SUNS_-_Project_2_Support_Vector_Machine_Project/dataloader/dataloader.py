import os
import pandas as pd
import sklearn.preprocessing
from termcolor import colored
from sklearn.ensemble import IsolationForest
from typing import Tuple, Any, Union, Optional, List
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self) -> None:
        """Initialize the NeuralNetwork Class."""

        self.__mode = 'dealing with'

    @property
    def mode(self) -> str:
        """Return NaN Value Removal Strategy.

        Returns:
            str : NaN Value Removal Mode

        """

        return self.__mode

    @staticmethod
    def load_file(file: str) -> pd.DataFrame:
        """Load file to DataFrame.

        Args:
            file (str): Path to file

        Returns:
            pandas.DataFrame : Data loaded to DataFrame

        """

        # Create empty DataFrame
        data = pd.DataFrame()

        # Check if we have passed in an existing file and not a directory
        if os.path.isfile(file) and file.endswith('csv'):
            # Read CSV File to pandas.DataFrame
            data = pd.read_csv(file)

        return data

    @staticmethod
    def display_info_about_dataframe(data: pd.DataFrame, title: str) -> None:
        """Display info about the passed DataFrame.

        Args:
            data (pandas.DataFrame): Data to work with
            title (str): Title to display

        """

        print(colored(text=f'{title}\n', color='green'))
        print(f'{data.info(verbose=True)}\n')

    @staticmethod
    def display_description_about_dataframe(
            data: pd.DataFrame,
            title: str
    ) -> None:
        """Display description of the passed DataFrame.

        Args:
            data (pandas.DataFrame): Data to work with
            title (str): Title to display

        """

        print(colored(text=f'{title}\n', color='green') +
              f'\n{data.describe(include="all").transpose()}\n')

    @staticmethod
    def display_detailed_nan_values_stats(
            data: pd.DataFrame,
            set_name: str,
            timing: str,
            mode: str
    ) -> None:
        """Display information regarding missing ( NaN ) values in DataFrame.

        Args:
            data (pandas.DataFrame): Data to work with
            set_name (str): Name of the set for descriptive title
            timing (str): Time representation of pre or post dropping/filling
            mode (str): Mode of removing NaN values ( 'dropping' / 'filling' )

        """

        # Display information for user
        print(colored(text=f'Number of NaN Values in the {set_name} {timing} '
                           f'{mode} NaN Values\n',
                      color='green'))

        # Iterate through every column
        for column in data.columns:
            # Display NaN statistic information
            print(f'Column {column} has {data[column].isna().sum()} '
                  f'({round(data[column].isna().sum() * 100 / len(data), 3)}'
                  f' %) NaN value(s) ')

        print()

    @staticmethod
    def remove_duplicate_values(
            data: pd.DataFrame,
            columns: List[str]
    ) -> pd.DataFrame:
        """Remove duplicate values based on columns.

        Args:
            data (pandas.DataFrame): Data to work with
            columns (str): Columns to check duplicity for

        Returns:
            pandas.DataFrame : Original DataFrame ignoring the duplicate values

        """

        return data.drop_duplicates(columns, keep='first')

    @staticmethod
    def remove_outliers(
            features: pd.Series,
            labels: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Remove outliers from data using the Isolation Forest Method.

        Args:
            features (pandas.Series): Feature Set
            labels (pandas.Series): Label Set

        Returns:
            Tuple[pd.Series, pd.Series]: Features and Labels without Outlier
                Values

        """

        # SOURCE:
        # https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
        # https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623

        iso = IsolationForest(contamination=0.1)

        yhat = iso.fit_predict(features)

        # select all rows that are not outliers
        mask = yhat != -1

        return features[mask, :], labels[mask]

    @staticmethod
    def get_correlation(data: pd.DataFrame, column: str) -> pd.Series:
        """Return a sorted Correlation Series for a given column.

        Args:
            data (pandas.DataFrame): The Data to work with
            column (str): Column name

        Returns:
            pandas.Series : Sorted Correlation Series for column.

        """
        return data.corr()[column].sort_values()

    @staticmethod
    def separate_label_and_features(
            data: pd.DataFrame,
            label: str
    ) -> Tuple[Any, Any]:
        """Separate label from other features.

        Args:
            data (pandas.DataFrame): Data to work with
            label (str): Label to separate form features

        """

        X = data.drop(label, axis=1)
        y = data[label]

        return X, y

    # noinspection PyPep8Naming
    @staticmethod
    def __split_train_and_validation_data(
            X_train: Any,
            y_train: Any,
            validation_percentage: Optional[float] = 0.33,
            validation_random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits Data based on parameters to Training, Validation and Testing
        Sets.

        Args:
            X_train (Any): Training Feature Set
            y_train (Any): Training Label Set
            validation_percentage (Union[Optional, float]): Represents the
                Validation Data Percentage in the split
            validation_random_state (Optional[int]): Random State for
                Validation Set Split

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] : Returns
                X_train, X_validation, y_train, y_validation.

        """

        # Split Training Data To Training and Validation Set
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train, y_train, test_size=validation_percentage,
            random_state=validation_random_state
        )

        return X_train, X_validation, y_train, y_validation

    # noinspection PyPep8Naming
    def split_data(
            self,
            label: str,
            data: pd.DataFrame,
            test_data: pd.DataFrame = None,
            validation: bool = False,
            test_percentage: Optional[float] = 0.33,
            test_random_state: Optional[int] = None,
            validation_percentage: Optional[float] = 0.33,
            validation_random_state: Optional[int] = None
    ) -> Union[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
              pd.Series],
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ]:
        """Splits Data based on parameters to Training, Validation and Testing
        Sets.

        Args:
            label (str): Label we want to predict
            data (pandas.DataFrame): Data or Training Data if Test Data is
                present
            test_data (pandas.DataFrame): Test Data if it is loaded to project
            validation (bool): Represents the need for Validation Data
            test_percentage (Optional[float]): Represents the Test Data
                Percentage in the split
            test_random_state (Optional[int]): Random State for Test Set Split
            validation_percentage (Optional[float]): Represents the Validation
                Data Percentage in the split
            validation_random_state (Optional[int]): Random State for
                Validation Set Split

        Returns:
            Union[
                Tuple[
                    pd.DataFrame, pd.DataFrame, pd.DataFrame,
                    pd.Series, pd.Series, pd.Series
                ],
                Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] : If we
                    need Validation data it returns X_train, X_validation,
                    X_test, y_train, y_validation, y_test otherwise returns
                    only X_train, X_test, y_train, y_test.

        """

        # If there is no Test Data loaded, so that means we should split the
        # data component
        if test_data is None:
            # Separate the label from features
            X, y = self.separate_label_and_features(data=data, label=label)

            # Split the data to Training and Testing Data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_percentage, random_state=test_random_state
            )

            # If we want to have Validation Data as well
            if validation:
                X_train, X_validation, y_train, y_validation = \
                    self.__split_train_and_validation_data(
                        X_train,
                        y_train,
                        validation_percentage=validation_percentage,
                        validation_random_state=validation_random_state
                    )

                return X_train, X_validation, X_test, y_train, y_validation, \
                       y_test

            return X_train, X_test, y_train, y_test

        # If there is a loaded Test Data, that means we only need to split the
        # Training Data if we want to have Validation Data
        else:
            # Separate the label from features in the Training Data
            X_train, y_train = self.separate_label_and_features(
                data=data,
                label=label
            )

            # Separate the label from features in the Testing Data
            X_test, y_test = self.separate_label_and_features(
                data=test_data,
                label=label
            )

            # If we want to have Validation Data
            if validation:
                # We split the Training Data
                X_train, X_validation, y_train, y_validation = \
                    self.__split_train_and_validation_data(
                        X_train,
                        y_train,
                        validation_percentage=validation_percentage,
                        validation_random_state=validation_random_state
                    )

                return X_train, X_validation, X_test, y_train, y_validation, \
                       y_test

            return X_train, X_test, y_train, y_test

    # noinspection PyPep8Naming
    @staticmethod
    def scale_data(
            scaler: sklearn.preprocessing._data,
            X_train: pd.Series,
            X_test: pd.Series,
            X_validation: Optional[pd.Series] = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
               Tuple[pd.DataFrame, pd.DataFrame]]:
        """Scales the Data in the passed DataFrames with the passed Scaler.

        Args:
            scaler (sklearn.preprocessing._data): Scaler to scale the data with
            X_train (pandas.Series): Training Set to scale
            X_test (pandas.Series): Testing Set to scale
            X_validation (Optional[pd.Series]): Validation Set to scale

        Returns:
            Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
               Tuple[pd.DataFrame, pd.DataFrame]] : Scaled Data based on passed
                arguments.

        """

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if X_validation is not None:
            X_validation = scaler.transform(X_validation)

            return X_train, X_validation, X_test

        return X_train, X_test
