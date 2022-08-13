import os
import pandas as pd
import sklearn.preprocessing
from termcolor import colored
from typing import Tuple, Any, Union, Optional
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self) -> None:
        """Initialize the NeuralNetwork Class."""

        self.__mode = 'dealing with'

    @property
    def mode(self):
        """Return NaN Value Removal Strategy."""

        return self.__mode

    @staticmethod
    def load_file(file: str) -> pd.DataFrame:
        """Load file to DataFrame.

        Args:
            file (str) : Path to file

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
            data (pandas.DataFrame) : Data to work with
            title (str) : Title to display

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
            data (pandas.DataFrame) : Data to work with
            title (str) : Title to display

        """

        print(colored(text=f'{title}\n', color='green') +
              f'\n{data.describe().transpose()}\n')

    @staticmethod
    def display_detailed_nan_values_stats(
            data: pd.DataFrame,
            set_name: str,
            timing: str,
            mode: str
    ) -> None:
        """Display information regarding missing ( NaN ) values in DataFrame.

        Args:
            data (pandas.DataFrame) : Data to work with
            set_name (str) : Name of the set for descriptive title
            timing (str) : Time representation of pre or post dropping/filling
            mode (str) : Mode of removing NaN values ( 'dropping' / 'filling' )

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
    def display_sorted_correlation_for_label(
            data: pd.DataFrame,
            column: str
    ) -> None:
        """Display sorted correlation values for given column.

        Args:
            data (pandas.DataFrame) : Data to work with
            column (str) : Column to display

        """

        print(data.corr()[column].sort_values())

    def __choose_nan_value_removal_strategy(self) -> None:
        """Ask the User which strategy to use when dealing with NaN values."""

        # By default we will fill the values as displayed in the prompt
        self.__mode = 'filling'

        user_input = input(
            colored(text='Please select strategy (D)rop or '
                         '(F)ill Nan Values: (D/[F]) ',
                    color='green'))

        # Change the value to dropping
        if len(user_input) != 0 and user_input.lower()[0] == 'd':
            self.__mode = 'dropping'

        print()

    def remove_nan_values(
            self,
            train_data: Optional[pd.DataFrame] = None,
            test_data: Optional[pd.DataFrame] = None
    ) -> None:
        """Remove NaN values based on DataLoader.__mode strategy.

        Args:
            train_data (Optional[pandas.DataFrame]) : Train data set to remove
                NaN values from
            test_data (Optional[pandas.DataFrame]) : Test data set to remove
                NaN values from

        """

        # Ask the User which strategy to use
        self.__choose_nan_value_removal_strategy()

        # Fill NaN Values
        if self.__mode == 'filling':
            if train_data is not None:
                train_data.fillna(value=train_data.mean(), inplace=True)

        # Drop NaN Values
        else:
            if train_data is not None:
                train_data.dropna(axis=0, inplace=True)

        # We shouldn't drop form the testing Data Set, so we are filling them
        if test_data is not None:
            test_data.fillna(value=train_data.mean(), inplace=True)

    @staticmethod
    def remove_duplicate_values(data: pd.DataFrame) -> None:
        """Drop duplicates from DataFrames.

        Args:
            data (pandas.DataFrame) : Data we clean from duplicates

        """

        data.drop_duplicates(inplace=True)

    @staticmethod
    def get_correlation(data: pd.DataFrame, column: str) -> pd.Series:
        """Return a sorted Correlation Series for a given column.

        Args:
            data (pandas.DataFrame) : The Data to work with
            column (str) : Column name

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
            data (pandas.DataFrame) : Data to work with
            label (str) : Label to separate form features

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
            X_train (Any) : Training Feature Set
            y_train (Any) : Training Label Set
            validation_percentage (Union[Optional, float]) : Represents the
                Validation Data Percentage in the split
            validation_random_state (Optional[int]) : Random State for
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
            label (str) : Label we want to predict
            data (pandas.DataFrame) : Data or Training Data if Test Data is
                present
            test_data (pandas.DataFrame) : Test Data if it is loaded to project
            validation (bool) : Represents the need for Validation Data
            test_percentage (Optional[float]) : Represents the Test Data
                Percentage in the split
            test_random_state (Optional[int]) : Random State for Test Set Split
            validation_percentage (Optional[float]) : Represents the Validation
                Data Percentage in the split
            validation_random_state (Optional[int]) : Random State for
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
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            X_validation: Optional[pd.DataFrame] = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
               Tuple[pd.DataFrame, pd.DataFrame]]:
        """Scales the Data in the passed DataFrames with the passed Scaler.

        Args:
            scaler (sklearn.preprocessing._data) : Scaler to scale the data with
            X_train (pandas.DataFrame) : Training Set to scale
            X_test (pandas.DataFrame) : Testing Set to scale
            X_validation (Optional[pd.DataFrame]) : Validation Set to scale

        Returns:
            Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
               Tuple[pd.DataFrame, pd.DataFrame]] : Scaled Data based on passed
                arguments.

        """

        X_train = scaler.fit_transform(X_train.values)
        X_test = scaler.transform(X_test.values)

        if X_validation is not None:
            X_validation = scaler.transform(X_validation.values)

            return X_train, X_validation, X_test

        return X_train, X_test
