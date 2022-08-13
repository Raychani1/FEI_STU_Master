import os
import ast
import json
import sklearn
import colorama
import pandas as pd
from sklearn import tree
from graphviz import Source
from termcolor import colored
import matplotlib.pyplot as plt
from ops.plotter import Plotter
from configs.config import CONFIG
from ops.evaluator import Evaluator
from dataloader.dataloader import DataLoader
from models.neural_network import NeuralNetwork
from typing import List, Optional, Set, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from models.support_vector_machine import SupportVectorMachine


class SupportVectorMachineProject:

    def __init__(self) -> None:
        """Initialize SupportVectorMachineProject Class."""

        # Load configurations
        self.__config = CONFIG

        # Create an instance of DataLoader
        self.__dataloader = DataLoader()

        # Create an instance of Plotter
        self.__plotter = Plotter()

        # Create an instance of Evaluator
        self.__evaluator = Evaluator()

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
            titles: List[str],
            valid: Optional[pd.DataFrame] = None
    ) -> None:
        """Display information and description for every DataFrame.
        Args:
            train (pandas.DataFrame) : Train Data Set
            test (pandas.DataFrame) : Test Data Set
            valid (Optional[pandas.DataFrame]) : Validation Data Set
            titles (List[str]) : List of titles to display along with the
                information and description about the DataFrames

        """

        self.__info_and_description(data=train, title=titles[0])

        if valid is not None:
            self.__info_and_description(data=valid, title=titles[1])

        self.__info_and_description(data=test, title=titles[-1])

    def __run_eda(
            self,
            training_data: pd.DataFrame,
            testing_data: pd.DataFrame
    ) -> None:
        """Run Exploratory Data Analysis.

        Args:
            training_data (pandas.DataFrame): Training Data Set
            testing_data (pandas.DataFrame): Testing Data Set

        """

        # Display Pair Plot for Training Data
        self.__plotter.display_pair_plot(
            data=training_data,
            hue='explicit',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'pair_plots'
            )
        )

        # Display Word Cloud for Training Data
        self.__plotter.display_word_cloud(
            data=training_data,
            column='name',
            title='train',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'word_clouds'
            )
        )

        # Display Word Cloud for Testing Data
        self.__plotter.display_word_cloud(
            data=testing_data,
            column='name',
            title='test',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'word_clouds'
            )
        )

        # Display Top 10 Most Popular Tracks for Training Data
        self.__plotter.display_top_10(
            data=training_data,
            base_columns=[
                'popularity',
                'artist_followers'
            ],
            columns_to_display=(
                'name',
                'popularity'
            ),
            title='train',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'bar_plots'
            )
        )

        # Display Top 10 Most Popular Tracks for Testing Data
        self.__plotter.display_top_10(
            data=testing_data,
            base_columns=[
                'popularity',
                'artist_followers'
            ],
            columns_to_display=(
                'name',
                'popularity'
            ),
            title='test',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'bar_plots'
            )
        )

        # Display Choropleth Map Plot for Training Data
        self.__plotter.display_choropleth(data=training_data)

        # Display Choropleth Map Plot for Testing Data
        self.__plotter.display_choropleth(data=testing_data)

    @staticmethod
    def __display_random_forest_feature_importance(
            data: pd.DataFrame,
            forest_regressor: sklearn.ensemble.RandomForestRegressor,
            path: str,
            save: bool = True
    ) -> None:
        """Display Random Forest Classifier Feature Importance.

        Args:
            data (pandas.Dataframe): Data to work with
            forest_regressor (sklearn.ensemble.RandomForestRegressor):
                Random Forest Regressor to check
            path (str): Save File Folder Path
            save (bool): Save File or not

        """

        # SOURCE:
        # https://stackoverflow.com/a/41900730
        # https://stackoverflow.com/questions/41900387/mapping-column-names-to-random-forest-feature-importances

        # Inform the User
        print(
            colored(
                '\nRandom Forest Regressor Feature Importance:',
                'green'
            )
        )

        # Export the importance for Forest Regressor
        feature_importance = forest_regressor.feature_importances_

        # Save Importance Data to dictionary
        feats = dict()  # a dict to hold feature_name: feature_importance
        for feature, importance in zip(data.columns, feature_importance):
            feats[feature] = importance  # add the name/value pair

        # Sort the dictionary to a descending order
        sorted_feats = dict(
            sorted(
                feats.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

        # Display the Descending Feature Importance Data
        for key, value in sorted_feats.items():
            print(f"{colored(f'Feature {key}:', 'green')} {value}")

        # Save the data to a DataFrame
        importances = pd.DataFrame.from_dict(feats, orient='index').rename(
            columns={0: 'Gini-importance'})

        # Create figure
        fig = plt.figure(figsize=(16, 10))

        # Add simple axes to plot on
        ax = fig.add_subplot(1, 1, 1)

        # Plot importance
        importances.sort_values(by='Gini-importance').plot(
            kind='bar',
            rot=45,
            ax=ax
        )

        # If we want to save the plot
        if save and (path is not None):
            # Draw the plot, so when we save it we make sure it is not blank
            plt.draw()

            # Save figure based on parameters
            plt.savefig(f"{path}/random_forest_feature_importance.png")

        else:
            plt.show(block=False)

    @staticmethod
    def __export_decision_tree(
            decision_tree: Any,
            path: str,
            feature_names: List[str],
    ) -> None:
        """Export Decision Tree to SVG File.

        Args:
            decision_tree (Any): Decision Tree to export
            path (str): Save File Folder Path
            feature_names (List[str]): Name of Features

        """

        # SOURCE:
        # https://stackoverflow.com/a/45533426

        # Inform the User
        print(
            colored(
                f'\nExporting Decision Tree to SVG File:\n',
                'green'
            ),
            f'{path}/decision_tree.svg'
        )

        # Export the Decision Tree Graph
        graph = Source(
            tree.export_graphviz(
                decision_tree,
                out_file=None,
                feature_names=feature_names
            )
        )

        # Specify the Output format
        svg_bytes = graph.pipe(format='svg')

        # Save to file
        with open(f'{path}/decision_tree.svg', 'wb') as f:
            f.write(svg_bytes)

    def __random_forest_process(
            self,
            X_train: pd.Series,
            y_train: pd.Series,
            X_test: pd.Series,
            y_test: pd.Series,
            columns: List[str],
            path: str
    ) -> None:
        """Run Random Forest Classifier Process on the Explicit column.

        Args:
            X_train (pandas.Series): Training Feature Set
            y_train (pandas.Series): Training Label Set
            X_test (pandas.Series): Testing Feature Set
            y_test (pandas.Series): Testing Label Set
            columns (List[str]): Column Names
            path (str): Save File Path

        """

        # Create the Random Forrest Regressor with 100 Estimators
        forest_regressor = RandomForestRegressor(n_estimators=100)

        # Train the Forest
        forest_regressor.fit(X=X_train, y=y_train)

        # Run Evaluation on the Forest
        self.__evaluator.evaluate(
            mode='Random Forest Regressor',
            model=forest_regressor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            file_name='random_forest_regressor.png'
        )

        # Select one Decision Tree
        decision_tree = forest_regressor.estimators_[69]

        # Export that Decision Tree
        self.__export_decision_tree(
            decision_tree=decision_tree,
            feature_names=columns,
            path=path
        )

        # Check the importance of the Forest input features
        self.__display_random_forest_feature_importance(
            forest_regressor=forest_regressor,
            data=pd.DataFrame(X_train, columns=columns),
            save=True,
            path=path
        )

    def __neural_network_process(
            self,
            training_data: pd.DataFrame,
            testing_data: pd.DataFrame
    ) -> None:
        """Execute Neural Network Process.

        Args:
           training_data (pandas.DataFrame): Training Data Set
           testing_data (pandas.DataFrame): Testing Data Set

        """

        # Split the Data to Train, Validation and Test so we can use it in
        # EarlyStopping
        X_train, X_valid, X_test, y_train, y_valid, y_test = \
            self.__dataloader.split_data(
                label='loudness',
                data=training_data,
                test_data=testing_data,
                validation=True
            )

        X_train_no_outliers, y_train_final = \
            self.__dataloader.remove_outliers(
                features=X_train.values,
                labels=y_train.values
            )

        X_valid_no_outliers, y_valid_final = \
            self.__dataloader.remove_outliers(
                features=X_valid.values,
                labels=y_valid.values
            )

        X_test_no_outliers, y_test_final = \
            self.__dataloader.remove_outliers(
                features=X_test.values,
                labels=y_test.values
            )

        # Create a scaler
        scaler = StandardScaler()

        # Scale the Features
        X_train_final, X_valid_final, X_test_final = \
            self.__dataloader.scale_data(
                scaler=scaler,
                X_train=X_train_no_outliers,
                X_validation=X_valid_no_outliers,
                X_test=X_test_no_outliers
            )

        # Create an instance of the Network
        neural_network = NeuralNetwork()

        # Build the Network
        neural_network.build(
            X_train=X_train,
            loss_function='mse'
        )

        # Train the Network
        history = neural_network.train(
            X_train=X_train_final,
            y_train=y_train_final,
            X_valid=X_valid_final,
            y_valid=y_valid_final,
            monitor='val_loss',
            early_stopping=True
        )

        # Display the Training History on plot ( only the loss, because this is
        # a regression problem, so we can not measure if we are classifying
        # entities to the right class - that would be a classification problem )
        self.__plotter.display_training_history(history)

        # Evaluate the Network
        neural_network.evaluate(
            X_test=X_test_final,
            y_test=y_test_final
        )

    @staticmethod
    def __get_artist_genres(data: pd.DataFrame) -> Set[str]:
        """Get all the unique artist genres from DataFrame.

        Args:
            data (pandas.DataFrame): Data to work with

        Returns:
            Set[str]: Unique Artist Genres

        """

        genres = list()

        for index, row in data.iterrows():
            genres.extend(ast.literal_eval(row['artist_genres']))

        return set(genres)

    @staticmethod
    def __extract_genre(
            genres: Set[str],
            keyword: str
    ) -> Tuple[Set[str], List[str]]:
        """Collect subgenres to a more wide genre.

        Args:
            genres (Set[str]): Unique Artist Genres
            keyword (str): Keyword to search for

        Returns:
            Tuple[Set[str], List[str]]: Rest of the original genres and the
                removed wide genre

        """

        # Collect Subgenres to a more wide genre
        genre = [s for s in genres if keyword in s]

        # Display information about the size of wide genre
        # print(f'{keyword.capitalize()}: {len(genre)}')

        # Remove the Wide genre from the set of unique Artist Genres
        genres = [x for x in genres if x not in genre]

        return set(genres), genre

    def __generate_new_genres(
            self,
            data: pd.DataFrame,
            file_specifier: str
    ) -> None:
        """Process Unique Artist Genres to JSON Files for further processing.

        Args:
            data (pandas.DataFrame): Data to work with
            file_specifier (str): Data Type Specifier [train | test]

        """

        new_genres = dict()

        # Get all the Unique Artist Genres
        genres = self.__get_artist_genres(data)
        # print(f'All Genres: {len(genres)}')

        # Extract genres to Wide genre
        genres, rock = self.__extract_genre(
            genres=genres,
            keyword='rock'
        )

        genres, new_genres['metal'] = self.__extract_genre(
            genres=genres,
            keyword='metal'
        )

        genres, new_genres['rap'] = self.__extract_genre(
            genres=genres,
            keyword='rap'
        )

        genres, new_genres['hop'] = self.__extract_genre(
            genres=genres,
            keyword='hop'
        )

        genres, new_genres['pop'] = self.__extract_genre(
            genres=genres,
            keyword='pop'
        )

        genres, punk = self.__extract_genre(
            genres=genres,
            keyword='punk'
        )

        rock.extend(punk)
        new_genres['rock'] = rock

        genres, new_genres['emo'] = self.__extract_genre(
            genres=genres,
            keyword='emo'
        )

        genres, soundtrack = self.__extract_genre(
            genres=genres,
            keyword='soundtrack'
        )

        genres, ost = self.__extract_genre(
            genres=genres,
            keyword=' ost'
        )

        soundtrack.extend(ost)
        new_genres['soundtrack'] = soundtrack

        genres, new_genres['classical'] = self.__extract_genre(
            genres=genres,

            keyword='classical'
        )

        genres, new_genres['indie'] = self.__extract_genre(
            genres=genres,
            keyword='indie'
        )

        genres, new_genres['folk'] = self.__extract_genre(
            genres=genres,
            keyword='folk'
        )

        genres, new_genres['blues'] = self.__extract_genre(
            genres=genres,
            keyword='blues'
        )

        genres, new_genres['jazz'] = self.__extract_genre(
            genres=genres,
            keyword='jazz'
        )

        genres, new_genres['country'] = self.__extract_genre(
            genres=genres,
            keyword='country'
        )

        genres, new_genres['electronic'] = self.__extract_genre(
            genres=genres,
            keyword='electronic'
        )

        genres, new_genres['r&b'] = self.__extract_genre(
            genres=genres,
            keyword='r&b'
        )

        genres, new_genres['soul'] = self.__extract_genre(
            genres=genres,
            keyword='soul'
        )

        genres, new_genres['alternative'] = self.__extract_genre(
            genres=genres,
            keyword='alternative'
        )

        genres, new_genres['techno'] = self.__extract_genre(
            genres=genres,
            keyword='techno'
        )

        genres, new_genres['house'] = self.__extract_genre(
            genres=genres,
            keyword='house'
        )

        genres, new_genres['reggae'] = self.__extract_genre(
            genres=genres,
            keyword='reggae'
        )

        genres, new_genres['edm'] = self.__extract_genre(
            genres=genres,
            keyword='edm'
        )

        genres, new_genres['drill'] = self.__extract_genre(
            genres=genres,
            keyword='drill'
        )

        genres, new_genres['americana'] = self.__extract_genre(
            genres=genres,
            keyword='americana'
        )

        genres, new_genres['lo-fi'] = self.__extract_genre(
            genres=genres,
            keyword='lo-fi'
        )

        genres, new_genres['hardcore'] = self.__extract_genre(
            genres=genres,
            keyword='hardcore'
        )

        genres, new_genres['dnb'] = self.__extract_genre(
            genres=genres,
            keyword='dnb'
        )

        genres, new_genres['trance'] = self.__extract_genre(
            genres=genres,
            keyword='trance'
        )

        genres, new_genres['worship'] = self.__extract_genre(
            genres=genres,
            keyword='worship'
        )

        genres, new_genres['singer_songwriter'] = self.__extract_genre(
            genres=genres,
            keyword='singer-songwriter'
        )

        genres, new_genres['psych'] = self.__extract_genre(
            genres=genres,
            keyword='psych'
        )

        genres, new_genres['ambient'] = self.__extract_genre(
            genres=genres,
            keyword='ambient'
        )

        genres, new_genres['electric'] = self.__extract_genre(
            genres=genres,
            keyword='electr'
        )

        genres, new_genres['wave'] = self.__extract_genre(
            genres=genres,
            keyword='wave'
        )

        genres, new_genres['funk'] = self.__extract_genre(
            genres=genres,
            keyword='funk'
        )

        genres, new_genres['bass'] = self.__extract_genre(
            genres=genres,
            keyword='bass'
        )

        genres, new_genres['disco'] = self.__extract_genre(
            genres=genres,
            keyword='disco'
        )

        genres, new_genres['band'] = self.__extract_genre(
            genres=genres,
            keyword='band'
        )

        genres, new_genres['musical'] = self.__extract_genre(
            genres=genres,
            keyword='musical'
        )

        genres, new_genres['musica'] = self.__extract_genre(
            genres=genres,
            keyword='musica'
        )

        genres, new_genres['deathcore'] = self.__extract_genre(
            genres=genres,
            keyword='deathcore'
        )

        genres, new_genres['choir'] = self.__extract_genre(
            genres=genres,
            keyword='choir'
        )

        genres, new_genres['thematic_music'] = self.__extract_genre(
            genres=genres,
            keyword='music'
        )

        genres, new_genres['core'] = self.__extract_genre(
            genres=genres,
            keyword='core'
        )

        genres, new_genres['beat'] = self.__extract_genre(
            genres=genres,
            keyword='beat'
        )

        genres, new_genres['dubstep'] = self.__extract_genre(
            genres=genres,
            keyword='dubstep'
        )

        dance = list()

        dance_styles = [
            "ballroom",
            "cha-cha-cha",
            "dance",
            "organ",
            "salsa",
            "samba",
            "swing",
            "tango",
            "zamba",
        ]

        for dance_style in dance_styles:
            dance.extend([s for s in genres if dance_style in s])

        new_genres['dance'] = dance

        # print(f'Dance: {len(dance)}')

        genres = [x for x in genres if x not in dance]

        instrumental = list()

        instruments = [
            "accordion",
            "guitar",
            "piano",
            "banjo",
            "cello",
            "violin",
            "viola",
            "instrumental",
            "orchestr",
            "bagpipe",
        ]

        for instrument in instruments:
            instrumental.extend([s for s in genres if instrument in s])

        new_genres['instrumental'] = instrumental
        # print(f'Instrumental: {len(instrumental)}')

        genres = [x for x in genres if x not in instrumental]
        # print(f'Other: {len(genres)}')

        new_genres['other'] = genres

        # Save new genres to file
        with open(
                f'output/processing_steps/new_genres_{file_specifier}.json', 'w'
        ) as genres_file:
            json.dump(new_genres, genres_file, indent=2)

    def __set_genre(
            self,
            data: pd.DataFrame,
            file_specifier: str
    ) -> pd.DataFrame:
        """Modify the Artist Genres based on the new genres from file.

        Args:
            data (pandas.DataFrame): Data to work with
            file_specifier (str): Data Type Specifier [train | test]

        Returns:
            pandas.DataFrame: Modified Artist Genres

        """

        # Generate new genres to file
        self.__generate_new_genres(data=data, file_specifier=file_specifier)

        # Open that file
        with open(
                f'output/processing_steps/new_genres_{file_specifier}.json'
        ) as genres_file:

            # Load the new genres
            genres = json.load(genres_file)

            # Iterate through the DataFrame
            for index, row in data.iterrows():

                # Get the old genres as List[str]
                old_genres = ast.literal_eval(row['artist_genres'])

                # Replace the old value with a new one
                for genre_index, genre in enumerate(old_genres):
                    for key, values in genres.items():
                        if genre in values:
                            old_genres[genre_index] = key

                # Write changes to DataFrame
                data['artist_genres'].iloc[index] = max(
                    set(old_genres),
                    key=old_genres.count
                )

        return data

    def __encode_genres(
            self,
            encoder: sklearn.preprocessing.LabelEncoder,
            training_data: pd.DataFrame,
            testing_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode the Artist Genre (str) to a Numerical Value (int).

        Args:
            encoder (sklearn.preprocessing.LabelEncoder): Label Encoder used for
                Genre encoding
            training_data (pandas.DataFrame): Training Data Set
            testing_data (pandas.DataFrame): Testing Data Set

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and Testing Data Set
                after encoding the Genre

        """

        training_data = training_data.copy()

        # Display Different Count of Genres in the Training Data Set
        self.__plotter.display_count_plot(
            data=training_data,
            column='artist_genres',
            data_type='training',
            title='Training Data Set',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'count_plots'
            )
        )

        # Fit and Transform on the Training Data Set
        training_data['artist_genres'] = encoder.fit_transform(
            training_data['artist_genres']
        )

        testing_data = testing_data.copy()

        # Display Different Count of Genres in the Testing Data Set
        self.__plotter.display_count_plot(
            data=testing_data,
            column='artist_genres',
            data_type='testing',
            title='Testing Data Set',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'count_plots'
            )
        )

        # Only Transform on the Training Data Set so the numbers are the same
        testing_data['artist_genres'] = encoder.transform(
            testing_data['artist_genres']
        )

        return training_data, testing_data

    @staticmethod
    def __process_date(data: pd.DataFrame) -> pd.DataFrame:
        """Process DateTime String to separate numerical columns.

        Args:
            data (pandas.DataFrame): Data to work with

        Returns:
            pandas.DataFrame: DataFrame with the processed Release Date

        """

        # Process (DateTime) String to DataTime Object
        data['release_date'] = pd.to_datetime(data['release_date'])

        # Get each attribute from DateTime Object and save them to separate
        # columns
        data['release_date_year'] = data['release_date'].dt.year
        data['release_date_month'] = data['release_date'].dt.month
        data['release_date_day'] = data['release_date'].dt.day

        # Remove the original Release Date column
        data.drop(['release_date'], axis=1, inplace=True)

        return data

    def __run_data_processing(
            self,
            training_data: pd.DataFrame,
            testing_data: pd.DataFrame
    ) -> Tuple[Any, Any, pd.Series, pd.Series]:
        """Run Data Processing Steps.

        Args:
            training_data (pandas.DataFrame): Training Data Set
            testing_data (pandas.DataFrame): Testing Data Set

        Returns:
            Tuple[Any, Any, pd.Series, pd.Series]: Processed Training and
                Testing Data

        """

        # Display information about Training and Testing Data
        self.__information(
            train=training_data,
            test=testing_data,
            titles=[
                'Information about the Training Data Set',
                'Information about the Testing Data Set'
            ]
        )

        # Display detailed NaN Value Report
        self.__detailed_nan_report(
            train=training_data,
            test=testing_data,
            timing='before'
        )

        # Process Date to Numerical Values
        self.__process_date(data=training_data)
        self.__process_date(data=testing_data)

        # Remove Duplicate Values based on Artist, Track Name and Release Year
        training_data_no_duplicates = self.__dataloader.remove_duplicate_values(
            data=training_data,
            columns=[
                'artist',
                'name',
                'release_date_year'
            ]
        )

        testing_data_no_duplicates = self.__dataloader.remove_duplicate_values(
            data=testing_data,
            columns=[
                'artist',
                'name',
                'release_date_year'
            ]
        )

        # Reset Index after dealing with Duplicate Values and Outliers
        training_data_no_duplicates.reset_index(inplace=True, drop=True)
        testing_data_no_duplicates.reset_index(inplace=True, drop=True)

        # Display information about Training and Testing Data
        self.__information(
            train=training_data_no_duplicates,
            test=testing_data_no_duplicates,
            titles=[
                'Information about the Training Data Set after ignoring '
                'Duplicate Values',
                'Information about the Testing Data Set after ignoring '
                'Duplicate Values'
            ]
        )

        # Replace Explicit Column True/False Values with Numerical form
        training_data_no_duplicates['explicit'].replace(
            {
                True: 1,
                False: 0
            },
            inplace=True
        )

        testing_data_no_duplicates['explicit'].replace(
            {
                True: 1,
                False: 0
            },
            inplace=True
        )

        # Set New Genre Categories for DataFrames
        training_data_with_new_genres = self.__set_genre(
            data=training_data_no_duplicates,
            file_specifier='train'
        )

        testing_data_with_new_genres = self.__set_genre(
            data=testing_data_no_duplicates,
            file_specifier='test'
        )

        # Encode New Genre Categories to Numerical Values
        label_encoder = LabelEncoder()

        training_data_encoded_genres, testing_data_encoded_genres = \
            self.__encode_genres(
                encoder=label_encoder,
                training_data=training_data_with_new_genres,
                testing_data=testing_data_with_new_genres
            )

        # Display Correlation Matrix in a Heatmap
        self.__plotter.display_heatmap(
            data=training_data,
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'heatmaps'
            )
        )

        # Display Correlation Matrix Values regarding Loudness
        print(
            colored('Correlation for Loudness\n', 'green'),
            self.__dataloader.get_correlation(training_data, 'loudness')
        )

        # Drop not strongly correlated columns
        train_subset = training_data_encoded_genres.drop(
            [
                'id',
                'artist_id',
                'artist',
                'name',
                'url',
                'playlist_id',
                'playlist_description',
                'playlist_name',
                'playlist_url',
                'query',
                'duration_ms',
                'mode',
                'key',
                'release_date_month',
                'release_date_day',
                'artist_followers'
            ],
            axis=1
        )

        test_subset = testing_data_encoded_genres.drop(
            [
                'id',
                'artist_id',
                'artist',
                'name',
                'url',
                'playlist_id',
                'playlist_description',
                'playlist_name',
                'playlist_url',
                'query',
                'duration_ms',
                'mode',
                'key',
                'release_date_month',
                'release_date_day',
                'artist_followers'
            ],
            axis=1
        )

        # After dropping the previous columns the only NaN values will be
        # Artist Followers, so we can replace that with 0 ( zero )
        train_subset.fillna(value=0, inplace=True)
        test_subset.fillna(value=0, inplace=True)

        # Display detailed NaN Value Report
        self.__detailed_nan_report(
            train=train_subset,
            test=test_subset,
            timing='after'
        )

        # Run Neural Network Process
        self.__neural_network_process(
            training_data=train_subset,
            testing_data=test_subset
        )

        # Split the Data
        X_train, X_test, y_train, y_test = \
            self.__dataloader.split_data(
                label='loudness',
                data=train_subset,
                test_data=test_subset,
                validation=False
            )

        X_train_no_outliers, y_train_final = \
            self.__dataloader.remove_outliers(
                features=X_train.values,
                labels=y_train.values
            )

        X_test_no_outliers, y_test_final = \
            self.__dataloader.remove_outliers(
                features=X_test.values,
                labels=y_test.values
            )

        columns = X_train.columns

        self.__info_and_description_for_every_dataframe(
            train=pd.DataFrame(
                data=X_train_no_outliers,
                columns=columns
            ),
            test=pd.DataFrame(
                data=X_test_no_outliers,
                columns=columns
            ),
            titles=[
                "Information about the Training Data Set after ignoring "
                "Outlier Values and Not Strongly Correlated Columns",
                "Information about the Testing Data Set after ignoring "
                "Outlier Values and Not Strongly Correlated Columns"
            ]
        )

        self.__plotter.display_histograms(
            data=X_train,
            skip_columns=['explicit'],
            title='Unscaled Training Data Set',
            bins=50,
            hue='explicit',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'histograms',
                'unscaled',
                'train'
            )
        )

        self.__plotter.display_histograms(
            data=X_test,
            skip_columns=['explicit'],
            title='Unscaled Testing Data Set',
            bins=50,
            hue='explicit',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'histograms',
                'unscaled',
                'test'
            )
        )

        scaler = StandardScaler()

        # Scale the Features
        X_train_final, X_test_final = self.__dataloader.scale_data(
            scaler=scaler,
            X_train=X_train_no_outliers,
            X_test=X_test_no_outliers
        )

        # Run the Random Forest Regression Process
        self.__random_forest_process(
            X_train=X_train_final,
            y_train=y_train_final,
            X_test=X_test_final,
            y_test=y_test_final,
            columns=columns,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'decision_trees'
            ),
        )

        self.__plotter.display_histograms(
            data=pd.DataFrame(X_train, columns=columns),
            skip_columns=['explicit'],
            title='Scaled Training Data Set',
            bins=50,
            hue='explicit',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'histograms',
                'scaled',
                'train'
            )
        )

        self.__plotter.display_histograms(
            data=pd.DataFrame(X_test, columns=columns),
            skip_columns=['explicit'],
            title='Scaled Testing Data Set',
            bins=50,
            hue='explicit',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'histograms',
                'scaled',
                'test'
            )
        )

        self.__info_and_description_for_every_dataframe(
            train=pd.DataFrame(X_train_final, columns=columns),
            test=pd.DataFrame(X_test_final, columns=columns),
            titles=[
                "Training Data Set after Scaling",
                "Testing Data Set after Scaling"
            ]
        )

        return X_train_final, X_test_final, y_train_final, y_test_final

    def run(self) -> None:
        """Run the whole Support Vector Machines Assignment."""

        # Remove plt.figure warning
        plt.rcParams.update({'figure.max_open_warning': 0})

        colorama.init()

        # Set Pandas Display Options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)

        # Read Data
        training_data = self.__dataloader.load_file(
            self.__config['data']['training_data']
        )

        testing_data = self.__dataloader.load_file(
            self.__config['data']['testing_data']
        )

        self.__run_eda(
            training_data=training_data,
            testing_data=testing_data
        )

        X_train, X_test, y_train, y_test = \
            self.__run_data_processing(
                training_data=training_data,
                testing_data=testing_data
            )

        svm = SupportVectorMachine()

        svm.build()

        svm.train(X_train=X_train, y_train=y_train)

        svm.evaluate(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            file_name='svm_default_settings.png'
        )

        svm2 = SupportVectorMachine()

        svm2.build()

        svm2.bagging_regression(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        svm3 = SupportVectorMachine()

        svm3.build()

        svm3.boosting_regression(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        svm4 = SupportVectorMachine()

        svm4.build()

        svm4.grid_search(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        plt.show()
