import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from ops.api_caller import APICaller
from yellowbrick.regressor import ResidualsPlot
from typing import Union, Dict, List, Any, Optional, Tuple


class Plotter:

    def __init__(self) -> None:
        """Initializes the Plotter Class."""

        self.__histogram_color = "rocket"
        self.__heatmap_color = "magma"

        # Create an instance of APICaller
        self.__api_caller = APICaller()

    def display_histograms(
            self,
            data: pd.DataFrame,
            skip_columns: List[str],
            title: str,
            units: Dict[str, str] = None,
            bins: Union[int, str] = 50,
            hue: str = None,
            save: bool = False,
            path: Optional[str] = None
    ) -> None:
        """Display Histogram for columns in DataFrame except the skip columns.

        Args:
            data (pandas.DataFrame): Data to display
            skip_columns (List[str]): Columns we do not want to display
            title (str): Title of the plot
            units (Dict[str, str]): Unit of each attribute
            bins (Union[int, str]): Bin width accepts int and 'auto' keyword
            hue (str): Displays different categories of Data
            save (bool): Save the plot or not
            path (Optional[str]): Save Folder Path

        """

        # Go through columns except the excluded columns
        for column in data.columns[~np.isin(data.columns, skip_columns)]:

            # Create new figure
            figure = plt.figure(figsize=(16, 9))

            # Add simple axes to plot on
            ax = figure.add_subplot(1, 1, 1)

            # Plot the Histogram based on parameters to the axes
            sns.histplot(
                data=data,
                x=column,
                bins=bins,
                palette=self.__histogram_color,
                hue=hue,
                legend=False if (hue and hue == 'explicit') else True,
                multiple="stack",
                ax=ax
            )

            # Set title for the plot
            ax.set_title(f'{title} - {column.capitalize()}')

            # Set x label for the plot
            # If we don't have any units specified we only display the column
            # name
            ax.set_xlabel(f'{column.capitalize()}')

            # If we have specified a units dictionary we will display the units
            # for the matching columns
            if units and units[column]:
                ax.set_xlabel(f'{column.capitalize()} ( {units[column]} )')

            # If we want to display the different types of wine
            if hue and hue == 'explicit':
                ax.legend(
                    title='Explicit', loc='upper right',
                    labels=['Explicit', 'Family Friendly']
                )

            # Set y label
            ax.set_ylabel('Count')

            # If we want to save the plots
            if save and (path is not None):

                # Generate file name from title
                file_name = '_'.join(list(map(str.lower, title.split(' '))))

                # Draw the plot, so when we save it we make sure it is not blank
                plt.draw()

                # Save figure based on parameters
                plt.savefig(f"{path}/{file_name}_{column}.png")

            else:
                # Display the plot
                plt.draw()

    @staticmethod
    def display_pair_plot(
            data: pd.DataFrame,
            save: bool = False,
            path: Optional[str] = None,
            hue: Optional[str] = None
    ) -> None:
        """Display Pair Plot for DataFrame.

        Args:
            data (pandas.DataFrame): Data to work with
            save (bool): Save the plot or not
            path (Optional[str]): Save Folder Path
            hue (Optional[str]): Hue column

        """

        fig = plt.figure(figsize=(16, 9))

        # Create the Heatmap
        sns.pairplot(
            data=data,
            hue=hue
        )

        # If we want to save the plots
        if save and (path is not None):

            # Draw the plot, so when we save it we make sure it is not blank
            plt.draw()

            # Save figure based on parameters
            plt.savefig(f"{path}/pair_plot.png")

        else:
            # Display the plot
            plt.draw()

    @staticmethod
    def display_count_plot(
            data: pd.DataFrame,
            column: str,
            data_type: str,
            title: str,
            save: bool = False,
            path: Optional[str] = None
    ) -> None:
        """Display Count Plot for a given column in DataFrame.

        Args:
            data (pandas.DataFrame): Data to work with
            column (str): Column to display information about
            data_type (str): Represents the type of DataFrame ( train | test )
            title (str): Plot title
            save (bool): Save the plot or not
            path (Optional[str]): Save Folder Path

        """

        # Create new figure
        figure = plt.figure(figsize=(16, 9))

        # Add simple axes to plot on
        ax = figure.add_subplot(1, 1, 1)

        # Create the Heatmap
        sns.countplot(
            data=data,
            x=column,
            order=data[column].value_counts().index,
            ax=ax
        ).set_title(
            f"{title} - {' '.join(list(map(str.capitalize, column.split('_'))))}"
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

        plt.tight_layout()

        # If we want to save the plots
        if save and (path is not None):

            # Draw the plot, so when we save it we make sure it is not blank
            plt.draw()

            # Save figure based on parameters
            plt.savefig(f"{path}/{data_type}_{column}_count_plot.png")
            plt.show(block=False)

        else:
            # Display the plot
            plt.draw()
            plt.show(block=False)

    def display_choropleth(self, data: pd.DataFrame) -> None:
        """Display Choropleth Map of Countries which listen to the most popular
        song of the DataFrame (which has associated Geo Data - Spotipy - Market)

        Args:
            data (pandas.DataFrame): Data to work with

        """

        # Sort the data base on 2 factors Artist Followers and Popularity
        sorted_data = data.sort_values(
            by=['popularity', 'artist_followers'],
            ascending=False
        )

        # Helping variables to determine if the most popular song has Geo Market
        # Data or not
        index = -1
        market = []

        # While we do not find a track that has Geo Market Data
        while not market:
            # We search for the next track using the Spotipy Package
            index += 1

            market = self.__api_caller.get_market(
                track=sorted_data.iloc[index]['id']
            )

        # Country Codes to encode ISO 3166-1 alpha-2 Country names to
        # ISO 3166-1 alpha-3 ones. We've tried the pycountry_convert package,
        # but the Republic of Kosovo threw exceptions.

        # SOURCE:
        # https://stackoverflow.com/a/56503073
        country_codes = {
            'AF': 'AFG',
            'AX': 'ALA',
            'AL': 'ALB',
            'DZ': 'DZA',
            'AS': 'ASM',
            'AD': 'AND',
            'AO': 'AGO',
            'AI': 'AIA',
            'AQ': 'ATA',
            'AG': 'ATG',
            'AR': 'ARG',
            'AM': 'ARM',
            'AW': 'ABW',
            'AU': 'AUS',
            'AT': 'AUT',
            'AZ': 'AZE',
            'BS': 'BHS',
            'BH': 'BHR',
            'BD': 'BGD',
            'BB': 'BRB',
            'BY': 'BLR',
            'BE': 'BEL',
            'BZ': 'BLZ',
            'BJ': 'BEN',
            'BM': 'BMU',
            'BT': 'BTN',
            'BO': 'BOL',
            'BA': 'BIH',
            'BW': 'BWA',
            'BV': 'BVT',
            'BR': 'BRA',
            'IO': 'IOT',
            'BN': 'BRN',
            'BG': 'BGR',
            'BF': 'BFA',
            'BI': 'BDI',
            'KH': 'KHM',
            'CM': 'CMR',
            'CA': 'CAN',
            'CV': 'CPV',
            'KY': 'CYM',
            'CF': 'CAF',
            'TD': 'TCD',
            'CL': 'CHL',
            'CN': 'CHN',
            'CX': 'CXR',
            'CC': 'CCK',
            'CO': 'COL',
            'KM': 'COM',
            'CG': 'COG',
            'CD': 'COD',
            'CK': 'COK',
            'CR': 'CRI',
            'CI': 'CIV',
            'HR': 'HRV',
            'CU': 'CUB',
            'CY': 'CYP',
            'CZ': 'CZE',
            'DK': 'DNK',
            'DJ': 'DJI',
            'DM': 'DMA',
            'DO': 'DOM',
            'EC': 'ECU',
            'EG': 'EGY',
            'SV': 'SLV',
            'GQ': 'GNQ',
            'ER': 'ERI',
            'EE': 'EST',
            'ET': 'ETH',
            'FK': 'FLK',
            'FO': 'FRO',
            'FJ': 'FJI',
            'FI': 'FIN',
            'FR': 'FRA',
            'GF': 'GUF',
            'PF': 'PYF',
            'TF': 'ATF',
            'GA': 'GAB',
            'GM': 'GMB',
            'GE': 'GEO',
            'DE': 'DEU',
            'GH': 'GHA',
            'GI': 'GIB',
            'GR': 'GRC',
            'GL': 'GRL',
            'GD': 'GRD',
            'GP': 'GLP',
            'GU': 'GUM',
            'GT': 'GTM',
            'GG': 'GGY',
            'GN': 'GIN',
            'GW': 'GNB',
            'GY': 'GUY',
            'HT': 'HTI',
            'HM': 'HMD',
            'VA': 'VAT',
            'HN': 'HND',
            'HK': 'HKG',
            'HU': 'HUN',
            'IS': 'ISL',
            'IN': 'IND',
            'ID': 'IDN',
            'IR': 'IRN',
            'IQ': 'IRQ',
            'IE': 'IRL',
            'IM': 'IMN',
            'IL': 'ISR',
            'IT': 'ITA',
            'JM': 'JAM',
            'JP': 'JPN',
            'JE': 'JEY',
            'JO': 'JOR',
            'KZ': 'KAZ',
            'KE': 'KEN',
            'KI': 'KIR',
            'KP': 'PRK',
            'KR': 'KOR',
            'KW': 'KWT',
            'KG': 'KGZ',
            'LA': 'LAO',
            'LV': 'LVA',
            'LB': 'LBN',
            'LS': 'LSO',
            'LR': 'LBR',
            'LY': 'LBY',
            'LI': 'LIE',
            'LT': 'LTU',
            'LU': 'LUX',
            'MO': 'MAC',
            'MK': 'MKD',
            'MG': 'MDG',
            'MW': 'MWI',
            'MY': 'MYS',
            'MV': 'MDV',
            'ML': 'MLI',
            'MT': 'MLT',
            'MH': 'MHL',
            'MQ': 'MTQ',
            'MR': 'MRT',
            'MU': 'MUS',
            'YT': 'MYT',
            'MX': 'MEX',
            'FM': 'FSM',
            'MD': 'MDA',
            'MC': 'MCO',
            'MN': 'MNG',
            'ME': 'MNE',
            'MS': 'MSR',
            'MA': 'MAR',
            'MZ': 'MOZ',
            'MM': 'MMR',
            'NA': 'NAM',
            'NR': 'NRU',
            'NP': 'NPL',
            'NL': 'NLD',
            'AN': 'ANT',
            'NC': 'NCL',
            'NZ': 'NZL',
            'NI': 'NIC',
            'NE': 'NER',
            'NG': 'NGA',
            'NU': 'NIU',
            'NF': 'NFK',
            'MP': 'MNP',
            'NO': 'NOR',
            'OM': 'OMN',
            'PK': 'PAK',
            'PW': 'PLW',
            'PS': 'PSE',
            'PA': 'PAN',
            'PG': 'PNG',
            'PY': 'PRY',
            'PE': 'PER',
            'PH': 'PHL',
            'PN': 'PCN',
            'PL': 'POL',
            'PT': 'PRT',
            'PR': 'PRI',
            'QA': 'QAT',
            'RE': 'REU',
            'RO': 'ROU',
            'RU': 'RUS',
            'RW': 'RWA',
            'BL': 'BLM',
            'SH': 'SHN',
            'KN': 'KNA',
            'LC': 'LCA',
            'MF': 'MAF',
            'PM': 'SPM',
            'VC': 'VCT',
            'WS': 'WSM',
            'SM': 'SMR',
            'ST': 'STP',
            'SA': 'SAU',
            'SN': 'SEN',
            'RS': 'SRB',
            'SC': 'SYC',
            'SL': 'SLE',
            'SG': 'SGP',
            'SK': 'SVK',
            'SI': 'SVN',
            'SB': 'SLB',
            'SO': 'SOM',
            'ZA': 'ZAF',
            'GS': 'SGS',
            'ES': 'ESP',
            'LK': 'LKA',
            'SD': 'SDN',
            'SR': 'SUR',
            'SJ': 'SJM',
            'SZ': 'SWZ',
            'SE': 'SWE',
            'CH': 'CHE',
            'SY': 'SYR',
            'TW': 'TWN',
            'TJ': 'TJK',
            'TZ': 'TZA',
            'TH': 'THA',
            'TL': 'TLS',
            'TG': 'TGO',
            'TK': 'TKL',
            'TO': 'TON',
            'TT': 'TTO',
            'TN': 'TUN',
            'TR': 'TUR',
            'TM': 'TKM',
            'TC': 'TCA',
            'TV': 'TUV',
            'UG': 'UGA',
            'UA': 'UKR',
            'AE': 'ARE',
            'GB': 'GBR',
            'US': 'USA',
            'UM': 'UMI',
            'UY': 'URY',
            'UZ': 'UZB',
            'VU': 'VUT',
            'VE': 'VEN',
            'VN': 'VNM',
            'VG': 'VGB',
            'VI': 'VIR',
            'WF': 'WLF',
            'EH': 'ESH',
            'XK': 'XKX',
            'YE': 'YEM',
            'ZM': 'ZMB',
            'ZW': 'ZWE'
        }

        # SOURCE:
        # https://plotly.com/python/choropleth-maps/

        # We create a Choropleth Map, displaying the Artist, the Track and
        # all the countries that is available in
        fig = px.choropleth(
            locations=[country_codes.get(item) for item in market],
            title=f"Countries listening to "
                  f"{sorted_data.iloc[index]['artist']} - "
                  f"{sorted_data.iloc[index]['name']}"
        )

        # Center the title
        fig.update_layout(title_x=0.5)

        # Display the Choropleth Map
        fig.show()

    def display_heatmap(
            self,
            data: pd.DataFrame,
            save: bool = False,
            path: Optional[str] = None
    ) -> None:
        """Display Heatmap of a given DataFrame.

        Args:
            data (pandas.DataFrame): Data to display
            save (bool): Save the plot or not
            path (Optional[str]): Save Folder Path

        """

        # Calculate correlation
        correlation = data.corr()

        # Create new figure
        figure = plt.figure(figsize=(16, 9))

        # Add simple axes to plot on
        ax = figure.add_subplot(1, 1, 1)

        # Create the Heatmap
        sns.heatmap(
            data=correlation,
            ax=ax,
            cmap=self.__heatmap_color,
            annot=True,
            linewidths=.3,
        )

        plt.tight_layout()

        # If we want to save the plots
        if save and (path is not None):

            # Draw the plot, so when we save it we make sure it is not blank
            plt.draw()
            plt.show(block=False)

            # Save figure based on parameters
            plt.savefig(f"{path}/train_correlation_heatmap.png")

        else:
            # Display the plot
            plt.draw()
            plt.show(block=False)

    @staticmethod
    def display_word_cloud(
            data: pd.DataFrame,
            column: str,
            title: Optional[str] = None,
            save: bool = False,
            path: Optional[str] = None
    ) -> None:
        """ Display Word Cloud for the given DataFrame.

        Args:
            data (pandas.DataFrame): Data to work with
            column (str): Column to process
            title (Optional[str]): Save File Name
            save (bool): Save the plot or not
            path (Optional[str]): Save Folder Path

        """

        # Get all the text data from the given column
        track_names = ' '.join(data[column])

        # SOURCE:
        # https://www.datacamp.com/community/tutorials/wordcloud-python

        # Create and generate a word cloud image:
        wordcloud = WordCloud(
            width=1200,
            height=750,
        ).generate(track_names)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        # If we want to save the plots
        if save and (path is not None):

            # Draw the plot, so when we save it we make sure it is not blank
            plt.draw()

            # Save figure based on parameters
            plt.savefig(f"{path}/{title}_{column}_word_cloud.png")
            plt.show(block=False)

        else:
            # Display the plot
            plt.draw()
            plt.show(block=False)

    @staticmethod
    def display_top_10(
            data: pd.DataFrame,
            base_columns: List[str],
            columns_to_display: Tuple[str, str],
            title: Optional[str] = None,
            save: bool = False,
            path: Optional[str] = None
    ) -> None:
        """Display Top 10 in given columns based on other columns.

        Args:
            data (pandas.DataFrame): Data to work with
            base_columns (List[str]): Based on these columns select Top 10
            columns_to_display (Tuple[str]): Columns to display
            title (Optional[str]): Save File Name
            save (bool): Save the plot or not
            path (Optional[str]): Save Folder Path

        """

        # Sort the data base on 2 factors Artist Followers and Popularity
        sorted_data = data.sort_values(
            by=base_columns,
            ascending=False
        )

        # Create new figure
        figure = plt.figure(figsize=(16, 9))

        # Add simple axes to plot on
        ax = figure.add_subplot(1, 1, 1)

        # Display Bar Plot of the Top 10
        sns.barplot(
            data=sorted_data.iloc[:10],
            x=columns_to_display[0],
            y=columns_to_display[1],
            ax=ax
        )

        # Set title for the plot
        ax.set_title(f'{title.capitalize()} Data Set - Top 10 Tracks')

        # Rotate the Bar Names to avoid clashes
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

        plt.tight_layout()

        # If we want to save the plots
        if save and (path is not None):

            # Draw the plot, so when we save it we make sure it is not blank
            plt.draw()

            # Save figure based on parameters
            plt.savefig(
                f"{path}/{title}_top_10_{columns_to_display[1]}.png"
            )
            plt.show(block=False)

        else:
            # Display the plot
            plt.draw()
            plt.show(block=False)

    @staticmethod
    def __draw_data_plot(
            data: pd.DataFrame,
            columns: List[str],
    ) -> None:
        """Plot columns of DataFrame.

        Args:
            data (pandas.DataFrame): Data to work with
            columns (List[str]): Columns to display

        """

        # Create Plot
        plt.figure(figsize=(16, 9))

        # Draw Plot
        data[columns].plot()

        plt.draw()
        plt.show(block=False)

    def display_training_history(self, history: pd.DataFrame) -> None:
        """Display Training History Plots.

        Args:
            history (pandas.DataFrame): Training History Data

        """

        self.__draw_data_plot(
            data=history,
            columns=[
                'loss',
                'val_loss'
            ]
        )

    @staticmethod
    def display_residuals(
            model: Any,
            X_train: pd.Series,
            y_train: pd.Series,
            X_test: pd.Series,
            y_test: pd.Series,
            file_path: str
    ) -> None:
        """Display Residuals in a simple plot.

        Args:
            model (Any): Regression Model
            X_train (pandas.Series): Training Feature Set
            y_train (pandas.Series): Training Label Set
            X_test (pandas.Series): Testing Feature Set
            y_test (pandas.Series): Testing Label Set
            file_path (str): Save File Path

        """

        # SOURCE:
        # https://www.scikit-yb.org/en/latest/api/regressor/residuals.html

        visualizer = ResidualsPlot(model)

        # Fit the training data to the visualizer
        visualizer.fit(X_train, y_train)

        # Evaluate the model on the test data
        visualizer.score(X_test, y_test)

        # Finalize and render the figure
        visualizer.show(outpath=file_path)
