import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Dict, List, Any


class Plotter:

    def __init__(self) -> None:
        """Initializes the Plotter Class."""

        self.__histogram_color = "rocket"
        self.__heatmap_color = "magma"

    def display_histograms(
            self,
            data: pd.DataFrame,
            skip_columns: List[str],
            title: str,
            units: Dict[str, str] = None,
            bins: Union[int, str] = 50,
            hue: str = None
    ) -> None:
        """Display Histogram for columns in DataFrame except the skip columns.

        Args:
            data (pandas.DataFrame) : Data to display
            skip_columns (List[str]) : Columns we do not want to display
            title (str) : Title of the plot
            units (Dict[str, str]) : Unit of each attribute
            bins (Union[int, str]) : Bin width accepts int and 'auto' keyword
            hue (str) : Displays different categories of Data

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
                legend=False if (hue and hue == 'type') else True,
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
            if hue and hue == 'type':
                ax.legend(title='Type', loc='upper right',
                          labels=['White', 'Red'])

            # Set y label
            ax.set_ylabel('Count')

            # Display the plot
            plt.draw()

    def display_heatmap(self, data: pd.DataFrame) -> None:
        """Display Heatmap of a given DataFrame.

        Args:
            data (pandas.DataFrame) : Data to display

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

        # Display the plot
        plt.draw()

    @staticmethod
    def display_unique_column_values(
            data: pd.DataFrame,
            column: str,
            kind: str,
            title: str
    ) -> None:
        """Display unique value counts in DataFrame column.

        Currently supports only pie charts :)

        Args:
            data (pandas.DataFrame) : Data to work with
            column (str) : Column we are interested in
            kind (str) : Type of plot to display,
            title (str) : Plot Title

        """

        # Inspiration for the Pie Chart Percentage and Value format:
        # https://stackoverflow.com/questions/6170246/how-do-i-use-matplotlib-autopct

        # Dictionary Value Pairs
        wine_type = {
            0: 'White',
            1: 'Red'
        }

        quality = {
            0: 'Low Quality',
            1: 'High Quality'
        }

        # Get all the Unique Values and their count in column
        unique_values = data[column].value_counts().reset_index()

        # Change values if we need one of these columns
        if column == 'type':
            unique_values['index'].replace(wine_type, inplace=True)
        elif column == 'quality':
            unique_values['index'].replace(quality, inplace=True)

        # If we need a Pie chart
        if kind == 'pie':
            plt.figure()

            plt.pie(
                x=unique_values[column],
                colors=sns.color_palette("rocket_r", len(unique_values)),
                startangle=90,
                autopct=lambda
                    p: f'{p:.2f}%  ({int(round(p * sum(unique_values[column]) / 100.0)):,.0f})'
            )

        # Add title to plot
        plt.title(title)

        # Add legend to plot
        plt.legend(
            unique_values['index'],
            bbox_to_anchor=(1, 1),
            loc="upper right",
            bbox_transform=plt.gcf().transFigure
        )

        plt.draw()

    @staticmethod
    def __draw_data_plot(
            data: pd.DataFrame,
            columns: List[str],
    ) -> None:
        # TODO - Docstring

        # Create Plot
        plt.figure(figsize=(16, 9))

        # Draw Plot
        data[columns].plot()

        plt.draw()

    def display_training_history(self, history: pd.DataFrame) -> None:
        """Display Training History Plots.

        Args:
            history (pandas.DataFrame) : Training History Data

        """

        self.__draw_data_plot(
            data=history,
            columns=[
                'loss',
                'val_loss'
            ]
        )

        self.__draw_data_plot(
            data=history,
            columns=[
                'accuracy',
                'val_accuracy'
            ]
        )

    @staticmethod
    def display_confusion_matrix(
            confusion_matrix: np.ndarray,
            labels: List[Any]
    ) -> None:
        """Display the passed in Confusion Matrix.

        Args:
            confusion_matrix (numpy.ndarray) : Confusion Matrix to display
            labels (List[Any]) : Labels for Columns and Indexes

        """

        # SOURCE:
        # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix

        # Convert Confusion Matrix to DataFrame
        confusion_matrix = pd.DataFrame(
            data=confusion_matrix,
            index=labels,
            columns=labels
        )

        # Create new Figure
        plt.figure(figsize=(16, 9))

        # Plot the Confusion Matrix
        sns.heatmap(confusion_matrix, annot=True, fmt='g')

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')

        plt.draw()
