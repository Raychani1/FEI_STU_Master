import os.path
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import io
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Union, Dict, List, Any, Optional
from keras.preprocessing.image import DirectoryIterator


class Plotter:

    def __init__(self) -> None:
        """Initializes the Plotter Class."""

        self.__histogram_color = "rocket"
        self.__heatmap_color = "magma"

    @staticmethod
    def display_bar_plot(
            data: pd.DataFrame,
            x_column: str,
            y_column: str,
            data_type: str,
            save: bool = False,
            path: Optional[str] = None
    ) -> None:
        """Display Count Plot for a given column in DataFrame.
        Args:
            data (pandas.DataFrame): Data to work with
            x_column (str): Column to display information about
            y_column (str): Column to display information about
            data_type (str): Represents the type of DataFrame ( train | test )
            save (bool): Save the plot or not
            path (Optional[str]): Save Folder Path

        """

        # Create new figure
        figure = plt.figure(figsize=(16, 9))

        # Add simple axes to plot on
        ax = figure.add_subplot(1, 1, 1)

        # Create the Heatmap
        sns.barplot(
            data=data,
            x=x_column,
            y=y_column,
        ).set_title(
            f'Number of Images by Category in {data_type.capitalize()}ing Data '
            f'Set'
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

        plt.tight_layout()

        # If we want to save the plots
        if save and (path is not None):

            # Draw the plot, so when we save it we make sure it is not blank
            plt.draw()

            # Save figure based on parameters
            plt.savefig(f"{path}/{data_type}_images_by_category_bar_plot.png")
            plt.show(block=False)

        else:
            # Display the plot
            plt.draw()
            plt.show(block=False)

    @staticmethod
    def display_samples(folder: str) -> None:
        """Display one sample from each category in given folder.
        Args:
            folder (str): Path to Folder containing Images sorted to Categories
        """

        for sub_folder in os.listdir(folder):
            path = os.path.join(folder, sub_folder)
            image = os.path.join(path, os.listdir(path)[0])

            plt.imshow(
                mpimg.imread(
                    image
                )
            )

            plt.title(
                f"{' '.join(list(map(str.capitalize, sub_folder.split('_'))))} "
                f"Sample 0 from {folder.split('/')[-1].capitalize()}ing Data "
                f"Set"
            )

            plt.show(block=False)

    @staticmethod
    def display_augmented_samples(data_generator: DirectoryIterator) -> None:
        """Display augmented samples.

        Args:
            data_generator (DirectoryIterator): Image Data Generator

        """

        x, y = data_generator.next()

        for i in range(0, 10):
            image = x[i]

            io.imshow(image)
            io.show()

    @staticmethod
    def __draw_data_plot(
            data: pd.DataFrame,
            columns: List[str],
            path: str
    ) -> None:
        # TODO - Docstring

        # Create Plot
        plt.figure(figsize=(16, 9))

        # Draw Plot
        data[columns].plot()

        # Draw the plot, so when we save it we make sure it is not blank
        plt.draw()

        # Save figure based on parameters
        plt.savefig(f"{path}.png")
        plt.show(block=False)

    def display_training_history(
            self,
            history: pd.DataFrame,
            mode: str
    ) -> None:
        """Display Training History Plots.

        Args:
            history (pandas.DataFrame): Training History Data
            mode (str): Mode of Train Execution

        """

        self.__draw_data_plot(
            data=history,
            columns=[
                'loss',
                'val_loss'
            ],
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'line_graphs',
                f'loss_{mode}_{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}'
            )
        )

        self.__draw_data_plot(
            data=history,
            columns=[
                'accuracy',
                'val_accuracy'
            ],
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'line_graphs',
                f'accuracy_{mode}_'
                f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}'
            )
        )

    @staticmethod
    def display_confusion_matrix(
            confusion_matrix: np.ndarray,
            labels: List[Any],
            path: str
    ) -> None:
        """Display the passed in Confusion Matrix.

        Args:
            confusion_matrix (numpy.ndarray): Confusion Matrix to display
            labels (List[Any]): Labels for Columns and Indexes
            path (str):
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
        figure = plt.figure(figsize=(16, 9))

        # Add simple axes to plot on
        ax = figure.add_subplot(1, 1, 1)

        # Plot the Confusion Matrix
        sns.heatmap(confusion_matrix, annot=True, fmt='g', ax=ax)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')

        plt.draw()
        plt.savefig(f"{path}.png")
        plt.show(block=False)

    @staticmethod
    def display_3d_scatter_plot(
            data: pd.DataFrame,
            value_columns: List[str],
            path: str
    ) -> None:
        """Display 3D Scatter Plot.

        Args:
            data (pandas.DataFrame): Data to display
            value_columns (List[str]): Value Column Names
            path (str): Path to output HTML File

        """

        # SOURCE:
        # https://plotly.com/python/3d-scatter-plots/
        # https://plotly.com/python/interactive-html-export/

        fig = px.scatter_3d(
            data_frame=data,
            x=value_columns[0],
            y=value_columns[1],
            z=value_columns[2],
            color=data['label']
        )

        fig.write_html(f'{path}.html')
