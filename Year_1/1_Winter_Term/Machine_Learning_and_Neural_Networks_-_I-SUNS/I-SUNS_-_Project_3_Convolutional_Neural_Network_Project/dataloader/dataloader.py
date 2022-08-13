import os
import numpy as np
import pandas as pd
from PIL import Image
import sklearn.preprocessing
from termcolor import colored
from typing import Tuple, Any, Union, Optional, Dict
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator


class DataLoader:

    def __init__(self) -> None:
        """Initialize the NeuralNetwork Class."""

        self.__mode = 'dealing with'

    @property
    def mode(self):
        """Return NaN Value Removal Strategy."""

        return self.__mode

    @staticmethod
    def load_all_data(
            target_size: int,
            batch_size: int,
            validation_percentage: float,
            augment: bool = False
    ) -> Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
        """Load file to DataFrame.

        Args:
            target_size (int): Target image size
            batch_size (int): Batch size for data generation
            validation_percentage (float): Percentage of Validation Data in
                Training Data
            augment (bool): Augment Data

        Returns:
            Tuple[DirectoryIterator,DirectoryIterator,DirectoryIterator]: Loaded
                Data

        """

        # SOURCE:
        # https://www.geeksforgeeks.org/cnn-image-data-pre-processing-with-generators/
        # https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator

        image_generator = ImageDataGenerator(
            rescale=1 / 255,
            validation_split=validation_percentage,
            horizontal_flip=True if augment else False,
            vertical_flip=True if augment else False,
            rotation_range=90 if augment else 0,

        )

        train_dataset = image_generator.flow_from_directory(
            batch_size=batch_size,
            directory=os.path.join(os.getcwd(), 'data', 'train'),
            target_size=(target_size, target_size),
            subset="training",
            class_mode='categorical'
        )

        validation_dataset = image_generator.flow_from_directory(
            batch_size=batch_size,
            directory=os.path.join(os.getcwd(), 'data', 'train'),
            target_size=(target_size, target_size),
            subset="validation",
            class_mode='categorical'
        )

        test_dataset = image_generator.flow_from_directory(
            batch_size=batch_size,
            directory=os.path.join(os.getcwd(), 'data', 'test'),
            target_size=(target_size, target_size),
            shuffle=False,
            class_mode='categorical'
        )

        return train_dataset, validation_dataset, test_dataset

    @staticmethod
    def count_samples(folder: str) -> Dict[str, int]:
        # TODO - Docstring

        samples: Dict[str, int] = dict()

        for sub_folder in sorted(os.listdir(folder)):
            samples[sub_folder] = len(
                os.listdir(
                    os.path.join(
                        folder,
                        sub_folder
                    )
                )
            )

        return samples
