import os
import umap
import colorama
import pandas as pd
import numpy as np
from pprint import pprint
from termcolor import colored

from models.support_vector_machine import SupportVectorMachine
from ops.plotter import Plotter
import matplotlib.pyplot as plt
from dataloader.dataloader import DataLoader
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from models.convolutional_neural_network import ConvolutionalNeuralNetwork


class ConvolutionalNeuralNetworkProject:

    def __init__(self):
        """Initialize the ConvolutionalNeuralNetworkProject Class."""

        # Create an instance of DataLoader
        self.__dataloader = DataLoader()

        # Create an instance of Plotter
        self.__plotter = Plotter()

        # Target Size for Images in this Project
        self.__target_size = 32

        # Batch Size for this Project
        self.__batch_size = 500

    def __run_eda(self):
        """Run Simple Exploratory Data Analysis for CNN Project."""

        # INSPIRATION:
        # https://medium.com/geekculture/eda-for-image-classification-dcada9f2567a

        # Display Bar Plot for Number of Images in the Training Data
        self.__plotter.display_bar_plot(
            data=pd.DataFrame(
                data=self.__dataloader.count_samples(
                    os.path.join(os.getcwd(), 'data', 'train')
                ).items(),
                columns=['Category', 'Number_of_Images']
            ),
            x_column='Category',
            y_column='Number_of_Images',
            data_type='train',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'bar_plots'
            )
        )

        # Display Bar Plot for Number of Images in the Testing Data
        self.__plotter.display_bar_plot(
            data=pd.DataFrame(
                data=self.__dataloader.count_samples(
                    os.path.join(os.getcwd(), 'data', 'test')
                ).items(),
                columns=['Category', 'Number_of_Images']
            ),
            x_column='Category',
            y_column='Number_of_Images',
            data_type='test',
            save=True,
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                'bar_plots'
            )
        )

        # Display Samples from the Training Data Set
        self.__plotter.display_samples(
            folder=os.path.join(
                os.getcwd(),
                'data',
                'train'
            )
        )

        # Display Samples from the Testing Data Set
        self.__plotter.display_samples(
            folder=os.path.join(
                os.getcwd(),
                'data',
                'test'
            )
        )

    @staticmethod
    def __display_information_about_data_set(
            data_set: DirectoryIterator,
            title: str
    ) -> None:
        """Display Information about Data Set.

        Args:
            data_set (DirectoryIterator): Data Set to work with
            title (str): Information for the user

        """

        print(colored(f'\nClasses in the {title.capitalize()} Data: ', 'green'))
        pprint(data_set.class_indices)

        print(
            colored(f'\nImage Shape in the {title.capitalize()} Data: ',
                    'green'),
            data_set.image_shape,
        )

    def __display_information_about_data_sets(
            self,
            training_data: DirectoryIterator,
            validation_data: DirectoryIterator,
            testing_data: DirectoryIterator
    ) -> None:
        """Display Information about all the Data Sets.

        Args:
            training_data (DirectoryIterator): Training Data Set
            validation_data (DirectoryIterator): Validation Data Set
            testing_data (DirectoryIterator): Testing Data Set

        """

        self.__display_information_about_data_set(
            title='training',
            data_set=training_data
        )
        self.__display_information_about_data_set(
            title='validation',
            data_set=validation_data
        )
        self.__display_information_about_data_set(
            title='testing',
            data_set=testing_data
        )

    def __extract_features(
            self,
            data_generator: DirectoryIterator,
            output_path: str
    ) -> None:
        """Extract Features from Images to CSV Files.

        Args:
            data_generator (DirectoryIterator): Generator containing Image Data
            output_path (str): Path to Output CSV File

        """

        # INSPIRATION:
        # https://keras.io/api/applications/

        # Temporary Data Storage
        feature_rows = []

        # Load trained VGG16 model with ImageNet Weights
        pretrained_model = VGG16(weights='imagenet', include_top=False)

        # Iterate through each File Path
        for filepath in data_generator.filepaths:
            # Process Image Data based on File Path
            list_of_features = pretrained_model.predict(
                preprocess_input(
                    np.expand_dims(
                        image.img_to_array(
                            image.load_img(
                                filepath,
                                target_size=(
                                    self.__target_size,
                                    self.__target_size
                                )
                            )
                        ),
                        axis=0
                    )
                )
            ).ravel().tolist()

            # Add the label
            list_of_features.insert(0, filepath.split('/')[-2])

            # Save temporarily
            feature_rows.append(list_of_features)

        # Set column names
        column_names = ['label']
        column_names.extend(list(range(0, len(list_of_features) - 1)))

        # Save values to DataFrame and Write to CSV File
        pd.DataFrame(
            feature_rows, columns=column_names
        ).to_csv(path_or_buf=output_path, index=False)

    # noinspection PyPep8Naming
    def run(self):
        """Run the whole Convolutional Neural Network Assignment."""

        # Remove plt.figure warning
        plt.rcParams.update({'figure.max_open_warning': 0})

        colorama.init()

        # # Run Simple Exploratory Data Analysis
        # self.__run_eda()

        # Read Data
        training_data, validation_data, testing_data = \
            self.__dataloader.load_all_data(
                target_size=self.__target_size,
                batch_size=self.__batch_size,
                validation_percentage=0.25,
                augment=False
            )

        # self.__plotter.display_augmented_samples(data_generator=training_data)
        #
        # # Display basic information about Data Sets
        # self.__display_information_about_data_sets(
        #     training_data=training_data,
        #     validation_data=validation_data,
        #     testing_data=testing_data
        # )
        #
        # print('Normal CNN')
        #
        # # Build Convolutional Neural Network
        # cnn = ConvolutionalNeuralNetwork(target_size=self.__target_size)
        #
        # cnn.build(
        #     loss_function='categorical_crossentropy',
        #     learning_rate=0.0001,
        #     mode='normal'
        # )
        #
        # # Train Convolutional Neural Network
        # cnn.train(
        #     training_data_generator=training_data,
        #     validation_data_generator=validation_data,
        #     batch_size=self.__batch_size,
        #     epochs=100,
        #     mode='normal',
        #     early_stopping=True,
        #     monitor='val_accuracy',
        #     patience=5
        # )
        #
        # # Evaluate Convolutional Neural Network
        # cnn.evaluate(
        #     testing_data_generator=testing_data,
        #     batch_size=self.__batch_size,
        #     mode='normal'
        # )
        #
        # print('Simple Overfit CNN - Load Weights')
        #
        # cnn2 = ConvolutionalNeuralNetwork(target_size=self.__target_size)
        #
        # cnn2.build(
        #     loss_function='categorical_crossentropy',
        #     learning_rate=0.0001,
        #     mode='normal',
        #     checkpoint_file=os.path.join(
        #         os.getcwd(),
        #         'models',
        #         'checkpoints',
        #         'checkpoint_normal_best_2021_12_09_-_03:04:52.hdf5'
        #     )
        # )
        #
        # # Train Convolutional Neural Network
        # cnn2.train(
        #     training_data_generator=training_data,
        #     validation_data_generator=validation_data,
        #     batch_size=self.__batch_size,
        #     epochs=0,
        #     mode='normal',
        #     early_stopping=False,
        #     # monitor='val_accuracy',
        #     # patience=10
        # )
        #
        # # Evaluate Convolutional Neural Network
        # cnn2.evaluate(
        #     testing_data_generator=testing_data,
        #     batch_size=self.__batch_size,
        #     mode='normal'
        # )
        #
        # print('Simple Overfit CNN')
        #
        # cnn3 = ConvolutionalNeuralNetwork(target_size=self.__target_size)
        #
        # cnn3.build(
        #     loss_function='categorical_crossentropy',
        #     learning_rate=0.0001,
        #     mode='overfit'
        # )
        #
        # # Train Convolutional Neural Network
        # cnn3.train(
        #     training_data_generator=training_data,
        #     validation_data_generator=validation_data,
        #     batch_size=self.__batch_size,
        #     epochs=40,
        #     mode='overfit',
        #     early_stopping=False,
        # )
        #
        # # Evaluate Convolutional Neural Network
        # cnn3.evaluate(
        #     testing_data_generator=testing_data,
        #     batch_size=self.__batch_size
        # )
        #
        # # Regularized CNN
        #
        # modes = [
        #     'l1_0.001',
        #     'l1_0.0001',
        #     'l1_0.00001',
        #     'l2_0.001',
        #     'l2_0.0001',
        #     'l2_0.00001'
        # ]
        #
        # for mode in modes:
        #
        #     reg, value = tuple(map(str.capitalize, mode.split('_')))
        #
        #     print(f'Regularisation {reg} CNN - {value}')
        #
        #     cnn4 = ConvolutionalNeuralNetwork(target_size=self.__target_size)
        #
        #     cnn4.build(
        #         loss_function='categorical_crossentropy',
        #         learning_rate=0.0001,
        #         mode=mode
        #     )
        #
        #     # Train Convolutional Neural Network
        #     cnn4.train(
        #         training_data_generator=training_data,
        #         validation_data_generator=validation_data,
        #         batch_size=self.__batch_size,
        #         epochs=60,
        #         mode=mode,
        #         early_stopping=False
        #     )
        #
        #     # Evaluate Convolutional Neural Network
        #     cnn4.evaluate(
        #         testing_data_generator=testing_data,
        #         batch_size=self.__batch_size,
        #         mode=mode
        #     )
        #
        #     del cnn4
        #
        # self.__extract_features(
        #     data_generator=training_data,
        #     output_path=os.path.join(
        #         os.getcwd(),
        #         'data',
        #         'encoded_data',
        #         'encoded_training_data.csv'
        #     )
        # )
        #
        # self.__extract_features(
        #     data_generator=validation_data,
        #     output_path=os.path.join(
        #         os.getcwd(),
        #         'data',
        #         'encoded_data',
        #         'encoded_validation_data.csv'
        #     )
        # )
        #
        # self.__extract_features(
        #     data_generator=testing_data,
        #     output_path=os.path.join(
        #         os.getcwd(),
        #         'data',
        #         'encoded_data',
        #         'encoded_testing_data.csv'
        #     )
        # )
        ####################################################################
        encoded_training_data = pd.read_csv(
            os.path.join(
                os.getcwd(),
                'data',
                'encoded_data',
                'encoded_training_data.csv'
            )
        )

        encoded_validation_data = pd.read_csv(
            os.path.join(
                os.getcwd(),
                'data',
                'encoded_data',
                'encoded_validation_data.csv'
            )
        )

        encoded_testing_data = pd.read_csv(
            os.path.join(
                os.getcwd(),
                'data',
                'encoded_data',
                'encoded_testing_data.csv'
            )
        )

        # Reduce Dimension of our Data

        transformer = umap.UMAP(
            n_neighbors=50,
            random_state=420,
            n_components=3
        ).fit(encoded_training_data.drop('label', axis=1))

        reduced_training_data = pd.DataFrame(
            transformer.transform(
                encoded_training_data.drop('label', axis=1)
            ),
            columns=['val1', 'val2', 'val3']
        )

        reduced_labeled_training_data = reduced_training_data.join(
            encoded_training_data['label']
        )

        reduced_validation_data = pd.DataFrame(
            transformer.transform(
                encoded_validation_data.drop('label', axis=1)
            ),
            columns=['val1', 'val2', 'val3']
        )

        reduced_labeled_validation_data = reduced_validation_data.join(
            encoded_validation_data['label']
        )

        reduced_testing_data = pd.DataFrame(
            transformer.transform(
                encoded_testing_data.drop('label', axis=1)
            ),
            columns=['val1', 'val2', 'val3']
        )

        reduced_labeled_testing_data = reduced_testing_data.join(
            encoded_testing_data['label']
        )

        X_train = reduced_labeled_training_data.drop('label', axis=1)
        y_train = reduced_labeled_training_data['label']

        X_val = reduced_labeled_validation_data.drop('label', axis=1)
        y_val = reduced_labeled_validation_data['label']

        X_test = reduced_labeled_testing_data.drop('label', axis=1)
        y_test = reduced_labeled_testing_data['label']

        # Encode New Genre Categories to Numerical Values
        label_encoder = LabelEncoder()

        y_train = label_encoder.fit_transform(y_train)
        y_val = label_encoder.transform(y_val)
        y_test = label_encoder.transform(y_test)

        svm = SupportVectorMachine()

        svm.build()

        svm.train(X_train=X_train, y_train=y_train)

        svm.evaluate(X_test=X_test, y_test=y_test, mode='normal')
        ####################################################################
        self.__plotter.display_3d_scatter_plot(
            data=reduced_labeled_validation_data,
            value_columns=['val1', 'val2', 'val3'],
            path=os.path.join(
                os.getcwd(),
                'output',
                'plots',
                '3d_scatter',
                'reduced_validation'
            )
        )

        # Make sure the plots are displayed
        plt.show()
