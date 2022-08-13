import ast
import os
from datetime import datetime
from turtle import distance
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import spatial
from termcolor import colored

from data_structures.graph import Graph


class Assignment4:

    def __init__(self, input_file: str) -> None:
        """Initializes the Assignment 4 Class.

        Args:
            input_file (str): Input file path.
        """

        self.__input_file: str = input_file

        # Generate output file path
        self.__output_file: str = os.path.join(
            os.getcwd(),
            'data',
            'output',
            f"{os.path.basename( self.__input_file).split('.')[0]}_output.txt"
        )

        # Truncate the content of file
        open(self.__output_file, 'w').close()

        # Placeholder for graph
        self.__graph = None

        # Edges and coordinate mappings
        self.__edges, self.__graph_point_dictionary, \
            self.__inv_graph_point_dictionary = self.__read_data()

        # Connected Components
        self.__connected_components = []

    def __read_data(
        self
    ) -> Tuple[List[List[int]], Dict[str, int], Dict[str, int]]:
        """Reads and Preprocesses Data.

        Returns:
            Tuple[List[List[int]], Dict[str, int], Dict[str, int]]: Edges and 
            Coordinate Mapping.
        """

        # Coordinate mappings for easier component connection search
        graph_point_dictionary: Dict[str, int] = {}
        inv_graph_point_dictionary: Dict[int, str] = {}

        # Timer for Data Reading Process
        start_time = datetime.now()

        with open(self.__input_file) as input_file:
            lines = input_file.readlines()

            graph_points = []
            edges = []

            for line in lines:
                data: List[str] = line.split()
                edges.append(data)
                graph_points.extend(data)

            for graph_point in sorted(set(graph_points)):
                inv_graph_point_dictionary[
                    len(graph_point_dictionary.keys())
                ] = graph_point

                graph_point_dictionary[graph_point] = len(
                    graph_point_dictionary.keys()
                )

            # Create Graph
            self.__graph = Graph(len(graph_point_dictionary.keys()))

            # Fill Graph
            for index, edge in enumerate(edges):
                edges[index] = [
                    graph_point_dictionary[edge[0]],
                    graph_point_dictionary[edge[1]]
                ]

                self.__graph.add_edge(
                    graph_point_dictionary[edge[0]],
                    graph_point_dictionary[edge[1]]
                )

        # Stop Timer and Display Processing Time
        end_time = datetime.now()
        print(
            colored('Data Reading Completed in:', 'green'),
            end_time - start_time,
        )

        return edges, graph_point_dictionary, inv_graph_point_dictionary

    def __connect_nearest_components(self, ouput_file: str) -> None:
        """Connects nearest connected components in graph.

        Args:
            ouput_file (str): Added Edges and Total Distance Output file path.
        """

        # SOURCE:
        # https://kanoki.org/2020/08/05/find-nearest-neighbor-using-kd-tree/
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html

        # Total distance counter
        min_sum = 0

        # Only connected component
        # (This variable will store all the verticies that belong to the large
        # connected component.)
        array = self.__connected_components.pop(0)

        # Iterate through the rest of unconnected connected components
        while len(self.__connected_components) != 0:

            # Create KD Tree from the existing large connected component
            tree = spatial.KDTree(array)

            # List to store distances from other unconnected connected
            # components
            current_values = []

            for index, connected_component in enumerate(
                self.__connected_components
            ):
                # Check to which point in tree is the closest the given
                # unconnected connected component
                results = tree.query(connected_component, k=1, workers=-1)

                # Find the minimal distance value in results
                min_distance: float = np.amin(results[0])

                # Find the starting point in results
                start = array[results[1][0]]

                # Find ending point based on the minimal distance
                end = connected_component[
                    np.where(results[0] == min_distance)[0][0]
                ]

                # Save the index and values of current result
                current_values.append(
                    (
                        index,
                        sorted([start, end]),
                        min_distance
                    )
                )

            # Select the best result out of collected results
            min_value = sorted(current_values, key=lambda x: x[2])[0]

            # Connect new component to the large one
            array.extend(self.__connected_components.pop(min_value[0]))

            # Save values to output file
            with open(ouput_file, 'a') as output:
                # Increase total distance counter
                min_sum += min_value[2]

                # Write the edge to file
                output.write(
                    f'{min_value[1][0]} {min_value[1][1]}\n'
                )

        # Save total distance to output file
        with open(ouput_file, 'a') as output:
            output.write(f'\n--------- Total distance ---------\n{min_sum}')

    def __call__(self, *args: Any, **kwds: Any) -> None:
        """Executes Assignment4 for given input file."""

        # Timer for Data Processing
        start_time = datetime.now()

        # Fetch all the connected components
        connected_components = self.__graph.get_connected_components()

        # Map original coordinates to verticies
        for component in connected_components:
            for index, element in enumerate(component):
                component[index] = ast.literal_eval(
                    self.__inv_graph_point_dictionary[element]
                )

        # Save connected components
        self.__connected_components = connected_components

        # Connect nearest components
        self.__connect_nearest_components(self.__output_file)

        # Stop Timer and Display Processing Time
        end_time = datetime.now()
        print(
            colored('Data Processed in:', 'green'),
            end_time - start_time,
            end='\n\n\n'
        )
