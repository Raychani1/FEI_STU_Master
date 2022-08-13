import os
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from termcolor import colored

from data_structures.node import Node


class OptimalBinarySearchTree:

    def __init__(self, word_frequency_limit: Optional[int] = 50000) -> None:
        """Initializes the OptimalBinarySearchTree Class.

        Args:
            word_frequency_limit (Optional[int]): Word Frequency Limit.
             Defaults to 50000.
        """

        # Word Frequency Limit for Optimal Binary Search Tree Nodes
        self.__word_frequency_limit = word_frequency_limit

        # Total Term Frequency for probability calculation
        self.__total_term_freq, self.__data = self.__read_data(
            os.path.join(
                os.getcwd(),
                'data',
                'dictionary.txt'
            )
        )

        # Dictionary with keys with above limit Term Frequencies and
        # Dummy Key Probabilities
        self.__dictionary: \
            Dict[str, Dict[str, Union[int, float]]] = self.__process_data()

        self.__primary_key_probabilities, self.__dummy_key_probabilities = \
            self.__fetch_key_probabilities()

        # Cost and Root Matrices
        self.__cost_matrix, self.__root_matrix = \
            self.__calculate_cost_and_root_matrices(
                n=len(self.__primary_key_probabilities) - 1
            )

        # Optimal Binary Search Tree
        self.__tree: Node = self.__build_tree(
            n=len(self.__primary_key_probabilities) - 1
        )

    @staticmethod
    def __read_data(path: str) -> Tuple[int, OrderedDict]:
        """Reads Input Data and calculates Total Term Frequency.

        Args:
            path (str): Input File Path

        Returns:
            Tuple[int, OrderedDict]: Total Term Frequency and Processed Input
             Data
        """

        # Timer for Data Reading Process
        start_time = datetime.now()

        # Dictionary to store words with their Term Frequency and Probability
        dictionary: Dict[str, Dict[str, Union[int, float]]] = {}

        # Sum of all Term Frequencies used for Probability calculation
        sum_total: int = 0

        # Read Data to Dictionary
        with open(path) as input_file:

            lines = input_file.readlines()

            for line in lines:
                data: List[Union[str, int]] = line.strip().split()
                data[0] = int(data[0])

                sum_total += data[0]

                dictionary[data[1]] = {'term_frequency': data[0]}

        # For each Word calculate Search Probability
        for key in dictionary.keys():
            dictionary[key]['probability'] = \
                dictionary[key]['term_frequency'] / sum_total

        # Stop Timer and Display Processing Time
        end_time = datetime.now()
        print(
            colored('Data Reading Completed in:', 'green'),
            end_time - start_time,
            end='\n\n'
        )

        # Order Dictionary based on Keys
        ordered_dictionary: OrderedDict = OrderedDict(
            sorted(dictionary.items())
        )

        return sum_total, ordered_dictionary

    def __dict_filter(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Selects dictionary entries which have the appropriate Term Frequency.

        Returns:
             Dict[str, Dict[str, Union[int, float]]]: Filtered Dictionary
        """

        return dict(
            [
                (i, self.__data[i]) for i in self.__data if
                self.__data[i]['term_frequency'] > self.__word_frequency_limit
            ]
        )

    def __calculate_dummy_key_probability(
            self,
            keys: List[str],
            start_key_index: int,
            stop_key_index: int
    ) -> float:
        """Calculates the Dummy Key Probability in key range.

        Args:
            keys (List[str]): Keys from the original Dictionary Input File
            start_key_index (int): Start key index
            stop_key_index (int): Stop key index

        Returns:
            float: Dummy Key Probability in given key range
        """

        range_sum: int = 0

        for key_index in range(start_key_index + 1, stop_key_index):
            range_sum += self.__data[keys[key_index]]['term_frequency']

        return range_sum / self.__total_term_freq

    def __process_data(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Processes data (Selects entries, Calculates Dummy Key Probabilities).

        Returns:
            Dict[str, Dict[str, Union[int, float]]]: Processed Data
        """

        # Timer for Data Processing
        start_time = datetime.now()

        # Fetch dictionary entries which have more than 50K Term Frequencies
        selected_data: Dict[str, Dict[str, Union[int, float]]] = \
            self.__dict_filter()

        # Add an empty key which will represent q_0 (since our first key is 'a'
        # and there is nothing before it)
        selected_data = {
            **{
                '': {
                    'dummy_key_probability': 0.0,
                    'probability': 0.0,
                    'term_frequency': 0
                }
            }, **selected_data
        }

        keys: List[str] = list(selected_data.keys())
        mixed_keys: List[str] = list(self.__data.keys())

        # Calculate Dummy Key Probabilities
        for i in range(1, len(keys) - 1):
            selected_data[keys[i]]['dummy_key_probability'] = \
                self.__calculate_dummy_key_probability(
                    keys=mixed_keys,
                    start_key_index=mixed_keys.index(keys[i]),
                    stop_key_index=mixed_keys.index(keys[i + 1])
            )

        # Calculate the q_n (the last Dummy Key Probability)
        selected_data[keys[-1]]['dummy_key_probability'] = \
            self.__calculate_dummy_key_probability(
                keys=mixed_keys,
                start_key_index=mixed_keys.index(keys[-1]),
                stop_key_index=len(mixed_keys)
        )

        # Stop Timer and Display Processing Time
        end_time = datetime.now()
        print(
            f"{colored('Data Processed in: ', 'green')}"
            f"{end_time - start_time}",
            end='\n\n'
        )

        return selected_data

    def __fetch_key_probabilities(self) -> Tuple[pd.Series, pd.Series]:
        """Fetches the Primary and Dummy Key Probabilities.

        Returns:
            Tuple[pd.Series, pd.Series]: Primary and Dummy Key Probabilities
        """

        primary_key_probabilities: List[float] = []
        dummy_key_probabilities: List[float] = []

        for key in self.__dictionary.keys():
            primary_key_probabilities.append(
                self.__dictionary[key]['probability'])
            dummy_key_probabilities.append(
                self.__dictionary[key]['dummy_key_probability'])

        return pd.Series(primary_key_probabilities), \
            pd.Series(dummy_key_probabilities)

    def __calculate_cost_and_root_matrices(self, n: int) -> Tuple[
            pd.DataFrame, pd.DataFrame]:
        """Calculates the Cost and Root Matrices.

        Args:
            n (int): Number of Primary Keys

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Cost and Root Matrices
        """

        # SOURCE: Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest,
        # Clifford Stein - Introduction to Algorithms, 3rd Edition,
        # Chapter 15.5, Page 402

        # Timer for Root Matrix Calculation Process
        start_time = datetime.now()

        # We've created Diagonal Matrices from Dummy Key Probabilities
        # Average / Optimal cost to search through every element in tree
        e = pd.DataFrame(
            np.diag(self.__dummy_key_probabilities),
            index=range(1, n + 2),
        )

        # Sum of Primary and Dummy Key Probabilities
        w = pd.DataFrame(
            np.diag(self.__dummy_key_probabilities),
            index=range(1, n + 2),
        )

        # We've created an empty Root Table
        root = pd.DataFrame(
            np.zeros((n, n)),
            index=range(1, n + 1),
            columns=range(1, n + 1)
        )

        # For every window length (1, 2, 3, ... n)
        for l in range(1, n + 1):

            # For every starting position of given window
            for i in range(1, n - l + 2):

                # Calculate where does the given window end
                j = i + l - 1

                # Initialize given cost to infinity
                e.at[i, j] = np.inf

                # Calculate the new sum of Primary and Dummy Key Probabilities
                w.at[i, j] = w.loc[i, j - 1] + \
                    self.__primary_key_probabilities[j] + \
                    self.__dummy_key_probabilities[j]

                # Try every combination in the given window, where r represent
                # the root of subtree
                for r in range(i, j + 1):

                    # Calculate the search cost for given tree
                    t = e._get_value(i, r - 1) + \
                        e._get_value(r + 1, j) + \
                        w._get_value(i, j)

                    # If the result is lower than the saved value we update that
                    # value, we also update our Root Matrix, because we've found
                    # a better solution
                    if t < e.at[i, j]:
                        e.at[i, j] = t
                        root.at[i, j] = r

        # Stop Timer and Display Processing Time
        end_time = datetime.now()
        print(
            colored('Root Matrix Calculated in:', 'green'),
            end_time - start_time,
            end='\n\n'
        )

        return e, root

    def __build_tree(self, n: int) -> Node:
        """Builds Optimal Binary Search Tree.

        Args:
            n (int) : Number of Keys in Dictionary

        Returns:
            Node: Root Node of Optimal Binary Search Tree
        """

        # SOURCE: Dr. Chris Burke - University of Nebraska -
        # Optimal Binary Search Trees
        # https://www.youtube.com/watch?v=CTUTPSXyBO8

        # Timer for Tree Building Process
        start_time = datetime.now()

        # Values that will be saved to our Optimal Binary Search Tree
        values: List[str] = list(self.__dictionary.keys())

        # Get the root of the tree
        # (Root of the Tree which contains Keys from 1 to n)
        root = Node(
            key=self.__root_matrix.at[1, n],
            value=values[int(self.__root_matrix._get_value(1, n))]
        )

        # We will use this stack for temporal calculations
        # A Tuple is saved, which consists of Node and the Range of Keys
        stack: List[Tuple[Node, int, int]] = [tuple([root, 1, n])]

        # While we have a Node to process
        while stack:

            # We take the Node we want to process
            u, i, j = stack.pop()

            # We search for it's Root key in given range
            # (This is only used for easier comparison)
            l = self.__root_matrix._get_value(i, j)

            # In this current form we might have
            # Left Subtree (K_i ... K_{l-1}),
            # Root Node (K_l) and
            # Right Subtree (K_{l+1} ... K_j)

            # If the upper bound is greater than our current key (K_l) so that
            # means that there is a present Right Subtree
            if l < j:
                # Build the Right Tree

                # Fetch the Key for the Right Subtree Range (K_{l+1} ... K_j)
                key_index: int = int(self.__root_matrix._get_value(l + 1, j))

                # Create new Node
                v = Node(
                    key=key_index,
                    value=str(values[key_index])
                )

                # Add to the Node we are currently processing as the Right Child
                u.right_child = v

                # Add it for further processing
                stack.append(tuple([v, l + 1, j]))

            # If the lower bound is lower than our current key (K_l) so that
            # means that there is a present Left Subtree
            if i < l:
                # Build the Left Tree

                # Fetch the Key for the Left Subtree Range (K_i ... K_{l-1})
                key_index: int = int(self.__root_matrix._get_value(i, l - 1))

                # Create new Node
                v = Node(
                    key=key_index,
                    value=str(values[key_index])
                )

                # Add to the Node we are currently processing as the Left Child
                u.left_child = v

                # Add it for further processing
                stack.append(tuple([v, i, l - 1]))

        # Stop Timer and Display Processing Time
        end_time = datetime.now()
        print(
            colored('Tree built in: ', 'green'),
            end_time - start_time,
            end='\n\n'
        )

        return root

    def preorder(
            self,
            root: Node,
            level: int,
            levels: Dict[int, List[Tuple[str]]]
    ) -> None:
        """Traverses Tree in Preorder fashion and stores Values to Dictionary
        based on their corresponding Level.

        Args:
            root (Node): Root of the Current Subtree
            level (int): Current Level
            levels (Dict[int, List[Tuple[str]]]): Node Values on each Level
        """

        # SOURCE: https://www.techiedelight.com/level-order-traversal-binary-tree/

        # Base case: Empty Tree
        if root == tuple([None]):
            return

        # Check if we already have values from given Level
        if level not in levels:

            # If not add new level with it's first Value
            levels[level] = [root.value[0]]

        else:
            # Otherwise update Values
            levels[level].append(root.value[0])

        # Recursively Search through the Left and Children of Current Node
        self.preorder(root.left_child, level + 1, levels)
        self.preorder(root.right_child, level + 1, levels)

    def __display_tree(self) -> None:
        """Displays Optimal Binary Search Tree."""

        # Dictionary to store Values of each Level
        levels: Dict[int, List[Tuple[str]]] = {}

        # Traverse the Tree and save Node Values to Dictionary on the
        # corresponding Level
        self.preorder(self.__tree, 1, levels)

        # Display Values on each Level
        for i in range(1, len(levels) + 1):
            print(colored(f'Level {i}:', 'green'), levels[i], end='\n\n')

    def display_statistics(self) -> None:
        """Displays Basic Statistics of Optimal Binary Search Tree."""

        # Display the Tree itself
        self.__display_tree()

        # Display the Average / Optimal cost of Searching through every
        # element in Tree
        print(
            colored('Cost of Search in Optimal Binary Search Tree:', 'green'),
            self.__cost_matrix._get_value(1, len(self.__dictionary) - 1),
            end='\n\n'
        )

    def __recursive_search(
        self,
        current_root: Node,
        query: str,
        number_of_comparisons: int
    ) -> Tuple[int, bool]:
        """Performs Recursive Search for Query in Optimal Binary Search Tree.

        Args:
            current_root (Node): Current Node
            query (str): Search Query
            number_of_comparisons (int): Number of Comparisons so far

        Returns:
            Tuple[int, bool]: Number of Comparisons and the outcome of Search
        """

        # Base Case : Empty Subtree (Not found in existing Tree)
        if current_root == tuple([None]):
            return number_of_comparisons, False

        # Get the Node Value
        root_value: str = current_root.value[0]

        # Check if it's a Match
        if root_value == query:
            number_of_comparisons += 1
            return number_of_comparisons, True

        # If it's not a Match got to either Left or Right Child
        if query < root_value:
            number_of_comparisons += 1
            return self.__recursive_search(
                current_root.left_child,
                query,
                number_of_comparisons
            )

        else:
            number_of_comparisons += 1
            return self.__recursive_search(
                current_root.right_child,
                query,
                number_of_comparisons
            )

    def __pocet_porovnani(self, query: str) -> Tuple[int, bool]:
        """Returns the Number of Comparisons and the outcome of Search for given
        query.

        Args:
            query (str): Search Query

        Returns:
            Tuple[int, bool]: Number of Comparisons and the outcome of Search
        """

        return self.__recursive_search(self.__tree, query, 0)

    def search(self, query: str) -> None:
        """Executes Search in Optimal Binary Search Tree.

        Args:
            query (str): Search Query
        """

        if len(query) > 0:

            number_of_comparisons, is_in_tree = self.__pocet_porovnani(
                query=query
            )

            if is_in_tree:
                print(
                    colored('\nSearch Results:', 'green'), query,
                    colored('is a Key.', 'green'),
                    colored('\nLocation : Level', 'green'),
                    number_of_comparisons,
                    colored('\nNumber of Comparisons:', 'green'),
                    number_of_comparisons,
                )
            else:
                print(
                    colored('\nSearch Results:', 'green'), query,
                    colored('is a Dummy Key.', 'green'),
                    colored('\nLocation : Level', 'green'),
                    number_of_comparisons + 1,
                    colored('\nNumber of Comparisons:', 'green'),
                    number_of_comparisons,
                )
