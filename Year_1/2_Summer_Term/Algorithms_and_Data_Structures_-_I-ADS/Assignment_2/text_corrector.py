import math
import os
from datetime import datetime
from typing import List, Optional, Tuple, Union
from unicodedata import normalize

from termcolor import colored

from correction_types import CorrectionType


class TextCorrector:

    def __init__(
            self,
            mode: Optional[str] = 'original',
            compare_directory: Optional[str] = os.path.join(
                os.getcwd(),
                'data',
                'compare'
            ),
            input_directory: Optional[str] = os.path.join(
                os.getcwd(),
                'data',
                'input'
            ),
            output_directory: Optional[str] = os.path.join(
                os.getcwd(),
                'data',
                'output'
            )
    ) -> None:
        """Initializes the Text Corrector Class.

        Args:
            mode (Optional[str]): Dictionary Based Execution Mode. Defaults to
             'original'.
            compare_directory (Optional[str]): Compare directory. Defaults to
             'project-folder/data/compare'.
            input_directory (Optional[str]): Input directory. Defaults to
             'project-folder/data/input'.
            output_directory (Optional[str]): Output directory. Defaults to
             'project-folder/data/output'.
        """

        # Load Dictionary
        self.__dictionary: List[str] = self.__read_file_to_list(
            filepath=os.path.join(
                os.getcwd(),
                'data',
                'dictionary.txt' if mode == 'original' else 'dictionary_2.txt'
            )
        )

        # Set Dictionary Based Execution Mode
        self.__mode = mode.lower().capitalize()

        # Inform the user about Execution Mode
        print(
            colored(f'Using {mode.lower().capitalize()} Dictionary', 'green'),
            end='\n\n\n'
        )

        # Set the compare Directory
        self.__compare_directory: str = compare_directory

        # Set the input Directory
        self.__input_directory: str = input_directory

        # Set the output Directory
        self.__output_directory: str = output_directory

    @staticmethod
    def __read_file_to_list(filepath: str) -> List[str]:
        """Loads file to List of Strings.

        Args:
            filepath (str): Input Filepath

        Returns:
            List[str]: File loaded to List of Strings
        """

        words: List[str] = []

        with open(filepath, encoding='cp1251') as input_file:
            for line in input_file:
                # Split lines to words and clean them
                words.extend(line.lower().strip().split())

        return words

    @staticmethod
    def __longest_common_substring(word1: str, word2: str) -> int:
        """Calculates the length of the Longest Common Substring.

        Args:
            word1 (str): Word #1 to compare
            word2 (str): Word #2 to compare

        Returns:
            int: Length of the Longest Common Substring
        """

        # SOURCE: 
        # https://www.youtube.com/watch?v=BysNXJHzCEs
        # https://github.com/mission-peace/interview/blob/master/src/com/interview/dynamic/LongestCommonSubstring.java

        # EXAMPLE: 

        #  [       c  a  r  r  i  b  e  a  n
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #    a [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        #    r [0, 0, 0, 2, 1, 0, 0, 0, 0, 0],
        #    r [0, 0, 0, 1, 3, 0, 0, 0, 0, 0],
        #    i [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        #    b [0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
        #    e [0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
        #    a [0, 0, 1, 0, 0, 0, 0, 0, 7, 0],
        #    n [0, 0, 0, 0, 0, 0, 0, 0, 0, 8]
        # ]

        # Fetch the length of words
        word1_length, word2_length = len(word1), len(word2)

        # Create table which will store common substring lengths
        table: List[List[int]] = [
            [0 for _ in range(word2_length + 1)]
            for _ in range(word1_length + 1)
        ]

        # Length of the longest common substring
        max_length: int = 0

        for i in range(1, word1_length + 1):
            for j in range(1, word2_length + 1):

                # If we've found a match increase the length of substring we
                # already have which is saved in the diagonally up left element
                if word1[i - 1] == word2[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1

                    # If we've found a new longest common substring we update
                    # this variable
                    if table[i][j] > max_length:
                        max_length = table[i][j]

        return max_length

    def run_longest_common_substring_correction(self, input_file: str) -> None:
        """Performs the Longest Common Substring Correction on given Input File.

        Args:
            input_file (str): Input File to process
        """

        # Timer for Longest Common Substring Correction
        start_time = datetime.now()

        # Fetch File name from Path
        file_name: str = os.path.basename(input_file)

        # Container for new Data
        corrected_data: List[str] = []

        # Load file to List of Strings
        input_data: List[str] = self.__read_file_to_list(filepath=input_file)

        # For each word in Input File
        for data in input_data:

            # If the word is in the dictionary, it is a correct one
            # However if it is not we need to check it for errors
            if data not in self.__dictionary:
                longest_common_substring_length: int = -1
                longest_common_substring: str = ''

                # We compare it to every word in our dictionary
                for word in self.__dictionary:

                    # Get the length of the Longest Common Substring for the
                    # misspelled word
                    lcs_value: int = self.__longest_common_substring(
                        word1=data, word2=word
                    )

                    # If the misspelled word is completely found in Dictionary
                    # word we've found a match
                    if lcs_value == len(data):
                        longest_common_substring = word
                        break

                    # If we've found a new longest matching substring we update
                    # the values
                    if lcs_value > longest_common_substring_length:
                        longest_common_substring_length = lcs_value
                        longest_common_substring = word

                # Update Data
                data = longest_common_substring

            # Normalize the word
            data = normalize('NFD', data).encode(
                'ascii', 'ignore'
            ).decode("utf-8")

            # Save the word to new Data
            corrected_data.append(data)

        # Write all the new data to Output File
        with open(
                os.path.join(
                    self.__output_directory,
                    f'{self.__mode}_LCSubstring_corrected_{file_name}'
                ),
                'w'
        ) as output_file:
            output_file.write(' '.join(corrected_data))

        # Stop Timer and Display Processing Time
        end_time = datetime.now()
        print(
            colored(
                f'Longest Common Substring Correction for {file_name} Completed'
                f' in: ',
                'green'
            ),
            f"{end_time - start_time}",
            end='\n\n'
        )

    @staticmethod
    def __longest_common_subsequence(word1: str, word2: str) -> int:
        """Calculates the length of the Longest Common Subsequence.

        Args:
            word1 (str): Word #1 to compare
            word2 (str): Word #2 to compare

        Returns:
            int: Length of the Longest Common Subsequence
        """

        # SOURCE: 
        # https://www.youtube.com/watch?v=NnD96abizww
        # https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/?ref=lbp
        # Program #3 (Modified)

        # EXAMPLE: 

        #  [       c  a  r  r  i  b  e  a  n
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #    a [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        #    r [0, 0, 1, 2, 2, 2, 2, 2, 2, 2],
        #    r [0, 0, 1, 2, 3, 3, 3, 3, 3, 3],
        #    i [0, 0, 1, 2, 3, 4, 4, 4, 4, 4],
        #    b [0, 0, 1, 2, 3, 4, 5, 5, 5, 5],
        #    e [0, 0, 1, 2, 3, 4, 5, 6, 6, 6],
        #    a [0, 0, 1, 2, 3, 4, 5, 6, 7, 7],
        #    n [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        # ]

        # Fetch the length of words
        word1_length, word2_length = len(word1), len(word2)

        # Create table which will store common subsequence lengths
        table: List[List[int]] = [
            [0 for _ in range(word2_length + 1)]
            for _ in range(word1_length + 1)
        ]

        # Next we will fill the table in bottom up fashion
        # Note: table[i][j] contains length of LCS of word1[0..i-1] and
        #       word2[0..j-1]

        for i in range(1, word1_length + 1):
            for j in range(1, word2_length + 1):

                # If the letters we are checking are the same we increase the
                # length of subsequence we already have which is saved in the
                # diagonally up left element
                if word1[i - 1] == word2[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1

                # However, if they don't match we need to determine the longest
                # subsequence from neighbouring elements
                else:
                    table[i][j] = max(table[i - 1][j], table[i][j - 1])

        # table[word1_length][word2_length] contains the length of LCS of 
        # word1[0..word1_length-1] & word2[0..word2_length-1]
        return table[word1_length][word2_length]

    def run_longest_common_subsequence_correction(
            self,
            input_file: str
    ) -> None:
        """Performs the Longest Common Subsequence Correction on given Input File.

        Args:
            input_file (str): Input File to process
        """

        # Timer for Longest Common Subsequence Correction
        start_time = datetime.now()

        # Fetch File name from Path
        file_name: str = os.path.basename(input_file)

        # Container for new Data
        corrected_data: List[str] = []

        # Load file to List of Strings
        input_data: List[str] = self.__read_file_to_list(filepath=input_file)

        # For each word in Input File
        for data in input_data:

            # If the word is in the dictionary, it is a correct one
            # However if it is not we need to check it for errors
            if data not in self.__dictionary:
                longest_common_subsequence_length: int = -1
                longest_common_subsequence: str = ''

                # We compare it to every word in our dictionary
                for word in self.__dictionary:

                    # Get the length of the longest common subsequence for the
                    # misspelled word
                    lcs_value: int = self.__longest_common_subsequence(
                        word1=data, word2=word
                    )

                    # If the misspelled word is completely found in Dictionary
                    # word we've found a match
                    if lcs_value == len(data):
                        longest_common_subsequence = word
                        break

                    # If we've found a new longest matching subsequence we
                    # update the values
                    if lcs_value > longest_common_subsequence_length:
                        longest_common_subsequence_length = lcs_value
                        longest_common_subsequence = word

                # Update Data
                data = longest_common_subsequence

            # Normalize the word
            data = normalize('NFD', data).encode(
                'ascii', 'ignore'
            ).decode("utf-8")

            # Save the word to new Data
            corrected_data.append(data)

        with open(
                os.path.join(
                    self.__output_directory,
                    f'{self.__mode}_LCSubsequence_corrected_{file_name}'
                ),
                'w'
        ) as output_file:
            output_file.write(' '.join(corrected_data))

        # Stop Timer and Display Processing Time
        end_time = datetime.now()
        print(
            colored(
                f'Longest Common Subsequence Correction for {file_name} '
                f'Completed in: ',
                'green'
            ),
            f"{end_time - start_time}",
            end='\n\n'
        )

    @staticmethod
    def __edit_distance(word1: str, word2: str) -> int:
        """Calculates the Minimal Edit Distance.

        Args:
            word1 (str): Word #1 to compare
            word2 (str): Word #2 to compare

        Returns:
            int: Minimal Edit Distance
        """

        # SOURCE: 
        # https://www.youtube.com/watch?v=We3YDTzNXEk
        # https://www.geeksforgeeks.org/edit-distance-dp-5/?ref=gcse Program #2

        # EXAMPLE: 

        #  [       c  a  r  r  i  b  e  a  n
        #      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        #    a [1, 1, 1, 2, 3, 4, 5, 6, 7, 8],
        #    r [2, 2, 2, 1, 2, 3, 4, 5, 6, 7],
        #    r [3, 3, 3, 2, 1, 2, 3, 4, 5, 6],
        #    i [4, 4, 4, 3, 2, 1, 2, 3, 4, 5],
        #    b [5, 5, 5, 4, 3, 2, 1, 2, 3, 4],
        #    e [6, 6, 6, 5, 4, 3, 2, 1, 2, 3],
        #    a [7, 7, 6, 6, 5, 4, 3, 2, 1, 2],
        #    n [8, 8, 7, 7, 6, 5, 4, 3, 2, 1]
        # ]

        # Fetch the length of words
        word1_length, word2_length = len(word1), len(word2)

        # Create a table to memoize result
        # of previous computations
        table = [[0 for _ in range(word2_length + 1)]
                 for _ in range(word1_length + 1)]

        # Fill table in bottom up manner
        for i in range(word1_length + 1):
            for j in range(word2_length + 1):

                # If first string is empty, only option is to
                # insert all characters of second string
                if i == 0:
                    table[i][j] = j

                # If second string is empty, only option is to
                # remove all characters of second string
                elif j == 0:
                    table[i][j] = i

                # If last characters are same, ignore last char
                # and recur for remaining string
                elif word1[i - 1] == word2[j - 1]:
                    table[i][j] = table[i - 1][j - 1]

                # If last character are different, consider all
                # possibilities and find minimum
                else:
                    table[i][j] = 1 + min(
                        table[i][j - 1],  # Insert / Add
                        table[i - 1][j],  # Remove / Delete
                        table[i - 1][j - 1]  # Replace / Edit
                    )

        return table[word1_length][word2_length]

    def run_edit_distance_correction(self, input_file: str) -> None:
        """Performs the Edit Distance Correction on given Input File.

        Args:
            input_file (str): Input File to process
        """

        # Timer for Edit Distance Correction
        start_time = datetime.now()

        # Fetch File name from Path
        file_name: str = os.path.basename(input_file)

        # Container for new Data
        corrected_data: List[str] = []

        # Load file to List of Strings
        input_data: List[str] = self.__read_file_to_list(filepath=input_file)

        # For each word in Input File
        for data in input_data:

            # If the word is in the dictionary, it is a correct one
            # However if it is not we need to check it for errors
            if data not in self.__dictionary:
                edit_distance: Union[float, int] = math.inf
                lowest_edit_distance: str = ''

                for word in self.__dictionary:

                    # Get the Edit Distance for the misspelled word
                    ed_value: int = self.__edit_distance(
                        word1=data, word2=word
                    )

                    # If the misspelled word is completely found in Dictionary
                    # word we've found a match
                    if ed_value == 0:
                        lowest_edit_distance = word
                        break

                    # If we've found a new lowest matching edit distance we
                    # update the values
                    if ed_value < edit_distance:
                        edit_distance = ed_value
                        lowest_edit_distance = word

                # Update Data
                data = lowest_edit_distance

            # Normalize the word
            data = normalize('NFD', data).encode(
                'ascii', 'ignore'
            ).decode("utf-8")

            # Save the word to new Data
            corrected_data.append(data)

        # Write all the new data to Output File
        with open(
                os.path.join(
                    self.__output_directory,
                    f'{self.__mode}_Edit_Distance_corrected_{file_name}'
                ),
                'w'
        ) as output_file:
            output_file.write(' '.join(corrected_data))

        # Stop Timer and Display Processing Time
        end_time = datetime.now()
        print(
            colored(
                f'Edit Distance Correction for {file_name} Completed in: ',
                'green'
            ),
            f"{end_time - start_time}",
            end='\n\n'
        )

    def compare(
            self,
            correction_type: str,
            correct_file: str,
            corrected_file: str
    ) -> None:
        """Compares two files and displays matching statistics.

        Args:
            correction_type (str): Type of Correction for Displaying Purposes
            correct_file (str): Correct Filepath
            corrected_file (str): Corrected Filepath
        """

        # Fetch Corrected File name from Path
        corrected_file_name: str = os.path.basename(corrected_file)

        # Load Correct File to List of Strings
        correct_file_data: List[str] = self.__read_file_to_list(
            filepath=correct_file
        )

        # Load Corrected File to List of Strings
        corrected_file_data: List[str] = self.__read_file_to_list(
            filepath=corrected_file
        )

        # If their length is not matching inform the user
        if len(correct_file_data) != len(corrected_file_data):
            print('Files are not the same length!')
            return

        # Match counter
        matches: int = 0

        # For each word in the corrected file
        for index in range(len(corrected_file_data)):

            # Increase the counter for every match
            if corrected_file_data[index] == correct_file_data[index]:
                matches += 1

        # Display Correction Match Statistics
        print(
            colored(
                f'{correction_type} Correction Matches for '
                f'{corrected_file_name}: ',
                'green'
            ),
            f"{matches}/{len(correct_file_data)} = "
            f"{matches / len(correct_file_data)} %",
            end='\n\n'
        )

    def run_correction_pipeline(self) -> None:
        """Runs Correction Pipeline on every input file."""

        # Initialize Correction Types
        corrections: List[Tuple[str, str]] = [
            (CorrectionType.LCSUBSTR, 'LCSubstring_corrected_input_'),
            (CorrectionType.LCSUBSEQ, 'LCSubsequence_corrected_input_'),
            (CorrectionType.EDITDIST, 'Edit_Distance_corrected_input_')
        ]

        # For each Input File
        for input_file in sorted(os.listdir(self.__input_directory)):
            input_file: str = os.path.join(self.__input_directory, input_file)
            input_num: str = input_file.split('_')[-1]

            # Run the Longest Common Substring Correction
            self.run_longest_common_substring_correction(
                input_file=input_file
            )

            # Run the Longest Common Subsequence Correction
            self.run_longest_common_subsequence_correction(
                input_file=input_file
            )

            # Run the Edit Distance Correction
            self.run_edit_distance_correction(
                input_file=input_file
            )

            # For each matching Correct File
            for compare_file in os.listdir(self.__compare_directory):
                compare_num: str = compare_file.split('_')[-1]

                if compare_num == input_num:
                    print('\n')

                    # For each Correction Type
                    for correction in corrections:
                        # Compare Matching Files
                        self.compare(
                            correction_type=correction[0],
                            correct_file=os.path.join(
                                self.__compare_directory,
                                compare_file
                            ),
                            corrected_file=os.path.join(
                                self.__output_directory,
                                f'{self.__mode}_{correction[1]}{compare_num}'
                            )
                        )

                    break

            print('\n\n')
