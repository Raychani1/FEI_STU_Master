from termcolor import colored

from data_structures.optimal_binary_search_tree import OptimalBinarySearchTree


if __name__ == '__main__':
    # Create a new Optimal Binary Search Tree
    optimal_binary_search_tree = OptimalBinarySearchTree(
        word_frequency_limit=50000
    )

    # Display basic statistics
    optimal_binary_search_tree.display_statistics()

    # Get User Input
    user_input: str = input(colored('Please enter Search Query: ', 'green'))

    # Perform Search until User enters 'exit()'
    while user_input != 'exit()':
        optimal_binary_search_tree.search(query=user_input)

        user_input = input(colored('\nPlease enter Search Query: ', 'green'))
