class Node:

    def __init__(self, key: int, value: str) -> None:
        """Initializes the Node Class.

        Args:
            key (int): Key of Node
            value (str): Value of Node
        """

        self.key = key,
        self.value = value,
        self.left_child = None,
        self.right_child = None,
