
from typing import List


class Graph:

    # SOURCE: https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/

    def __init__(self, num_of_verticies: int) -> None:
        """Initializes the Graph Class.

        Args:
            num_of_verticies (int): Number of Verticies in Graph.
        """

        self.__num_of_verticies = num_of_verticies
        self.__adj = [[] for _ in range(num_of_verticies)]

    def __dfs_util(
        self,
        temp: List[int],
        v: int,
        visited: List[bool]
    ) -> List[int]:

        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent to this vertex v
        for i in self.__adj[v]:
            if visited[i] == False:

                # Update the list
                temp = self.__dfs_util(temp, i, visited)

        return temp

    def add_edge(self, v: int, w: int) -> None:
        """Adds undirected edge to graph.

        Args:
            v (int): Start vertex.
            w (int): End vertex.
        """

        # Bidirectional edge = Undirected edge
        self.__adj[v].append(w)
        self.__adj[w].append(v)

    def get_connected_components(self) -> List[List[int]]:
        """Gets connected components from undirected graph.

        Returns:
            List[List[int]]: List of Connected Components.
        """

        # Vertex visit indicator
        visited = []

        # Connected components holder
        cc = []

        # Mark every vertex unvisited
        for _ in range(self.__num_of_verticies):
            visited.append(False)

        # For each vertex
        for v in range(self.__num_of_verticies):

            # If the given vertex is unvisited
            if visited[v] == False:

                # Visit it and all the connected verticies
                temp = []
                cc.append(self.__dfs_util(temp, v, visited))

        return cc
