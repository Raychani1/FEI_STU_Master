from typing import List, Tuple


class SATSolver:

    def __init__(self) -> None:
        """Initializes the SAT Solver Class."""

        # SOURCE:
        # https://www.geeksforgeeks.org/2-satisfiability-2-sat-problem/amp/

        self.__MAX: int = 1000
        self.__adj: List[List[int]] = []
        self.__adj_inv: List[List[int]] = []
        self.__visited: List[bool] = [False] * self.__MAX
        self.__visited_inv: List[bool] = [False] * self.__MAX
        self.__s: List[int] = []
        self.__scc: List[int] = [0] * self.__MAX
        self.__counter: int = 1

    def __reset_values(self) -> None:
        """Resets variables once new file is being solved."""

        self.__MAX: int = 10000
        self.__adj: List[List[int]] = []
        self.__adj_inv: List[List[int]] = []

        for _ in range(self.__MAX):
            self.__adj.append([])
            self.__adj_inv.append([])

        self.__visited: List[bool] = [False] * self.__MAX
        self.__visited_inv: List[bool] = [False] * self.__MAX
        self.__s: List[int] = []
        self.__scc: List[int] = [0] * self.__MAX
        self.__counter: int = 1

    def __add_edges(self, start: int, end: int) -> None:
        """Adds edge to base Graph.

        Args:
            start (int): Edge Starting Point.
            end (int): Edge Ending Point.
        """

        # SOURCE:
        # https://www.geeksforgeeks.org/2-satisfiability-2-sat-problem/amp/

        self.__adj[start].append(end)

    def __add_edges_inverse(self, start: int, end: int) -> None:
        """Adds Edge to Inverted (Transposed) Graph.

        Args:
            start (int): Edge Starting Point.
            end (int): Edge Ending Point.
        """

        # SOURCE:
        # https://www.geeksforgeeks.org/2-satisfiability-2-sat-problem/amp/

        self.__adj_inv[end].append(start)

    def __dfs_first(self, vertex: int) -> None:
        """Computes finish time for each vertex.

        Args:
            vertex (int): Current Vertex.
        """

        # SOURCE:
        # https://www.geeksforgeeks.org/2-satisfiability-2-sat-problem/amp/

        # Ignore already visited Vertex
        if self.__visited[vertex]:
            return

        # Update current Vertex visit status on first visit
        self.__visited[vertex] = True

        # Visit adjacent Vertices of the current Vertex in the base Graph
        for element in self.__adj[vertex]:
            self.__dfs_first(element)

        # Once the current Vertex is processed save it to the stack
        self.__s.append(vertex)

    def __dfs_second(self, vertex: int) -> None:
        """Computes Strongly Connected Component Group value.

        Args:
            vertex (int): Current Vertex.
        """

        # SOURCE:
        # https://www.geeksforgeeks.org/2-satisfiability-2-sat-problem/amp/

        # Ignore already visited Vertex
        if self.__visited_inv[vertex]:
            return
        
        # Update current Vertex visit status on first visit
        self.__visited_inv[vertex] = True

        # Visit adjacent Vertices of the current Vertex in the Inverted 
        # (Transposed) Graph
        for element in self.__adj_inv[vertex]:
            self.__dfs_second(element)

        # Saves the number of Strongly Connected Component where the processed
        # Vertex and its adjacent Vertices belong        
        self.__scc[vertex] = self.__counter

    def __is_2_satisfiable(
        self, 
        num_of_var: int, 
        num_of_clausules: int,
        var1: List[int],
        var2: List[int]
    ) -> List[bool]:
        """Checks if 2 SAT Problem is satisfiable.

        Args:
            num_of_var (int): Number of Variables in 2 SAT Problem.
            num_of_clausules (int): Number of Clausules in 2 SAT Problem.
            var1 (List[int]): Left Side Operands.
            var2 (List[int]): Right Side Operands.

        Returns:
            List[bool]: Variable values if Problem is satisfiable.
        """

        # SOURCE:
        # https://www.geeksforgeeks.org/2-satisfiability-2-sat-problem/amp/

        # Variable to store SAT Problem Variable values
        variable_values: List[bool] = [False] * num_of_var

        # We iterate trough each clausule
        for i in range(num_of_clausules):

            # In each step we are building the Inverted (Transposed) Graph as 
            # well
            # The '+ num_of_var' represents the negation            

            # ( A OR B ) ==> ( ( (NOT A) => B ) AND ( (NOT B) => A ) )
            if var1[i] > 0 and var2[i] > 0:

                # ( (NOT A) => B )
                self.__add_edges(var1[i] + num_of_var, var2[i])
                self.__add_edges_inverse(var1[i] + num_of_var, var2[i])

                # ( (NOT B) => A )
                self.__add_edges(var2[i] + num_of_var, var1[i])
                self.__add_edges_inverse(var2[i] + num_of_var, var1[i])

            # ( A OR (NOT B) ) ==> ( ( (NOT A) => (NOT B) ) AND ( B => A ) )
            elif var1[i] > 0 and var2[i] < 0:

                # ( (NOT A) => (NOT B) )
                self.__add_edges(var1[i] + num_of_var, num_of_var - var2[i])
                self.__add_edges_inverse(
                    var1[i] + num_of_var, num_of_var - var2[i]
                )

                # ( B => A )
                self.__add_edges(-var2[i], var1[i])
                self.__add_edges_inverse(-var2[i], var1[i])

            # ( (NOT A) OR B ) ==> ( ( A => B) AND ( (NOT B) => (NOT A) ) )
            elif var1[i] < 0 and var2[i] > 0:

                # ( A => B)
                self.__add_edges(-var1[i], var2[i])
                self.__add_edges_inverse(-var1[i], var2[i])

                # ( (NOT B) => (NOT A) )
                self.__add_edges(var2[i] + num_of_var, num_of_var - var1[i])
                self.__add_edges_inverse(
                    var2[i] + num_of_var, num_of_var - var1[i]
                )

            # ( (NOT A) OR (NOT B) ) ==> ( ( A => (NOT B) ) AND ( B => (NOT A) ) )
            else:

                # ( A => (NOT B) ) 
                self.__add_edges(-var1[i], num_of_var - var2[i])
                self.__add_edges_inverse(-var1[i], num_of_var - var2[i])

                # ( B => (NOT A) )
                self.__add_edges(-var2[i], num_of_var - var1[i])
                self.__add_edges_inverse(-var2[i], num_of_var - var1[i])

        # We visit each Vertex in the Graph ( Base and Negated version of it )
        for i in range(1, 2*num_of_var + 1):
            if not self.__visited[i]:
                self.__dfs_first(i)

        # Strongly Connected Components Calculation
        # While we have Vertices on the Stack
        while len(self.__s) > 0:

            # We pop the top element from Stack
            top: int = self.__s.pop()

            # If we haven't processed this element yet
            if not self.__visited_inv[top]:

                # We run DFS on it
                self.__dfs_second(top)
                self.__counter += 1

        # For each Vertex we check to which Strongly Connected Component does
        # it belong
        for i in range(1, num_of_var + 1):

            # If the Base and the Negated Value of the Vertex belong to the 
            # same Strongly Connected Component (have the same value), that 
            # is a contradiction and the given 2 SAT Problem is unsatisfiable.
            if self.__scc[i] == self.__scc[i + num_of_var]:
                print('The given expression is unsatisfiable.', end='\n\n')

                return []
            else:
                # We determine the value of a given 2 SAT Problem Variable 
                # wether the Base Value belongs to a bigger Strongly Connected 
                # Component or not. Because weaker elements tend to have lower
                # group value.
                variable_values[i-1] = self.__scc[i] > self.__scc[i + num_of_var]

        print('The given expression is satisfiable.')

        return variable_values

    @staticmethod
    def __display_variables(variables: List[bool]) -> None:
        """Displays Variable Values.

        Args:
            variables (List[bool]): Variable Values to Display.
        """

        for i, value in enumerate(variables):
                print(f'X{i+1} = {value}')
        print()

    def __read_data(
        self, 
        file_path: str
    ) -> Tuple[int, int, List[int], List[int]]:
        """Loads data from file.

        Args:
            file_path (str): Input FIle Path

        Returns:
            Tuple[int, int, List[int], List[int]]: Number of Variables, Number 
            of Clausules, Left Side Operands, Right Side Operands
        """

        with open(file_path, 'r') as input_file:
            lines = input_file.readlines()

            num_of_var, num_of_clausules = list(
                map(int, lines[0].strip().split())
            )

            var1: List[int] = []
            var2: List[int] = []

            for line in lines[1:]:
                values: List[int] = list(map(int, line.strip().split()))

                var1.append(values[0])
                var2.append(values[1])

        return (num_of_var, num_of_clausules, var1, var2)

    def solve(self, file_path: str) -> None:
        """Runs SAT Solver.

        Args:
            file_path (str): Input File Path.
        """

        # Load Data from File
        num_of_var, num_of_clausules, var1, var2 = self.__read_data(file_path)

        # Prepare to process new File
        self.__reset_values()

        # Check if 2 SAT Problem is satisfiable
        vairable_values: List[bool] = self.__is_2_satisfiable(
            num_of_var, 
            num_of_clausules, 
            var1, 
            var2
        )

        # If we've got back values it means it is satisfiable and we need to 
        # display them
        if len(vairable_values) != 0:
            self.__display_variables(vairable_values)
