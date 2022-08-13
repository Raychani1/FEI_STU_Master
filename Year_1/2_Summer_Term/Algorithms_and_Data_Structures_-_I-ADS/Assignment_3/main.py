import os

from sat_solver import SATSolver


if __name__ == '__main__':
    sat_solver: SATSolver = SATSolver()

    for file in sorted(os.listdir(os.path.join(os.getcwd(), 'data'))):
        sat_solver.solve(os.path.join(os.getcwd(), 'data', file))
