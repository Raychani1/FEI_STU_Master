import os
import sys
import platform
from executor.neural_network_project import NeuralNetworkProject

if __name__ == '__main__':
    # Activate colors in Windows Command Line and Powershell
    if platform.system == 'Windows':
        os.system('color')

    neural_network_project = NeuralNetworkProject(sys.argv[1]).run()
