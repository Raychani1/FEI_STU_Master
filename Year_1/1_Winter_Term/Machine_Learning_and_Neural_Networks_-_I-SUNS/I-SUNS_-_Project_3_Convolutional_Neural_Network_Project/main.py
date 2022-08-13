import os
import platform
from executor.convolutional_neural_network_project import \
    ConvolutionalNeuralNetworkProject

if __name__ == '__main__':
    # Activate colors in Windows Command Line and Powershell
    if platform.system == 'Windows':
        os.system('color')

    ConvolutionalNeuralNetworkProject().run()
