import os
import sys
from assignment_3.configs.config import PLATFORM_OS
from assignment_3.assignment_3 import Assignment3

if __name__ == '__main__':
    # Activate colors in Windows Command Line and Powershell
    if PLATFORM_OS == 'Windows':
        os.system('color')

    # Run the Project
    assignment_3 = Assignment3(sys.argv[1])

