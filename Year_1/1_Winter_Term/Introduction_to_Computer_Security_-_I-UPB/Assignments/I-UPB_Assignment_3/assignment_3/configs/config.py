import os
import platform

# Platform related definitions
PLATFORM_OS = platform.system()
DELIMITER = '\\' if PLATFORM_OS == 'Windows' else '/'

# Directory definitions
ROOT_DIR = os.path.abspath('.')
