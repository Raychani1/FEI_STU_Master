import os
import platform
from pathlib import Path

# Platform related definitions
PLATFORM_OS = platform.system()
DELIMITER = '\\' if PLATFORM_OS == 'Windows' else '/'

# Directory definitions
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(ROOT_DIR, 'data')
