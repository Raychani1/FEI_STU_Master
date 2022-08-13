import os
from datetime import datetime
from termcolor import colored
from assingment_2.configurations.config import PLATFORM_OS, DELIMITER, DATA_DIR

    
def generate_file(file_name: str) -> None:
    """Generate a text ( .txt ) file with the given name.

    Args:
        file_name (str): Name of the new text file

    """

    size = 1024 * 1024 * 1024  # 1GB

    if PLATFORM_OS == 'Linux':

        start_time = datetime.now()

        os.system(
            f'base64 /dev/urandom 2>/dev/null | head -c {size} '
            f'> {DATA_DIR}{DELIMITER}{file_name}.txt'
        )

        end_time = datetime.now()
        print(
            f'{colored("1GB File Generated in: ", "green")}'
            f'{end_time - start_time}'
        )
