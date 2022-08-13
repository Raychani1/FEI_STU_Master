import sys
from assignment_3.configs.config import *


def setup_environment(requirements: str) -> None:
    """Install all the required packages from the requirements file.
    Args:
        requirements (object): Requirements file
    """

    # Install packages for Linux
    if PLATFORM_OS == 'Linux':
        os.system(
            f"pip --disable-pip-version-check install -r {requirements} | "
            f"grep -v 'already satisfied'"
        )

    # Install 
    elif PLATFORM_OS == 'Windows':
        with open(requirements, 'r') as requirements_file:
            lines = requirements_file.readlines()
            for requirement in lines:
                os.system(
                    f"pip --disable-pip-version-check install {requirement}"
                )

    print('\nInstalled Packages:')
    os.system("pip freeze")
    print()


def create_directory(folder: str):
    """Creates Data Directory to save files connected with dec/encryption."""

    # If the folder does not exist
    if not os.path.exists(folder):

        # Inform the user
        print(f'Creating folder {folder} , because it does not exist.')

        # Create folder recursively
        os.system(f"mkdir {'-p' if PLATFORM_OS == 'Linux' else ''} {folder}")


def setup():
    """Sets up the Virtual Environment and creates Data Directory."""

    # Setup virtual environment
    setup_environment(f'{ROOT_DIR}{DELIMITER}requirements.txt')

    # Create Key Store Folder
    create_directory(folder=sys.argv[1])

    # Create Output Folder
    create_directory(folder=sys.argv[2])


if __name__ == '__main__':
    setup()
