from assingment_2.configurations.config import *


def setup_environment(requirements: object) -> None:
    """Install all the required packages from the requirements file.

    Args:
        requirements (object): Requirements file

    """

    if PLATFORM_OS == 'Linux':
        os.system(
            f"pip --disable-pip-version-check install -r {requirements} | "
            f"grep -v 'already satisfied'"
        )

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


def create_data_directory():
    """Creates Data Directory to save files connected with dec/encryption."""

    if not os.path.exists(f'{ROOT_DIR}{DELIMITER}data'):
        os.system(f'mkdir data')


def setup():
    """Sets up the Virtual Environment and creates Data Directory."""

    setup_environment(f'{ROOT_DIR}{DELIMITER}requirements.txt')
    create_data_directory()


if __name__ == '__main__':
    setup()
