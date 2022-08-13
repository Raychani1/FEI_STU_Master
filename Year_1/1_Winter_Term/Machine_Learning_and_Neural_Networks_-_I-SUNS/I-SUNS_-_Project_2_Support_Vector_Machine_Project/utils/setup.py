import os
import sys
import platform


def setup_environment(requirements: str) -> None:
    """Install all the required packages from the requirements file.
    Args:
        requirements (str): Requirements File Path

    """

    # Get current Operating System
    platform_os = platform.system()

    # Install packages for Linux
    if platform_os == 'Linux':
        os.system(
            f"pip --disable-pip-version-check install -r {requirements} | "
            f"grep -v 'already satisfied'"
        )

    # Install packages for Windows
    elif platform_os == 'Windows':
        with open(requirements, 'r') as requirements_file:
            lines = requirements_file.readlines()
            for requirement in lines:
                os.system(
                    f"pip --disable-pip-version-check install {requirement}"
                )

    print('\nInstalled Packages:')
    os.system("pip freeze")
    print()


def setup():
    """Sets up the Virtual Environment and creates Data Directory."""

    # Setup virtual environment
    setup_environment(f'{sys.argv[1]}/requirements.txt')


if __name__ == '__main__':
    setup()
