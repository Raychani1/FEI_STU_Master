import sys
from assingment_2 import utility
from assingment_2.encryptor import Encryptor
from assingment_2.configurations.config import DATA_DIR, DELIMITER


def upb_assignment_2():
    """UPB Assignment 2 - Encryption and Decryption"""

    # Get the file name from the command line
    file_name = sys.argv[1]

    # Generate 1GB .txt file
    utility.generate_file(file_name)

    # Create Encryptor object
    encryptor = Encryptor(
        file=f'{DATA_DIR}{DELIMITER}{file_name}.txt',
        number_of_bytes=16
    )

    # Encrypt and Decrypt file
    encryptor.run()


if __name__ == '__main__':
    upb_assignment_2()
