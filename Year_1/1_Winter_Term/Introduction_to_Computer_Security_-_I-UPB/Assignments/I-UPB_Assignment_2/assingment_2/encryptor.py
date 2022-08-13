import os
import pyminizip
from datetime import datetime
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from termcolor import colored
from assingment_2.configurations.config import DATA_DIR, DELIMITER


class Encryptor:

    def __init__(self, file: str, number_of_bytes: int) -> None:
        """Initializes all the necessary attributes for the Encryptor class.

        Args:
            file (str): Path to file which needs to be encrypted and decrypted
            number_of_bytes (int): Key length in bytes

        """

        # File to encrypt and decrypt
        self.__file = file

        # File name - which will be used for simplicity as password for ZIP file
        self.__file_name = self.__file.split(DELIMITER)[-1].split('.')[0]

        # Key length in bits
        self.__key_length = number_of_bytes

        # Generated key
        self.__key = self.__generate_key()

    def __generate_key(self) -> bytes:
        """Generates key and saves it to password protected ZIP file.

        Returns:
            bytes : Generated key

        """

        # Generate Key based on defined key length
        key = get_random_bytes(self.__key_length)

        # Write key to txt file
        with open(f'{DATA_DIR}{DELIMITER}key.txt', 'w') as key_file:
            key_file.writelines(str(key))

        # Create password protected ZIP file
        self.__create_password_protected_zip()

        return key

    def __create_password_protected_zip(self) -> None:
        """Creates password protected ZIP file"""

        # Source: https://www.geeksforgeeks.org/create-password-protected-zip-of-a-file-using-python/

        # Input file path
        input_file = f"{DATA_DIR}{DELIMITER}key.txt"

        # Prefix path
        prefix = None

        # Output zip file path
        output_file = f"{DATA_DIR}{DELIMITER}key.zip"

        # Set password value
        password = self.__file_name

        # Compress level
        com_lvl = 5

        # Compressing file
        pyminizip.compress(
            input_file,
            None,
            output_file,
            password,
            com_lvl
        )

        # Remove plain text key file
        os.remove(f'{DATA_DIR}{DELIMITER}key.txt')

        print(
            f'\n{colored("Your key is saved to: ", "green")}'
            f'{DATA_DIR}{DELIMITER}key.zip\n'
            f'The password is the argument passed at the beginning! \n'
        )

    def __encrypt(self):
        """Encrypts file."""

        # Start the timer
        start_time = datetime.now()

        # Create cipher
        cipher = AES.new(self.__key, AES.MODE_EAX)

        # Open file to encrypt
        with open(self.__file, 'rb') as f:
            # Read line of file
            contents = f.read()

            # Encrypt line
            ciphertext, tag = cipher.encrypt_and_digest(
                contents
            )

            # Write to output file
            file_out = open(
                f"{DATA_DIR}{DELIMITER}{self.__file_name}.enc.txt", "wb"
            )

            [file_out.write(x) for x in (cipher.nonce, tag, ciphertext)]
            file_out.close()

            del file_out

        # Stop the timer and display the time difference ( for encryption )
        end_time = datetime.now()

        print(
            f'{colored("1GB File Encrypted in: ", "green")}'
            f'{end_time - start_time}'
        )

    def __decrypt(self):
        """Decrypts encrypted file."""

        # Start the timer
        start_time = datetime.now()

        #
        file_in = open(f"{DATA_DIR}{DELIMITER}{self.__file_name}.enc.txt", "rb")
        nonce, tag, ciphertext = [file_in.read(x) for x in (16, 16, -1)]

        # Create cipher
        cipher = AES.new(self.__key, AES.MODE_EAX, nonce)

        try:
            # Decrypt encrypted data and check for signs of tampering
            data = cipher.decrypt_and_verify(ciphertext, tag)

            # Stop the timer and display the time difference ( for decryption )
            end_time = datetime.now()
            print(
                f'{colored("1GB File Decrypted in: ", "green")}'
                f'{end_time - start_time}'
            )

            # Start the timer
            start_time = datetime.now()

            # Save the data to output file
            with open(f'{DATA_DIR}{DELIMITER}decrypted.txt', 'w') as output:
                output.writelines(str(data, 'utf-8'))

            # Stop the timer and display the time difference
            # ( for saving decrypted data )
            end_time = datetime.now()

            print(
                f'{colored("1GB Decrypted File Saved in: ", "green")}'
                f'{end_time - start_time}'
            )

        except ValueError:
            print(
                colored('Looks like somebody tampered with the file!', 'yellow')
            )

    def run(self):
        """Run both encrypt and decrypt on the file."""

        self.__encrypt()
        self.__decrypt()
