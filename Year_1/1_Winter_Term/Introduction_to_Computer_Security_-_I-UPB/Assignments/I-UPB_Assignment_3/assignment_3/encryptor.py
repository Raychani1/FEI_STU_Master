import os
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.PublicKey.RSA import RsaKey
from Crypto.Random import get_random_bytes
from termcolor import colored

from assignment_3.configs.config import DELIMITER, ROOT_DIR


class Encryptor:

    def __init__(self, folder_path: str) -> None:
        """Initialize Encryptor Class.

        Args:
            folder_path (str) : Path to Key Store Folder

        """

        # Set Key Store Folder Path
        self.__folder = folder_path

        # Set Output Folder Path
        self.__output = os.path.join(ROOT_DIR, 'output')

    def __generate_public_rsa_key(self, key: RsaKey, name: str) -> None:
        """Generate Public RSA Key.

        Args:
            key (RsaKey) : Generated RSA Key Pair
            name (str) : File Name

        """

        # Source/Inspiration:
        # https://pycryptodome.readthedocs.io/en/latest/src/examples.html#generate-an-rsa-key
        # https://pycryptodome.readthedocs.io/en/latest/src/examples.html#generate-public-key-and-private-key

        # Export public RSA key
        public_key = key.publickey().export_key()

        # Save public RSA key to file
        file_out = open(f'{self.__folder}{DELIMITER}{name}.pem', "wb")
        file_out.write(public_key)
        file_out.close()

    def __generate_private_rsa_key(self, key: RsaKey, name: str) -> None:
        """Generate Private RSA Key.

        Args:
            key (RsaKey) : Generated RSA Key Pair
            name (str) : File Name

        """

        # Source/Inspiration:
        # https://pycryptodome.readthedocs.io/en/latest/src/examples.html#generate-an-rsa-key

        # Ask the user for passphrase to secure the Private RSA Key
        secret_code = input(
            colored('Please enter passphrase for Private Key Encryption: ',
                    'green')
        )

        # Export Private RSA Key
        encrypted_private_key = key.export_key(
            passphrase=secret_code,
            pkcs=8,
            protection="scryptAndAES128-CBC"
        )

        # Save Private RSA Key to File
        file_out = open(f'{self.__folder}{DELIMITER}{name}_private.bin', "wb")
        file_out.write(encrypted_private_key)
        file_out.close()

    def __generate_rsa_keys(self, name: str) -> None:
        """Generate RSA Key Pair.

        Args:
            name (str) : File Name

        """

        # Inform the user
        print(colored('Generating RSA Key Pair\n ', 'green'))

        # Generate new RSA key
        key: RsaKey = RSA.generate(2048)

        # Save private RSA key to file
        self.__generate_private_rsa_key(key=key, name=name)

        # Save public RSA key to file
        self.__generate_public_rsa_key(key=key, name=name)

    def encrypt(self, file_path: str) -> None:
        """Encrypt File.

        Args:
            file_path (str) : Path to File waiting for Encryption

        """

        # Get the file name
        file_name = file_path.split(DELIMITER)[-1].split('.')[0]

        # Get the file extension
        file_extension = file_path.split(DELIMITER)[-1].split('.')[-1]

        # Generate new RSA Key pair
        self.__generate_rsa_keys(name=file_name)

        # Read file which we want to encrypt
        file_to_encrypt = open(file_path, 'rb').read()

        # Create encrypted file
        file_out = open(f'{self.__output}{DELIMITER}'
                        f'{file_name}.enc.{file_extension}', "wb")

        # Read Public RSA Key from file
        public_rsa_key = RSA.import_key(open(f'{self.__folder}{DELIMITER}'
                                             f'{file_name}.pem').read())

        # Generate new Session Key
        session_key = get_random_bytes(16)

        # Encrypt the Session Key with the Public RSA Key
        cipher_rsa = PKCS1_OAEP.new(public_rsa_key)
        enc_session_key = cipher_rsa.encrypt(session_key)

        # Encrypt the data with the AES session key
        cipher_aes = AES.new(
            key=session_key,
            mode=AES.MODE_GCM,
            mac_len=16
        )
        ciphertext, tag = cipher_aes.encrypt_and_digest(file_to_encrypt)

        # Save encrypted data to output file
        [file_out.write(x) for x in
         (enc_session_key, cipher_aes.nonce, tag, ciphertext)]

        file_out.close()

        print()

    def decrypt(self, file_path: str) -> None:
        """Decrypt File.

        Args:
            file_path (str) : Path to File waiting for Decryption

        """

        # Get the file name
        file_name = file_path.split(DELIMITER)[-1].split('.')[0]

        # Get the file extension
        file_extension = file_path.split(DELIMITER)[-1].split('.')[-1]

        # Ask the user for passphrase to access the Private RSA Key
        secret_code = input(
            colored('Please enter passphrase for Private Key Decryption: ',
                    'green')
        )

        # Read the encoded session key
        encoded_key = open(f'{self.__folder}{DELIMITER}'
                           f'{file_name}_private.bin', "rb").read()

        # Read encrypted file
        file_in = open(f'{file_path}', "rb")

        try:
            # Try to access Private RSA Key with the Passphrase
            private_key = RSA.import_key(encoded_key, passphrase=secret_code)
        except ValueError:
            # If the user have entered the wrong passphrase trow an error
            # and inform the user
            print(colored("\nLooks like you've entered the wrong passphrase for"
                          " this Private RSA Key. Please try again!\n ",
                          'green'))

            # Exit this method
            return

        enc_session_key, nonce, tag, ciphertext = \
            [file_in.read(x) for x in
             (private_key.size_in_bytes(), 16, 16, -1)]

        try:

            # Decrypt the Encrypted Session Key with the Private RSA Key
            cipher_rsa = PKCS1_OAEP.new(private_key)
            session_key = cipher_rsa.decrypt(enc_session_key)

            # Decrypt the data with the AES session key
            cipher_aes = AES.new(session_key, AES.MODE_GCM, nonce)
            data = cipher_aes.decrypt_and_verify(ciphertext, tag)

            # Save the decrypted data to file
            with open(f'{self.__output}{DELIMITER}'
                      f'decrypted_{file_name}.{file_extension}', 'bw') \
                    as output_file:
                output_file.write(data)

        except ValueError:
            # If something went wrong because the file was tampered with
            print(colored("\nLooks like somebody tampered with the "
                          "encrypted file\n", 'green'))

            # Exit method
            return

        print()
