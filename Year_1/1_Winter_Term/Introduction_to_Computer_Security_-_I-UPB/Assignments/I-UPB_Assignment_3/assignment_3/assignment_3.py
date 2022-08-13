from termcolor import colored
from assignment_3.configs.config import *
from assignment_3.encryptor import Encryptor


class Assignment3:

    def __init__(self, key_folder_path: str) -> None:
        """Initialize Assignment3 Class.

        Args:
            key_folder_path (str) : Path to the Key Store Folder

        """

        # Create instance of Encryptor
        self.__encryptor = Encryptor(folder_path=key_folder_path)

        # Load Key Store Folder
        self.__key_folder_path = key_folder_path

        # Run the application
        self.__run_application()

    def __run_application(self) -> None:
        """Welcome User and read Input."""

        # Display welcome message
        print(colored('Welcome to Encryptor 2.0!\n', 'green'))
        print(f"For help type {colored('help', 'green')} and hit Enter\n\n")

        # Read User input
        self.__deal_with_user_input()

    def __deal_with_user_input(self) -> None:
        """Reads User Input and execute commands."""

        # Default Starting Value
        user_input = list()
        user_input.append('')

        # The program will run while the user does not want to exit
        while user_input[0].lower() != 'exit' and len(user_input) != 0:
            try:
                # Read User Input
                user_input = input(colored('[Encryptor 2.0] $ ', 'green'))\
                    .split()

                # The User wants to see the help menu
                if user_input[0].lower() == 'help':

                    # Display help menu
                    self.__display_help()

                # The User wants to encrypt a file
                elif user_input[0].lower() == 'enc':

                    # First we need to check if the file exists
                    if not os.path.exists(user_input[1]):

                        # If the file does not exit we inform the User
                        print(colored('File does not exist!\n', 'red'))

                    else:
                        # Otherwise we start the encryption process
                        self.__encryptor.encrypt(user_input[1])

                # The User wants to decrypt a file
                elif user_input[0].lower() == 'dec':

                    # We need to get the file to be able to find the Private RSA
                    # Key
                    file_name = user_input[1].split(DELIMITER)[-1].split('.')[0]

                    # We need to check if the file exists
                    if not os.path.exists(user_input[1]):
                        print(colored('File does not exist!\n', 'red'))
                    else:
                        # We need to check if the Private RSA Key is in the Key
                        # Store Folder
                        if not os.path.exists(f'{self.__key_folder_path}'
                                              f'{DELIMITER}{file_name}'
                                              f'_private.bin'):

                            # Inform the User about the missing Private RSA Key
                            print(colored(f'Private Encrypted RSA Key '
                                          f'{file_name}_private.bin not found '
                                          f'in {self.__key_folder_path} '
                                          f'folder!\n',
                                          'red'))
                        else:
                            # If everything is okay decrypt the file
                            self.__encryptor.decrypt(user_input[1])

                # The User wants to compare two files
                elif user_input[0] == 'diff':

                    # Call the built in File Compare Methods
                    os.system(f"{'diff' if PLATFORM_OS == 'Linux' else 'fc'}"
                              f" {user_input[1]} {user_input[2]}")

            # If something went wrong
            except IndexError:
                # Inform the User
                print(colored('Missing argument(s)!\n', 'red'))

                # Clear previous value
                user_input = list()
                user_input.append('')
        
        print()

    @staticmethod
    def __display_help() -> None:
        """Display Help Menu."""

        # Display Help Row
        print(colored('help', 'green'),
              ' - Displays help regarding application controls '
              '( You are viewing this right now )\n')

        # Display Encryption Row
        print(colored('enc [file_path]', 'green'),
              '- Create RSA Key pair and encrypt selected file\n')

        # Display Decryption Row
        print(colored('dec [encrypted_file_path]', 'green'),
              '- Decrypt encrypted file using the RSA Key Pair\n')

        # Display Differences Row
        print(colored('diff [file_path_1] [file_path_2]', 'green'),
              '- Compare two files\n'),

        # Display Exit Row
        print(colored('exit', 'green'),
              '- Exit Encryptor 2.0\n')
