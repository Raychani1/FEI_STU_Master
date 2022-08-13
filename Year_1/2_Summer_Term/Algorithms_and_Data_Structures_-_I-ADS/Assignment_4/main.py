import os

from termcolor import colored 

from assignment_4 import Assignment4


if __name__ == '__main__':
    data_dir: str = os.path.join(
        os.getcwd(), 
        'data',  
        'input',           
    )

    for file in sorted(os.listdir(data_dir)):
        input_file: str = os.path.join(data_dir, file)
        
        print(colored('File:', 'green'), file)

        assignment4: Assignment4 = Assignment4(
            input_file = input_file
        )

        assignment4()

        del assignment4
