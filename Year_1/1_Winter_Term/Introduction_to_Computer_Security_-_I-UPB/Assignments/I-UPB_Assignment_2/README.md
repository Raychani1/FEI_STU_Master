# I-UBP_Assignment_2

# Run 

## Linux
Open up Terminal. Navigate to the Project folder. Run the project with a file name (without extensions).
```
./UBP_Assignment_2 [Your_file_name]
```

## Windows
Open up Command line (CMD). Navigate to the Project folder. Create a Virtual Environment (if you don't have one) with the following command:
```
python -m venv venv
```

Activate the Virtual Environment.
```
.\venv\Scripts\activate
```
Run [setup.py](#).
```
python setup.py
```

Navigate to the data folder
```
cd data
```

Enter your file name to the following command. Run the command, and wait for it to finish.
```
echo 'This is just a sample line appended to create a big file.. ' > [Your_file_name].txt
    for /L %i in (1,1,24) do type [Your_file_name].txt >> [Your_file_name].txt
```

Navigate back to the project folder
```
cd ..
```

Once the function is finished run the [main.py](#) script with the same file name (without extensions). Wait for it to finish. The saving process might take some time.
```
python main.py [Your_file_name]
```

