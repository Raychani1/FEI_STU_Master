# Set up Output Folder
$output_directory = Join-Path -Path $pwd -ChildPath "\output"

# Set up the Key Store Folder
if($args.Count -lt 1){
    $key_directory = Join-Path -Path $pwd -ChildPath "\keys"
    
} elseif ($args.Count -gt 1){
    Write-Host -ForegroundColor 'Red' 'Too many arguments!'
    exit 1
} else {
    $key_directory = $args[0]
}

$venv_directory = Join-Path -Path $pwd -ChildPath "\venv"

# Check if the Virtual Environment already exists
if (Test-Path -Path $venv_directory) {
    Write-Host -ForegroundColor 'Green' "Found existing Virtual Environment`n"
} else {
    Write-Host -ForegroundColor 'Green' "Creating Virtual Environment`n"
    python.exe -m venv venv
}

# Activate the Virtual Environment
Write-Host -ForegroundColor 'Green' "Activating Virtual Environment`n"
.\venv\Scripts\activate.bat

# Run the Setup script
Write-Host -ForegroundColor 'Green' "Running setup.py`n"
python.exe ./setup.py $key_directory $output_directory

# Run the Main script
Write-Host -ForegroundColor 'Green' "Running main.py`n"
Start-Sleep -s 3
Clear-Host
python.exe ./main.py $key_directory

# Deactivate the Virtual Environment
Write-Host -ForegroundColor 'Green' "Deactivating Virtual Environment`n"
.\venv\Scripts\deactivate.bat