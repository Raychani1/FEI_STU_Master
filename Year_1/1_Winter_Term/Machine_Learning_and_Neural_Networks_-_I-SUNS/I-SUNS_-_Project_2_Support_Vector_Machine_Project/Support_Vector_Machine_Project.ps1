$folders=@("configs", "data", "dataloader", "executor", "models", "ops", "output", "output/plots", "output/plots/bar_plots", "output/plots/choropleth", "output/plots/count_plots", "output/plots/decision_trees", "output/plots/heatmaps", "output/plots/histograms", "output/plots/neural_network", "output/plots/pair_plots", "output/plots/residuals", "output/plots/word_clouds", "output/processing_steps", "utils")
$files=@("support_vector_machine_project.py", "spotify_test.csv", "spotify_train.csv", "setup.py", "plotter.py", "neural_network.py", "grid_search_output.txt", "dataloader.py", "config.py", "support_vector_machine.py", "api_caller.py", "evaluator.py")

# Create Project Folders
function create_folders([string[]] $Folders) {
    foreach ($folder in $Folders){

        $project_folder = Join-Path -Path $pwd -ChildPath $folder

        if (!(Test-Path -Path $project_folder)) {
            Write-Host -ForegroundColor 'Green' "Creating Project Folder $folder`n"
            mkdir "$folder"
        }
    }      
}

# Move Source Files to Project Folders
function move_files_to_folder([string[]] $Files) {
    foreach ($file in $Files){
        $file_name = Join-Path -Path $pwd -ChildPath $file
        

        if(Test-Path -Path $file_name -PathType Leaf) {
            switch($file){
                {($file -eq "support_vector_machine_project.py")} {
                    Write-Host -ForegroundColor 'Green' "Moving $file to Project Folder executor `n"
                    $destination = Join-Path -Path $pwd -ChildPath "executor/$file"
                    break;
                }

                {($file -eq "spotify_test.csv") -or ($_ -eq "spotify_train.csv")} {
                    Write-Host -ForegroundColor 'Green' "Moving $file to Project Folder data `n"
                    $destination = Join-Path -Path $pwd -ChildPath "data/$file"
                    break;
                }

                {($_ -eq "config.py")} {
                    Write-Host -ForegroundColor 'Green' "Moving $file to Project Folder configs `n"
                    $destination = Join-Path -Path $pwd -ChildPath "configs/$file"
                    break;
                }

                {($file -eq "setup.py")} {
                    Write-Host -ForegroundColor 'Green' "Moving $file to Project Folder utils `n"
                    $destination = Join-Path -Path $pwd -ChildPath "utils/$file"
                    break;
                }

                {($file -eq "plotter.py") -or ($_ -eq "api_caller.py") -or ($_ -eq "evaluator.py")} {
                    Write-Host -ForegroundColor 'Green' "Moving $file to Project Folder ops `n"
                    $destination = Join-Path -Path $pwd -ChildPath "ops/$file"
                    break;
                }

                {($file -eq "neural_network.py") -or ($_ -eq "support_vector_machine.py")} {
                    Write-Host -ForegroundColor 'Green' "Moving $file to Project Folder models `n"
                    $destination = Join-Path -Path $pwd -ChildPath "models/$file"
                    break;
                }

                {($file -eq "grid_search_output.txt")} {
                    Write-Host -ForegroundColor 'Green' "Moving $file to Project Folder output `n"
                    $destination = Join-Path -Path $pwd -ChildPath "output/$file"
                    break;
                }

                {($file -eq "dataloader.py")} {
                    Write-Host -ForegroundColor 'Green' "Moving $file to Project Folder dataloader `n"
                    $destination = Join-Path -Path $pwd -ChildPath "dataloader/$file"
                    break;
                }
            }
            Move-Item -Path $file_name -Destination $destination;
        }
    }
}

# Check for existence of Virtual Environment
function check_for_virtual_environment {
    $venv_directory = Join-Path -Path $pwd -ChildPath "\venv"

    # Check if the Virtual Environment already exists
    if (Test-Path -Path $venv_directory) {
        Write-Host -ForegroundColor 'Green' "Found existing Virtual Environment`n"
    } else {
        Write-Host -ForegroundColor 'Green' "Creating Virtual Environment`n"
        python.exe -m venv venv
    }
}

# Activate Virtual Environment
function activate_virtual_environment {
    Write-Host -ForegroundColor 'Green' "Activating Virtual Environment`n"
    .\venv\Scripts\Activate.ps1
}

# Get current Virtual Environment name
function get_virtual_environment {
    Write-Host -ForegroundColor 'Green' "Current Virtual Environment`n"
    Write-Host "$env:VIRTUAL_ENV`n"
}

# Run the setup.py script
function run_setup {
    Write-Host -ForegroundColor 'Green' "Running setup.py`n"
    python.exe ./utils/setup.py $pwd
}

# Run the whole project
function run($arguments) {
    if ( $arguments.Count -eq 0 ){
        create_folders($folders)
        move_files_to_folder($files)
        check_for_virtual_environment
        activate_virtual_environment
        get_virtual_environment
        run_setup
    } else {
        Write-Host -ForegroundColor 'Red' "Too many arguments!`n"
    }
}

run($args)
