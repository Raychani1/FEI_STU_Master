# MSUS - Project 1

## About The Project
The main objective of this Project is to model / implement a process-driven application consisting of a single process. 

## Requirements
The process must meet all of the following conditions:
* The process must contain at least 1 `TaskRef` data field ("static" is sufficient, i.e. it has an initial value set and its value does not need to change during the process run).
* The process must contain at least one data field of type `enumeration`, or `emuration_map`, or `multichoice`, or `multichoice_map`.
* The process must start with just one location with just one token (to facilitate both modeling and testing).
* The process must contain at least 10 tasks (executable transitions, i.e., at least 10 L1 liveness transitions).
* The process must contain at least one "decision" (i.e., at least one XOR split, .i.e., a construct where two or more transitions are executable at the same moment and the system/user must decide which one to execute).
* The execution of a decision (i.e., the execution of one transition in a decision) must be automated by actions (cf. builder -> action edit -> functions -> async run with execution of task).
* The process must contain the actions by which it is changed:
    - Changing the behavior of a data reference (e.g., hiding or unhiding it, or changing it to an editable input in a form)
    - Setting a value (e.g. calculating a value, composing text, etc.)
    - Setting options (setting options for `enumeration` / `enumeration_map`, or `multichoice` / `multichoice_map` data field) 

## Finished Project
[![Project 1][project_1]](#)

<!-- MARKDOWN LINKS & IMAGES -->
[project_1]: https://raw.githubusercontent.com/Raychani1/raychani1.github.io/main/projects/petriflow/projects/documentation/I-MSUS_--_Project_1_-_AIS_University_Assignment_Ladislav_Rajcsanyi_97914_2.png