# MSUS - Project 2

## About The Project
The main objective of this Project is to model / implement a process-driven application consisting of multiple processes.

## Requirements
The submitted solution must consist of at least 3 processes.

In order to be recognized as developing a theme, the processes must collectively meet all of the following conditions:
* Contain at least 1 `TaskRef` data field that are dynamically configurable by actions.
* Contain at least 1 `TaskRef` data field that references a task from another process.
* Contain at least 1 data field of type `enumeration`, or `emuration_map`, or `multichoice`, or `multichoice_map`.
* Begin with just one location with just one token (to facilitate both modeling and testing).
* Contain at least 10 tasks (executable transitions, i.e., at least 10 L1 liveness transitions).
* Contain at least one "decision" (i.e., at least one XOR split, .i.e., a construct where two or more transitions are executable at a time and the system/user must decide which to execute).
* The execution of the decision (i.e., the execution of one transition in the decision) must be automated by actions (cf. builder -> action edit -> functions -> async run with execution of task),
or by a combination of variable weights and auto-triggered transitions.
* Defined process roles that are appropriately chosen for the topic
* The process must contain the actions by which it is modified:
    - Changing the behavior of a data reference (for example, hiding or unhiding it, or changing it to an editable input in a form)
    - Setting a value (e.g. calculating a value, composing text, etc.)
    - Setting options (setting options for `enumeration` / `enumeration_map`, or `multichoice` / `multichoice_map` data field) 
    - Starting / ending a task in another instance of another process

## Finished Project
[![Project 2][project_2]](#)

<!-- MARKDOWN LINKS & IMAGES -->
[project_2]: https://raw.githubusercontent.com/Raychani1/raychani1.github.io/main/projects/petriflow/projects/documentation/I-MSUS_--_Project_2_-_AIS_University_Assignment_Ladislav_Rajcsanyi_97914.png