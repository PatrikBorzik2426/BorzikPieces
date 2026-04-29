## General Description

- This project has for a task to prepare MLOPS pipeline via Domino Workflow with Git repo at this link https://github.com/Tauffer-Consulting/domino
- The task itself is to prepare two workflows, one for histopatology model and one for radiology

## Current Status Histopatology
- no provided data

## Current Status Radialogy
- all of the clean data are located in ./tp-radiology-adonema/tp_radiology_adonema

### Current Semi-functional - Pieces
- all current pieces are in ./pieces

### Current Semi-fucntional - Workflow
Loader
|
-- Data Split
|   |
|   -- Preprocess (Training)
|   -- Preprocess (Test)
|   -- Preprocess (Validation)
|           | (all of preprocesses merge)
|           -- Pituitari Dataset
|                   |
|                   -- Model Training
|                       |
|                       -- Model Inference
|
-- EDA
    |
    -- Visualisation

## Current task for claude
- analyse the original project on github
    - how to make pieces
    - how to setup the application in docker
    - best approach in uploading and updating new custom pieces
    - setupu domino-workflow locally
- analyse radiology project
    - analyse all elements of current state of the project
    - analyse and define all necessary building block to make workflow out of it
    - validate the current pieces in comaprison to the workflow defines by the radiology project
- prepare / update the pieces based on the comaprison
    - make any necessary changes to prapre working workflow
    - run the application locally for the testing
