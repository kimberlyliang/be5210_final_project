# BE 521: Final Project Part 1
Brain-Computer Interface Project: Predicting Finger Movements from ECoG Data

## Project Overview
This project involves predicting finger flexion using intracranial EEG (ECoG) data from three human subjects. The data comes from the 4th BCI Competition (Miller et al. 2008).

## Data Files
This project requires two data files that are not included in the repository due to their size:
- `final_proj_part1_data.pkl`: Contains the ECoG and data glove recordings
- `testRfunction.pkl`: Contains test data for validating the R matrix function

To run this project, you will need to:
1. Obtain these files from the course materials
2. Place them in the root directory of the project

## Project Structure
- `Final_Project_Part_1_Questions.ipynb`: Main notebook containing the project code

## Data Format
- ECoG data:
  - Subject 1: 61 channels
  - Subject 2: 46 channels
  - Subject 3: 64 channels
- Data glove: 5 channels (one per finger)
- Sampling rate: 1000 Hz

## Requirements
- Python 3.x
- Required packages:
  - numpy
  - matplotlib
  - pandas
  - scipy
  - pickle

## Usage
1. Install required packages
2. Place data files in project root
3. Run the Jupyter notebook

## Note
The data files are ignored by Git (see .gitignore) due to their size. Contact course staff if you need access to the data files.

## Authors
Kimberly Liang
Suraj Oruganti