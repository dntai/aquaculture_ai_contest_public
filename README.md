# 2021 Aquaculture Artificial Intelligence Idea Contest

## Overview
+ **Subject**: Free Topic (Food-organism utilization throughout the AI-based aquaculture industry)
+ **Home page**: http://sarc.jnu.ac.kr/contest/20211105/
+ **Motivation**:
  + Stable mass feeding management of food organisms is importance of 'artificial seed culture' industry
  + Problems cause serious economic loss in the aquaculture artificial seed production industry
    + Reduction of aquaculture food organisms (plankton)
    + Difficulty in managing mass culture/feeding of food organisms
    + Mass mortality in the process of seed production
    + Decreased utilization of food organisms by field
    ![](images/food_organism_problem.png)
    > A sharp decline in marine food organisms (plankton) due to global warming and marine environmental pollution
  + It is necessary to select and concentrate the government R&D AI data and technology to solve fundamental problems such as instability and low productivity in the aquaculture industry

## Team information: 
+ Team Name: **ADLER**
+ Affiliation: Chonnam National University, South Korea

## Setup Project
### Project Structure:
```
project
├── aquaculture
│   ├── app_v2
│   ├── apps
│   ├── assets
│   │   ├── cache
│   │   ├── data <-- Setup Data step
│   │   │   └── final_info.csv 
│   │   └── models
│   ├── exps
│   ├── utils
│   ├── cli_main.py
│   └── common.py
│   └── ...
├── data
│   ├── a2i_data <-- copy csv, 먹이생물 into here
│   │   ├── csv
│   │   │   ├── 10월01일
│   │   │   │   ├── 2-1-1-1-1-1001-0010000.csv
│   │   │   │   └── 2-1-1-1-1-1001-0020000.csv
│   │   │   ├── 10월04일
│   │   │   └── ...
│   │   └── 먹이생물
│   │       ├── 10월01일
│   │       │   ├── 고성
│   │       │   │   ├── 2-1-1-2-2-1001-0120001.jpg
│   │       │   │   ├── 2-1-1-2-2-1001-0120002.jpg
│   │       │   │   └── ...
│   │       │   ├── 일해
│   │       │   │   ├── 2-1-1-2-2-1001-0110001.jpg
│   │       │   │   ├── 2-1-1-2-2-1001-0110002.jpg
│   │       │   │   └── ...
│   │       ├── 10월04일
│   │       └── ...
│   ├── preprocessed <-- Setup Data step
│   │   ├── full_info.hdf5
│   │   ├── full_info.xlsx
│   │   ├── final_info.csv
│   │   ├── final_info.xlsx
│   │   └── final_info.hdf5
│   └── exps
└── images
```
### Setup Environments
+ Install Anaconda3 at https://www.anaconda.com/products/individual
+ Activate environment base
```bash
# Linux
conda activate base

# Window
activate base
```
  + Create environment a2i
```bash
conda create -n a2i python=3.8
```
  + Activate environment a2i
```bash
# Linux
conda activate a2i

# Window
activate a2i
```
  + Install requirements packages at environment base
```bash
pip install -r requirements.txt 
```

### Setup Data
+ Copy csv (sensors data), 먹이생물 (microscopy images) to folder **data**
+ Open console 
+ Go to project root 
```bash
# Linux
cd <project dir>

# Window
cd /d <project dir>
```
+ Activate Environment a2i
```bash
# Linux
conda activate a2i

# Window
activate a2i
```
+ Generate index files
```bash
python aquaculture/cli_main.py index
python aquaculture/cli_main.py detect-all 
```

## How to run program
### Run application
+ Go to project root
+ Activate Environment a2i
+ Type command
```bash
python aquaculture/cli_main.py app2 --app-type dash
```
+ Open Web browser and type url: http://localhost:8050

### Experiment Demo 
+ Open html in notebooks folder
+ 
  + 01_datapreprocess.ipynb
  + 02_cell_counts.ipynb
  + 03_data_analsyis.ipynb
  + 04_tabnet_checking.ipynb


