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
+ Project Structure:
```
project
├── aquaculture
│   ├── app_v2
│   ├── apps
│   ├── assets
│   │   ├── cache
│   │   ├── data
│   │   │   └── final_info.csv <-- run cli_main.py data to create 
│   │   └── models
│   ├── exps
│   ├── utils
│   ├── cli_main.py
│   └── common.py
│   └── ...
├── data
│   ├── a2i_data   <-- pointer to <mnt/a2i_data>
│   ├── preprocessed 
│   └── exps
├── images
└── mnt
    └── a2i_data <-- copy or link to AI competition data
        ├── csv
        │   ├── 10월01일
        │   │   ├── 2-1-1-1-1-1001-0010000.csv
        │   │   └── 2-1-1-1-1-1001-0020000.csv
        │   ├── 10월04일
        │   └── ...
        └── 먹이생물
            ├── 10월01일
            │   ├── 고성
            │   │   ├── 2-1-1-2-2-1001-0120001.jpg
            │   │   ├── 2-1-1-2-2-1001-0120002.jpg
            │   │   └── ...
            │   ├── 일해
            │   │   ├── 2-1-1-2-2-1001-0110001.jpg
            │   │   ├── 2-1-1-2-2-1001-0110002.jpg
            │   │   └── ...
            ├── 10월04일
            └── ...
```
+ **mnt/a2i_data** : link or copy AI_competition data containing 2 folders: **csv** (sensor data), **먹이생물** (images data)
  + Go to folder **mnt**
  + Type command
```bash
# Linux
ln -s <path to folder contain 'csv' file and '먹이생물' file> a2i_data

# Window
mklink /J a2i_data <path to folder contain 'csv' file and '먹이생물' file>
``` 
 
+ Setup **Python Environments**:
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

## How to run program
+ Activate environment a2i 
### Export data files
### Run application
+ Go to project root
+ Activate Environment a2i
+ Type command
```bash
python aquaculture/cli_main.py app2 --app-type dash
```
+ Open Web browser and type url: http://localhost:8050

### Run demo experiments 
+ Go to project root
+ Start jupyter-lab and run jupyter notebooks in aquaculture/exps
  + 01_datapreprocess.ipynb
  + 02_cell_counts.ipynb
  + 03_data_analsyis.ipynb
  + 04_tabnet_checking.ipynb


