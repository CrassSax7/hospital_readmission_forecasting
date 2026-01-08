# Hospital Readmissions Forecasting

## Overview
This repository predicts hospital readmission rates using CMS data and the Area Deprivation Index (ADI). The pipeline includes:

1. **Data Preparation**: Cleans and merges hospital readmissions, infection, and ADI datasets into a single, consistent dataset (`final_merged_dataset.csv`).
2. **Model Training**: Trains Linear Regression and Random Forest models on the pre-cleaned dataset.

## Repo Structure

hospital_readmission_forecasting/
├─ data/ # Raw CSV files
├─ outputs/ # Preprocessed CSVs
├─ models/ # Saved model artifacts
├─ src/ # Scripts
│ ├─ prepare_data.py # Cleans & merges data
│ └─ train_readmissions_model.py # Trains models
├─ requirements.txt
├─ .gitignore
└─ README.md



## How to Run

```bash
pip install -r requirements.txt
python src/prepare_data.py
python src/train_readmissions_model.py
