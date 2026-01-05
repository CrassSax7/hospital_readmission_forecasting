# Hospital Readmissions Prediction

This repository contains a **hospital readmissions forecasting pipeline**. The pipeline uses hospital readmission and infection data to predict a **Composite Readmission Score** for hospitals, leveraging Linear Regression and Random Forest models.

## Project Structure

hospital_readmission_forecasting/
│
├── data/ # Raw CSV data files (not included)
├── models/ # Saved trained models and feature artifacts
├── outputs/ # Final merged datasets, logs, evaluation outputs
├── src/
│ └── train_readmissions_model.py # Main pipeline script
├── requirements.txt
├── .gitignore
└── README.md


## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/hospital_readmission_forecasting.git
cd hospital_readmission_forecasting
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
cd src
python train_readmissions_model.py

