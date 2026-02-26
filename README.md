# 🏥 Hospital Readmission Forecasting

**Author:** J. Casey Brookshier  
**Last Updated:** July 2025  

## 📌 Project Overview

This project builds an end-to-end, reproducible machine learning pipeline to predict hospital readmission risk using publicly available CMS quality metrics, healthcare-associated infection data, and socioeconomic deprivation indicators (Area Deprivation Index, ADI).

The goal is to help healthcare administrators and policy analysts identify facilities at higher risk of readmission penalties and target interventions more effectively.

---

## 🎯 Objective

To develop a predictive model for hospital-level readmission performance by:

- Cleaning and standardizing multiple CMS datasets
- Integrating clinical quality, infection control, and socioeconomic risk factors
- Engineering a composite readmission risk score
- Comparing linear and tree-based regression models
- Producing deployable model artifacts

---

## 📊 Data Sources

All data are publicly available:

- **CMS Hospital Readmissions Reduction Program (FY2025)**  
  Hospital-level readmission metrics by clinical condition

- **Healthcare-Associated Infections – Hospital**  
  Facility-level infection control performance indicators

- **Area Deprivation Index (ADI)**  
  ZIP-code–level socioeconomic disadvantage metrics

> Raw data files are stored in `/data`.  
> The analytic dataset is generated programmatically.

---

---

## 📊 Key Findings

### 🔧 Data Engineering
- Rebuilt analytic dataset from raw hospital data sources  
- Final dataset shape: **2,862 rows × 60 features**  
- Cleaned, merged, and validated structured healthcare data  

---

### 🤖 Model Performance

| Model              | RMSE   | R² Score |
|--------------------|--------|----------|
| Linear Regression  | 0.0253 | 0.7936   |
| Random Forest      | 0.0147 | 0.9299   |

---

### 🌲 Random Forest Cross-Validation

- **Mean CV RMSE:** 0.0193  
- **Standard Deviation:** 0.0043  

✔ Random Forest substantially outperformed Linear Regression  
✔ Strong predictive accuracy (R² ≈ 0.93)  
✔ Low cross-validation variance → model is stable and generalizable  

---

### 📦 Outputs Generated

- Clean analytic dataset  
- Trained model artifacts  
- Performance metrics  
- Saved prediction outputs  

---

### 💡 Business Interpretation

- Ensemble models significantly improve readmission prediction accuracy  
- Model stability suggests strong real-world deployment potential  
- Feature-rich dataset (60 variables) captures meaningful signal for hospital readmission risk  

---


## 🧱 Project Structure

hospital_readmission_forecasting/
│
├── data/
│ ├── FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv
│ ├── Healthcare_Associated_Infections-Hospital.csv
│ ├── CO_2023_ADI_9 Digit Zip Code_v4_0_1.csv
│ └── hospital_readmissions_analytic_table.csv # auto-generated
│
├── artifacts/
│ ├── random_forest_model.pkl
│ ├── feature_names.pkl
│ └── imputer.pkl
│
├── src/
│ ├── prepare_data.py
│ └── train_readmissions_model.py
│
├── requirements.txt
└── README.md

---

## 🚀 How to Run (Mac)

```bash
git clone https://github.com/CrassSax7/hospital_readmission_forecasting.git
cd hospital_readmission_forecasting

python3 -m venv venv
source venv/bin/activate


python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python src/prepare_data.py

python src/train_readmissions_model.py
```

---

## 🚀 How to Run (Windows)

```bash
git clone https://github.com/CrassSax7/hospital_readmission_forecasting.git
cd hospital_readmission_forecasting

python -m venv venv
venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python src\prepare_data.py

python src\train_readmissions_model.py
```




