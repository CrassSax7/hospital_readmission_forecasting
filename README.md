# 🏥 Hospital Quality Forecasting: Predicting Readmission Risk

## 📌 Project Overview
This project builds a **machine learning model to predict hospital readmission risk** using integrated **clinical performance metrics, infection control indicators, and socioeconomic deprivation data**.

By combining CMS Hospital Readmissions Reduction Program data, Healthcare-Associated Infection measures, and the Area Deprivation Index (ADI), the model helps identify hospitals at higher risk of readmission penalties—enabling data-driven quality improvement and policy decisions.

---

## 🎯 Objective
To develop a predictive model that estimates **30-day hospital readmission performance** and identifies key drivers of excess readmissions, with an emphasis on:
- Infection control quality
- Socioeconomic risk (ADI)
- Condition-specific readmission patterns

---

## 📊 Data Sources
- **CMS Hospital Readmissions Reduction Program (FY2025)**  
  Condition-level excess readmission ratios for U.S. hospitals

- **Healthcare-Associated Infections Dataset**  
  Facility-level infection metrics (CLABSI, CAUTI, C. Diff, MRSA, SSI)

- **Area Deprivation Index (ADI)**  
  ZIP-level socioeconomic disadvantage indicators

> All data used are publicly available.

---

## 🧠 Modeling Approach
**Workflow:**  
Clean → Standardize → Aggregate → Integrate → Model → Evaluate

### Feature Engineering
- Pivoted condition-level readmission metrics
- Aggregated infection scores by facility
- Mapped ZIP-level ADI to hospitals
- Created a **Composite Readmission Score** as the modeling target

### Models Compared
- Linear Regression (baseline)
- Random Forest Regressor (final model)

---

## 📈 Results
| Model | RMSE | R² |
|------|------|----|
| Linear Regression | ~0.029 | ~0.74 |
| Random Forest | **~0.018** | **~0.89–0.90** |

**Top Predictors**
- Excess readmission ratios (HF, Hip/Knee, Pneumonia)
- Infection metrics (C. Diff, MRSA)
- Socioeconomic deprivation (ADI)

---

## 🗺️ Outputs
- Trained Random Forest model
- Feature importance rankings
- Final integrated modeling dataset
- Serialized model artifacts for deployment

---

## 📁 Repository Structure
hospital-readmissions-forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│ ├── FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv
│ ├── Healthcare_Associated_Infections-Hospital.csv
│ └── CO_2023_ADI_9_Digit_Zip_Code.csv
│
├── src/
│ └── train_readmissions_model.py
│
├── models/
│ ├── random_forest_model.pkl
│ └── feature_names.pkl
│
└── outputs/
└── final_merged_dataset.csv




---

## ▶️ How to Run
```bash
pip install -r requirements.txt
cd src
python train_readmissions_model.py

