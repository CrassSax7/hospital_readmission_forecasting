# ============================================
# Hospital Readmissions Prediction Pipeline
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# -----------------------------
# Paths (Relative for GitHub)
# -----------------------------
DATA_DIR = "../data/"
MODEL_DIR = "../models/"
OUTPUT_DIR = "../outputs/"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

READMISSIONS_PATH = DATA_DIR / "FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv"
INFECTIONS_PATH = DATA_DIR / "Healthcare_Associated_Infections-Hospital.csv"
ADI_PATH = DATA_DIR / "CO_2023_ADI_9 Digit Zip Code_v4_0_1.csv"


# -----------------------------
# Load Data
# -----------------------------
readmissions_df = pd.read_csv(READMISSIONS_PATH)
infections_df = pd.read_csv(INFECTIONS_PATH)
adi_df = pd.read_csv(ADI_PATH)

# -----------------------------
# Clean Readmissions Data
# -----------------------------
keep_cols = [
    'Facility Name', 'Facility ID', 'State',
    'Measure Name', 'Excess Readmission Ratio',
    'Predicted Readmission Rate', 'Expected Readmission Rate',
    'Number of Readmissions'
]

readm = readmissions_df[keep_cols].copy()
numeric_cols = keep_cols[4:]

readm[numeric_cols] = readm[numeric_cols].apply(
    pd.to_numeric, errors='coerce'
)

readm = readm.dropna(subset=['Excess Readmission Ratio'])

readm_pivot = readm.pivot_table(
    index=['Facility ID', 'Facility Name', 'State'],
    columns='Measure Name',
    values=numeric_cols
).reset_index()

readm_pivot.columns = [
    f"{metric.replace(' ', '_')}_{measure.replace('-', '').replace(' ', '')}"
    if isinstance(metric, str) else metric
    for metric, measure in readm_pivot.columns
]

readm_pivot.rename(columns={'Facility ID_': 'Facility ID'}, inplace=True)
readm_pivot['Facility ID'] = readm_pivot['Facility ID'].astype(str)

# -----------------------------
# Clean Infection Data
# -----------------------------
inf = infections_df[['Facility ID', 'Measure Name', 'Score']].copy()
inf['Score'] = pd.to_numeric(inf['Score'], errors='coerce')
inf = inf.dropna(subset=['Score'])

inf_pivot = inf.pivot_table(
    index='Facility ID',
    columns='Measure Name',
    values='Score',
    aggfunc='mean'
).reset_index()

inf_pivot.columns = [
    f"Infection_{c.replace(' ', '_').replace('/', '_')}"
    if c != 'Facility ID' else c
    for c in inf_pivot.columns
]

inf_pivot['Facility ID'] = inf_pivot['Facility ID'].astype(str)

# -----------------------------
# Merge Readmissions + Infections
# -----------------------------
merged = pd.merge(
    readm_pivot,
    inf_pivot,
    on='Facility ID',
    how='left'
)

# -----------------------------
# ADI Processing
# -----------------------------
adi = adi_df.rename(columns={
    'ZIP_4': 'ZIP',
    'ADI_NATRANK': 'ADI_National_Rank',
    'ADI_STATERNK': 'ADI_State_Rank'
})

adi['ZIP'] = adi['ZIP'].astype(str).str.zfill(5)
adi[['ADI_National_Rank', 'ADI_State_Rank']] = adi[
    ['ADI_National_Rank', 'ADI_State_Rank']
].apply(pd.to_numeric, errors='coerce')

adi = adi.dropna()

adi_agg = adi.groupby('ZIP', as_index=False).mean()

facility_zip = infections_df[['Facility ID', 'ZIP Code']].drop_duplicates()
facility_zip['Facility ID'] = facility_zip['Facility ID'].astype(str)
facility_zip['ZIP Code'] = facility_zip['ZIP Code'].astype(str).str.zfill(5)

facility_adi = pd.merge(
    facility_zip,
    adi_agg,
    left_on='ZIP Code',
    right_on='ZIP',
    how='left'
).drop(columns=['ZIP'])

merged = pd.merge(
    merged,
    facility_adi,
    on='Facility ID',
    how='left'
)

# -----------------------------
# Target Engineering
# -----------------------------
excess_cols = [c for c in merged.columns if c.startswith("Excess_Readmission_Ratio")]
merged['Composite_Readmission_Score'] = merged[excess_cols].mean(axis=1)

merged = merged.dropna(subset=['Composite_Readmission_Score'])

# -----------------------------
# Modeling Dataset
# -----------------------------
leak_cols = [c for c in merged.columns if "Predicted_Readmission" in c or "Expected_Readmission" in c]
id_cols = ['Facility ID', 'Facility_Name_', 'State_']
count_cols = [c for c in merged.columns if c.startswith("Number_of_Readmissions")]

X = merged.drop(columns=leak_cols + id_cols + count_cols + ['Composite_Readmission_Score'])
y = merged['Composite_Readmission_Score']

X = X.fillna(X.mean())

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Models
# -----------------------------
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
).fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    return rmse, r2

print("Linear Regression:", evaluate(lr, X_test, y_test))
print("Random Forest:", evaluate(rf, X_test, y_test))

# -----------------------------
# Cross Validation
# -----------------------------
cv_rmse = np.sqrt(-cross_val_score(
    rf, X, y, cv=5, scoring="neg_mean_squared_error"
))

print("CV RMSE Mean:", cv_rmse.mean())

# -----------------------------
# Save Artifacts
# -----------------------------
with open(MODEL_DIR + "random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open(MODEL_DIR + "feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

merged.to_csv(OUTPUT_DIR + "final_merged_dataset.csv", index=False)
