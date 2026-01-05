# ============================================
# Hospital Readmissions Model Training
# Using pre-cleaned final_merged_dataset.csv
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from pathlib import Path

# -----------------------------
# Paths (Relative for GitHub)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "outputs"  # final_merged_dataset.csv is stored here
MODEL_DIR = BASE_DIR / "models"

os.makedirs(MODEL_DIR, exist_ok=True)

MERGED_CSV_PATH = DATA_DIR / "final_merged_dataset.csv"

# -----------------------------
# Load pre-cleaned dataset
# -----------------------------
merged = pd.read_csv(MERGED_CSV_PATH)

# -----------------------------
# Ensure 'Facility ID' is string
# -----------------------------
if 'Facility ID' not in merged.columns:
    raise KeyError("'Facility ID' column is missing from final_merged_dataset.csv")
merged['Facility ID'] = merged['Facility ID'].astype(str)

# -----------------------------
# Target Engineering
# -----------------------------
excess_cols = [c for c in merged.columns if c.startswith("Excess_Readmission_Ratio")]
merged['Composite_Readmission_Score'] = merged[excess_cols].mean(axis=1)
merged = merged.dropna(subset=['Composite_Readmission_Score'])

# -----------------------------
# Prepare Modeling Dataset
# -----------------------------
leak_cols = [c for c in merged.columns if "Predicted_Readmission" in c or "Expected_Readmission" in c]
id_cols = ['Facility ID', 'Facility Name', 'State']
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
with open(MODEL_DIR / "random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open(MODEL_DIR / "feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Model training completed. Artifacts saved in 'models/' directory.")
