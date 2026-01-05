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
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MERGED_CSV_PATH = DATA_DIR / "final_merged_dataset.csv"

# -----------------------------
# Load merged CSV
# -----------------------------
merged = pd.read_csv(MERGED_CSV_PATH)

# Ensure column names are stripped of whitespace
merged.columns = merged.columns.str.strip()

# Check required columns exist
required_cols = ['Facility ID', 'Composite_Readmission_Score']
for col in required_cols:
    if col not in merged.columns:
        raise ValueError(f"Required column '{col}' not found in CSV!")

# -----------------------------
# Modeling dataset
# -----------------------------
target_col = 'Composite_Readmission_Score'

# Drop columns not used for modeling (IDs, counts, etc.)
id_cols = ['Facility ID', 'Facility Name', 'State']
leak_cols = [c for c in merged.columns if 'Predicted_Readmission' in c or 'Expected_Readmission' in c]
count_cols = [c for c in merged.columns if c.startswith('Number_of_Readmissions')]

X = merged.drop(columns=id_cols + leak_cols + count_cols + [target_col], errors='ignore')
y = merged[target_col]

# Fill missing numeric values
X = X.fillna(X.mean())

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train models
# -----------------------------
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)

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
# Cross-validation
# -----------------------------
cv_rmse = np.sqrt(-cross_val_score(
    rf, X, y, cv=5, scoring='neg_mean_squared_error'
))
print("CV RMSE Mean:", cv_rmse.mean())

# -----------------------------
# Save artifacts
# -----------------------------
with open(MODEL_DIR / "random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open(MODEL_DIR / "feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

merged.to_csv(OUTPUT_DIR / "final_modeling_dataset.csv", index=False)
