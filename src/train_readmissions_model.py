# ============================================================
# train_readmissions_model.py
# ============================================================
# Purpose:
#   Train and evaluate readmission risk models using the
#   prepared analytic dataset.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import pickle

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

DATA_FILE = DATA_DIR / "hospital_readmissions_analytic_table.csv"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

REQUIRED_COLUMNS = {
    "Facility ID",
    "Facility Name",
    "State",
    "composite_readmission_score",
}

missing = REQUIRED_COLUMNS - set(df.columns)
if missing:
    raise ValueError(f"Required column(s) missing: {missing}")

print(f"✅ Loaded dataset: {df.shape}")

# ============================================================
# MODEL PREP
# ============================================================

TARGET = "composite_readmission_score"

X = df.drop(columns=[
    "Facility ID",
    "Facility Name",
    "State",
    TARGET,
])

y = df[TARGET]

X = X.dropna(axis=1, how="all")

imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index,
)

# ============================================================
# TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model):
    preds = model.predict(X_test)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
    }

print("\n📊 Model Performance")
print("Linear Regression:", evaluate(lr))
print("Random Forest:", evaluate(rf))

cv_rmse = np.sqrt(
    -cross_val_score(
        rf,
        X,
        y,
        cv=5,
        scoring="neg_mean_squared_error",
    )
)

print("\n📈 Random Forest CV RMSE")
print("Mean:", round(cv_rmse.mean(), 4), "Std:", round(cv_rmse.std(), 4))

# ============================================================
# SAVE ARTIFACTS
# ============================================================

ARTIFACT_DIR.mkdir(exist_ok=True)

with open(ARTIFACT_DIR / "random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open(ARTIFACT_DIR / "feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

with open(ARTIFACT_DIR / "imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

print("\n✅ Model artifacts saved")
