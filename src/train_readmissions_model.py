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
# create train/test datasets, perform k-fold cross-validation
from sklearn.model_selection import train_test_split, cross_val_score
# create baseline linear regression model, random forest model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# model evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score
# handle missing values -> replace with column means
from sklearn.impute import SimpleImputer
# serialize python objects to disk
import pickle

# ============================================================
# PATHS -> # find project root dynamically 
# (relative paths for location independent functionality)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1
# define where input data is and where trained models saved
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
# define path to dataset created by prev script
DATA_FILE = DATA_DIR / "hospital_readmissions_analytic_table.csv"

# ============================================================
# LOAD DATA -> load merged data set into df
# ============================================================

df = pd.read_csv(DATA_FILE)

# define relevant columns for effective modeling
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
# MODEL PREP -> define prediction target (dependent variable)
# ============================================================

TARGET = "composite_readmission_score"

# create feature matrix x, minus non-pred identifiers, target
X = df.drop(columns=[
    "Facility ID",
    "Facility Name",
    "State",
    TARGET,
])

# extract target vector
y = df[TARGET]

# remove missing columns
X = X.dropna(axis=1, how="all")

# implement mean value imputer for missing values
imputer = SimpleImputer(strategy="mean")

# fit imputer on full dataset, wrap output into DF w/ proper labels
X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index,
)

# ============================================================
# TRAIN / TEST -> 80/20 train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# instantiate LR model, fit to train data
lr = LinearRegression()
lr.fit(X_train, y_train)

# 100 trees, reproducible randomness, use all CPU cores
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train) #train random forest

# ============================================================
# EVALUATION -> function to predict on test data
# compute RMSE, R^2, return metrics as dictionary
# ============================================================

def evaluate(model):
    preds = model.predict(X_test)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
    }

# LR & random forest eval on test set, print results
print("\n📊 Model Performance")
print("Linear Regression:", evaluate(lr))
print("Random Forest:", evaluate(rf))

# perform 5-fold cross validation on entire dataset
cv_rmse = np.sqrt(
    -cross_val_score(
        rf,
        X,
        y,
        cv=5,
        scoring="neg_mean_squared_error",
    )
)

# print ave RMSE, variability of model performance
print("\n📈 Random Forest CV RMSE")
print("Mean:", round(cv_rmse.mean(), 4), "Std:", round(cv_rmse.std(), 4))

# ============================================================
# SAVE ARTIFACTS -> create artifacts/dir if it doesn't exist
# save trained RF mod, feature names, fitted imputer for 
# consistent missing values
# ============================================================

ARTIFACT_DIR.mkdir(exist_ok=True)

with open(ARTIFACT_DIR / "random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open(ARTIFACT_DIR / "feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

with open(ARTIFACT_DIR / "imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

print("\n✅ Model artifacts saved")
