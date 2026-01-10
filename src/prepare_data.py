# ============================================================
# Hospital Readmission Forecasting – Data Preparation Script
# ============================================================
# Author: J. Casey Brookshier
# Purpose: Build analytic dataset for readmission modeling
# Inputs: Raw CMS Readmissions, Infections, ADI files
# Output: hospital_readmissions_analytic_table.csv
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG (RELATIVE PATHS)
# ============================================================
# find project root dynamically (relative paths for location independent functionality)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
# define where merged dataset to be written
OUTPUT_FILE = DATA_DIR / "hospital_readmissions_analytic_table.csv"

# define filepaths for raw data sources
READMISSIONS_FILE = DATA_DIR / "FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv"
INFECTIONS_FILE   = DATA_DIR / "Healthcare_Associated_Infections-Hospital.csv"
ADI_FILE          = DATA_DIR / "CO_2023_ADI_9 Digit Zip Code_v4_0_1.csv"

# Define readmission columns of interest, centralize metric control
READMISSION_METRICS = [
    "Excess Readmission Ratio",
    "Predicted Readmission Rate",
    "Expected Readmission Rate",
]

# log confirmation of start of pipeline execution
print("🔧 Rebuilding analytic dataset from raw sources")

# ============================================================
# LOAD DATA -> read CSV's into pandas DF
# ============================================================

readm_df = pd.read_csv(READMISSIONS_FILE)
infect_df = pd.read_csv(INFECTIONS_FILE)
adi_df    = pd.read_csv(ADI_FILE)

# ============================================================
# CANONICAL KEY NORMALIZATION -> force Facility ID into string
# ============================================================

readm_df["Facility ID"] = readm_df["Facility ID"].astype(str)
infect_df["Facility ID"] = infect_df["Facility ID"].astype(str)
# normalize ZIP into 5 digit strings for reliable ZIP/ADI join
infect_df["ZIP Code"] = infect_df["ZIP Code"].astype(str).str.zfill(5)

# ============================================================
# CLEAN READMISSIONS DATA
# ============================================================
# create working copy
readm = readm_df.copy()

# convert verbose CMS names into clean feature identifiers
readm["measure_code"] = (
    readm["Measure Name"]
    .str.replace("READM-30-", "", regex=False)
    .str.replace("-HRRP", "", regex=False)
    .str.lower()
)
# convert readmission metric to numeric, invalid values to NaN
for col in READMISSION_METRICS:
    readm[col] = pd.to_numeric(readm[col], errors="coerce")
    
# remove hospital w/o valid readmission signal
readm = readm.dropna(subset=["Excess Readmission Ratio"])

# reshape data long ->wide, one facility one row, one condition one column
readm_pivot = readm.pivot_table(
    index=["Facility ID", "Facility Name", "State"],
    columns="measure_code",
    values=READMISSION_METRICS
)

# flatten pandas' multi-index columns, create ML-ready feature names
readm_pivot.columns = [
    f"{metric.replace(' ', '_').lower()}__{measure}"
    for metric, measure in readm_pivot.columns
]
# change index fields back into columns
readm_pivot = readm_pivot.reset_index()

# check for facility ID pipeline corruption
assert "Facility ID" in readm_pivot.columns, "Facility ID missing after pivot"

# ============================================================
# CLEAN INFECTIONS DATA
# ============================================================
# working copy
infect = infect_df.copy()

# infection scores to numeric, remove invalid rows
infect["Score"] = pd.to_numeric(infect["Score"], errors="coerce")
infect = infect.dropna(subset=["Score"])

# standardize measure names into clean column identifiers
infect["measure_clean"] = (
    infect["Measure Name"]
    .str.replace("[^A-Za-z0-9]+", "_", regex=True)
    .str.lower()
)

# aggregate infection metrics per hospital
infect_pivot = infect.pivot_table(
    index="Facility ID",
    columns="measure_clean",
    values="Score",
    aggfunc="mean"
).add_prefix("infection__")

# prep for merge
infect_pivot = infect_pivot.reset_index()

# ============================================================
# CLEAN ADI DATA 
# ============================================================
# improve column readability
adi = adi_df.rename(columns={
    "ZIP_4": "zip",
    "ADI_NATRANK": "adi_national",
    "ADI_STATERNK": "adi_state",
})

# Keep only required columns to avoid dtype pollution
adi = adi[["zip", "adi_national", "adi_state"]]

# norm ZIP to 5 value string
adi["zip"] = adi["zip"].astype(str).str.zfill(5)

# adi values numeric/valid
adi["adi_national"] = pd.to_numeric(adi["adi_national"], errors="coerce")
adi["adi_state"] = pd.to_numeric(adi["adi_state"], errors="coerce")
adi = adi.dropna(subset=["adi_national", "adi_state"])
adi = adi.astype({
    "adi_national": "float64",
    "adi_state": "float64",
})
# aggregate 9 digit zip to 5 digit
adi_zip = (
    adi.groupby("zip", as_index=False)
       .mean(numeric_only=True)
)
# verify aggregation produces numeric
assert adi_zip[["adi_national", "adi_state"]].apply(
    lambda s: np.issubdtype(s.dtype, np.number)
).all(), "ADI aggregation produced non-numeric output"

# ============================================================
# FACILITY → ZIP → ADI BRIDGE
# ============================================================
# build facility to ZIP lookup table
facility_zip = (
    infect_df[["Facility ID", "ZIP Code"]]
    .drop_duplicates()
    .rename(columns={"ZIP Code": "zip"})
)
facility_zip["Facility ID"] = facility_zip["Facility ID"].astype(str)
facility_zip["zip"] = facility_zip["zip"].astype(str).str.zfill(5)

# attach SES context by facility
facility_adi = facility_zip.merge(
    adi_zip,
    on="zip",
    how="left"
)

assert "Facility ID" in facility_adi.columns, "Facility ID missing in facility_adi"

# ============================================================
# FINAL ANALYTIC TABLE -> readmissions + infections +SES data
# ============================================================
# create merged ML ready DF
final_df = (
    readm_pivot
    .merge(infect_pivot, on="Facility ID", how="left")
    .merge(
        facility_adi[["Facility ID", "adi_national", "adi_state"]],
        on="Facility ID",
        how="left"
    )
)

# ============================================================
# TARGET CONSTRUCTION
# ============================================================
# identify excess readmission metrics
excess_cols = [
    c for c in final_df.columns
    if c.startswith("excess_readmission_ratio__")
]
# create model target, AVE excess readmission
final_df["composite_readmission_score"] = final_df[excess_cols].mean(axis=1)

# ============================================================
# FINAL VALIDATION -> raise error if key columns missing
# ============================================================

REQUIRED_COLUMNS = {
    "Facility ID",
    "Facility Name",
    "State",
    "composite_readmission_score",
}
missing = REQUIRED_COLUMNS - set(final_df.columns)
if missing:
    raise ValueError(f"Missing required columns in final dataset: {missing}")

# ============================================================
# SAVE OUTPUT, write analytic dataset to disk, report size
# ============================================================

final_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Analytic dataset saved: {OUTPUT_FILE.resolve()}")
print(f"📦 Final shape: {final_df.shape}")
