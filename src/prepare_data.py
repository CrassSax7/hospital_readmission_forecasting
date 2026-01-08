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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "hospital_readmissions_analytic_table.csv"

READMISSIONS_FILE = DATA_DIR / "FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv"
INFECTIONS_FILE   = DATA_DIR / "Healthcare_Associated_Infections-Hospital.csv"
ADI_FILE          = DATA_DIR / "CO_2023_ADI_9 Digit Zip Code_v4_0_1.csv"

READMISSION_METRICS = [
    "Excess Readmission Ratio",
    "Predicted Readmission Rate",
    "Expected Readmission Rate",
]

print("🔧 Rebuilding analytic dataset from raw sources")

# ============================================================
# LOAD DATA
# ============================================================

readm_df = pd.read_csv(READMISSIONS_FILE)
infect_df = pd.read_csv(INFECTIONS_FILE)
adi_df    = pd.read_csv(ADI_FILE)

# ============================================================
# CANONICAL KEY NORMALIZATION
# ============================================================

readm_df["Facility ID"] = readm_df["Facility ID"].astype(str)
infect_df["Facility ID"] = infect_df["Facility ID"].astype(str)

infect_df["ZIP Code"] = infect_df["ZIP Code"].astype(str).str.zfill(5)

# ============================================================
# CLEAN READMISSIONS DATA
# ============================================================

readm = readm_df.copy()

readm["measure_code"] = (
    readm["Measure Name"]
    .str.replace("READM-30-", "", regex=False)
    .str.replace("-HRRP", "", regex=False)
    .str.lower()
)

for col in READMISSION_METRICS:
    readm[col] = pd.to_numeric(readm[col], errors="coerce")

readm = readm.dropna(subset=["Excess Readmission Ratio"])

readm_pivot = readm.pivot_table(
    index=["Facility ID", "Facility Name", "State"],
    columns="measure_code",
    values=READMISSION_METRICS
)

readm_pivot.columns = [
    f"{metric.replace(' ', '_').lower()}__{measure}"
    for metric, measure in readm_pivot.columns
]

readm_pivot = readm_pivot.reset_index()

assert "Facility ID" in readm_pivot.columns, "Facility ID missing after pivot"

# ============================================================
# CLEAN INFECTIONS DATA
# ============================================================

infect = infect_df.copy()

infect["Score"] = pd.to_numeric(infect["Score"], errors="coerce")
infect = infect.dropna(subset=["Score"])

infect["measure_clean"] = (
    infect["Measure Name"]
    .str.replace("[^A-Za-z0-9]+", "_", regex=True)
    .str.lower()
)

infect_pivot = infect.pivot_table(
    index="Facility ID",
    columns="measure_clean",
    values="Score",
    aggfunc="mean"
).add_prefix("infection__")

infect_pivot = infect_pivot.reset_index()

# ============================================================
# CLEAN ADI DATA (BULLETPROOF)
# ============================================================

adi = adi_df.rename(columns={
    "ZIP_4": "zip",
    "ADI_NATRANK": "adi_national",
    "ADI_STATERNK": "adi_state",
})

# Keep only required columns to avoid dtype pollution
adi = adi[["zip", "adi_national", "adi_state"]]

adi["zip"] = adi["zip"].astype(str).str.zfill(5)

adi["adi_national"] = pd.to_numeric(adi["adi_national"], errors="coerce")
adi["adi_state"] = pd.to_numeric(adi["adi_state"], errors="coerce")

adi = adi.dropna(subset=["adi_national", "adi_state"])

adi = adi.astype({
    "adi_national": "float64",
    "adi_state": "float64",
})

adi_zip = (
    adi.groupby("zip", as_index=False)
       .mean(numeric_only=True)
)

assert adi_zip[["adi_national", "adi_state"]].apply(
    lambda s: np.issubdtype(s.dtype, np.number)
).all(), "ADI aggregation produced non-numeric output"

# ============================================================
# FACILITY → ZIP → ADI BRIDGE
# ============================================================

facility_zip = (
    infect_df[["Facility ID", "ZIP Code"]]
    .drop_duplicates()
    .rename(columns={"ZIP Code": "zip"})
)

facility_zip["Facility ID"] = facility_zip["Facility ID"].astype(str)
facility_zip["zip"] = facility_zip["zip"].astype(str).str.zfill(5)

facility_adi = facility_zip.merge(
    adi_zip,
    on="zip",
    how="left"
)

assert "Facility ID" in facility_adi.columns, "Facility ID missing in facility_adi"

# ============================================================
# FINAL ANALYTIC TABLE
# ============================================================

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

excess_cols = [
    c for c in final_df.columns
    if c.startswith("excess_readmission_ratio__")
]

final_df["composite_readmission_score"] = final_df[excess_cols].mean(axis=1)

# ============================================================
# FINAL VALIDATION
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
# SAVE OUTPUT
# ============================================================

final_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Analytic dataset saved: {OUTPUT_FILE.resolve()}")
print(f"📦 Final shape: {final_df.shape}")
