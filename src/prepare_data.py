# ============================================================
# prepare_data.py
# ============================================================
# Purpose:
#   Build a single analytic dataset for hospital readmission
#   modeling from raw CMS + ADI data.
#
# Output:
#   data/hospital_readmissions_analytic_table.csv
# ============================================================

import pandas as pd
from pathlib import Path

# ============================================================
# PATHS (RELATIVE, GITHUB-SAFE)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

READMISSIONS_FILE = DATA_DIR / "FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv"
INFECTIONS_FILE   = DATA_DIR / "Healthcare_Associated_Infections-Hospital.csv"
ADI_FILE          = DATA_DIR / "CO_2023_ADI_9 Digit Zip Code_v4_0_1.csv"

OUTPUT_FILE = DATA_DIR / "hospital_readmissions_analytic_table.csv"

# ============================================================
# LOAD DATA
# ============================================================

readm = pd.read_csv(READMISSIONS_FILE)
infect = pd.read_csv(INFECTIONS_FILE)
adi = pd.read_csv(ADI_FILE)

# ============================================================
# CANONICAL KEYS
# ============================================================

readm["Facility ID"] = readm["Facility ID"].astype(str)
infect["Facility ID"] = infect["Facility ID"].astype(str)
infect["ZIP Code"] = infect["ZIP Code"].astype(str).str.zfill(5)

# ============================================================
# CLEAN READMISSIONS (LONG → WIDE)
# ============================================================

readm["measure_code"] = (
    readm["Measure Name"]
    .str.replace("READM-30-", "", regex=False)
    .str.replace("-HRRP", "", regex=False)
    .str.lower()
)

readm["Excess Readmission Ratio"] = pd.to_numeric(
    readm["Excess Readmission Ratio"], errors="coerce"
)

readm = readm.dropna(subset=["Excess Readmission Ratio"])

readm_pivot = readm.pivot_table(
    index=["Facility ID", "Facility Name", "State"],
    columns="measure_code",
    values="Excess Readmission Ratio",
)

# 🔑 CRITICAL FIX: restore index columns
readm_pivot = readm_pivot.reset_index()

assert "Facility ID" in readm_pivot.columns, \
    "Facility ID missing after pivot"

# ============================================================
# CLEAN INFECTIONS
# ============================================================

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
    aggfunc="mean",
).reset_index()

infect_pivot = infect_pivot.add_prefix("infection__")
infect_pivot = infect_pivot.rename(
    columns={"infection__Facility ID": "Facility ID"}
)
# ============================================================
# CLEAN ADI (BULLETPROOF VERSION)
# ============================================================

adi = adi.rename(columns={
    "ZIP_4": "zip",
    "ADI_NATRANK": "adi_national",
    "ADI_STATERNK": "adi_state",
})

# Keep only required columns (prevents hidden dtype pollution)
adi = adi[["zip", "adi_national", "adi_state"]]

adi["zip"] = adi["zip"].astype(str).str.zfill(5)

# Force numeric conversion explicitly
adi["adi_national"] = pd.to_numeric(adi["adi_national"], errors="coerce")
adi["adi_state"] = pd.to_numeric(adi["adi_state"], errors="coerce")

# Drop rows with missing ADI
adi = adi.dropna(subset=["adi_national", "adi_state"])

# 🔒 ENSURE NUMERIC TYPES (this line matters)
adi = adi.astype({
    "adi_national": "float64",
    "adi_state": "float64",
})

adi_zip = (
    adi.groupby("zip", as_index=False)
       .mean(numeric_only=True)
)

# ============================================================
# FINAL MERGE
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
# TARGET VARIABLE (CANONICAL NAME)
# ============================================================

readmission_cols = [
    c for c in final_df.columns
    if c not in ["Facility ID", "Facility Name", "State"]
    and not c.startswith("infection__")
    and c not in ["adi_national", "adi_state"]
]

final_df["composite_readmission_score"] = final_df[readmission_cols].mean(axis=1)

# ============================================================
# SCHEMA VALIDATION
# ============================================================

REQUIRED_COLUMNS = {
    "Facility ID",
    "Facility Name",
    "State",
    "composite_readmission_score",
}

missing = REQUIRED_COLUMNS - set(final_df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ============================================================
# SAVE
# ============================================================

final_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Analytic dataset written to {OUTPUT_FILE}")
print("Shape:", final_df.shape)
