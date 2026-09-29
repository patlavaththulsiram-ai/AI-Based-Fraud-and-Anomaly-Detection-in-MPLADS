"""Data Preprocessing & Cleaning Engine
Standardizes textual features, handles missing values via statistical imputation,
normalizes dates and numeric representations, segregates corrupt/invalid records,
and maintains an immutable, chronological cleaning audit trail.
"""

import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from config import Config
from src.validate import validate_dataset

def clean_numeric_value(val: Any) -> float:
    """Cleans numeric values by stripping currency symbols, commas, and whitespace."""
    if pd.isna(val) or val is None or val == "":
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    # Strip symbols
    clean_str = re.sub(r"[^\d.-]", "", str(val).strip())
    try:
        return float(clean_str)
    except (ValueError, TypeError):
        return np.nan

def clean_date_string(date_val: Any) -> str:
    """Parses date string safely into ISO format YYYY-MM-DD or empty string."""
    if pd.isna(date_val) or date_val is None or str(date_val).strip() in ["", "nan", "NaT"]:
        return ""
    try:
        dt = pd.to_datetime(date_val, errors="coerce")
        if pd.isna(dt):
            return ""
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

def preprocess_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Cleans and preprocesses the MPLADS dataset.
    
    Returns:
        (clean_df, quarantine_df, cleaning_log)
    """
    cleaning_log = []
    
    def log_action(step: str, description: str, impact: str, status: str = "success"):
        cleaning_log.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "step": step,
            "description": description,
            "impact": impact,
            "status": status
        })

    raw_count = len(df)
    log_action("Ingestion", f"Received raw dataframe with {raw_count} records and {len(df.columns)} columns.", f"Initial dataset loaded.")

    work_df = df.copy()

    # 1. Deduplication of identical rows
    before_dup = len(work_df)
    work_df = work_df.drop_duplicates()
    dups_removed = before_dup - len(work_df)
    log_action("Deduplication", f"Checked for exact duplicate rows across all columns.", f"Removed {dups_removed} identical rows.")

    # 2. String Standardization & Whitespace Trimming
    text_cols = ["mp_name", "constituency", "state", "district", "implementing_agency", "work_category", "work_description", "status"]
    for col in text_cols:
        if col in work_df.columns:
            work_df[col] = work_df[col].astype(str).str.strip()
            # Standardize casing for categorical labels
            if col in ["state", "district", "work_category", "status"]:
                work_df[col] = work_df[col].str.title()
            elif col == "implementing_agency":
                # Clean multiple spaces
                work_df[col] = work_df[col].str.replace(r"\s+", " ", regex=True)
    log_action("Text Normalization", f"Trimmed whitespace and normalized casing across {len(text_cols)} text fields.", "Uniformity established.")

    # 3. Numeric Normalization
    numeric_cols = ["sanctioned_amount", "estimated_cost", "authorized_amount", "expenditure", "progress_percent", "latitude", "longitude"]
    for col in numeric_cols:
        if col in work_df.columns:
            work_df[col] = work_df[col].apply(clean_numeric_value)
    log_action("Numeric Formatting", f"Stripped currency symbols, commas, and parsed numeric values.", f"Cleaned {len(numeric_cols)} numeric attributes.")

    # 4. Date Normalization
    date_cols = ["sanction_date", "expected_completion_date", "actual_completion_date"]
    for col in date_cols:
        if col in work_df.columns:
            work_df[col] = work_df[col].apply(clean_date_string)
    log_action("Date Standardization", f"Standardized sanction and completion dates to ISO format (YYYY-MM-DD).", "Date formats aligned.")

    # 5. Validation and Quarantine Isolation
    report, validated_df = validate_dataset(work_df)
    is_valid_mask = validated_df["is_record_valid"].values
    
    quarantine_df = work_df[~is_valid_mask].copy()
    clean_df = work_df[is_valid_mask].copy()
    
    quarantine_count = len(quarantine_df)
    log_action("Data Quarantine", f"Screened dataset for fatal violations (negative values, inverted dates, progress > 100%).", 
               f"Isolated {quarantine_count} corrupt records to quarantine; retained {len(clean_df)} valid records.")

    # 6. Missing Value Imputation on Cleaned Records
    # Numeric: Median imputation
    for num_col in ["estimated_cost", "sanctioned_amount", "authorized_amount", "expenditure"]:
        if num_col in clean_df.columns:
            missing_count = clean_df[num_col].isna().sum()
            if missing_count > 0:
                median_val = clean_df[num_col].median()
                clean_df[num_col] = clean_df[num_col].fillna(median_val)
                log_action("Numeric Imputation", f"Imputed {missing_count} missing values in '{num_col}' with cohort median ({median_val:.2f}).", f"Complete coverage in {num_col}.")

    # Progress percent default
    if "progress_percent" in clean_df.columns:
        missing_prog = clean_df["progress_percent"].isna().sum()
        if missing_prog > 0:
            clean_df["progress_percent"] = clean_df["progress_percent"].fillna(0.0)
            log_action("Progress Default", f"Set {missing_prog} null progress values to 0.0%.", "Progress initialized.")

    # Categorical: Mode or 'Unknown'
    cat_imputations = {
        "work_category": "General Public Infrastructure",
        "implementing_agency": "District Implementing Agency",
        "status": "Ongoing"
    }
    for cat_col, default_str in cat_imputations.items():
        if cat_col in clean_df.columns:
            null_cats = (clean_df[cat_col].isna()) | (clean_df[cat_col] == "") | (clean_df[cat_col].str.lower() == "nan")
            cat_count = null_cats.sum()
            if cat_count > 0:
                mode_val = clean_df[cat_col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 and mode_val.iloc[0] != "" else default_str
                clean_df.loc[null_cats, cat_col] = fill_val
                log_action("Categorical Imputation", f"Imputed {cat_count} blank entries in '{cat_col}' with '{fill_val}'.", f"Standardized {cat_col}.")

    # Reset indices
    clean_df = clean_df.reset_index(drop=True)
    quarantine_df = quarantine_df.reset_index(drop=True)

    log_action("Cleaning Completion", f"Cleaning pipeline finished. Ready for quality scoring and feature engineering.", 
               f"{len(clean_df)} verified records ready.")

    return clean_df, quarantine_df, cleaning_log
