"""Data Validation Engine
Performs structural, type, and semantic consistency audits on MPLADS datasets.
Counts total, valid, and invalid records, flags missing required fields,
negative amounts, inverted dates, and abnormal progress metrics.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from config import Config

def validate_dataset(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Validates a canonicalized MPLADS dataframe.
    
    Returns:
        (report_dict, row_validation_df)
        where row_validation_df contains boolean flag `is_record_valid` and `validation_issues` list.
    """
    total_records = len(df)
    if total_records == 0:
        return {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "validation_rate": 0.0,
            "field_summary": {},
            "issues": ["Dataset is empty."],
            "warnings": []
        }, pd.DataFrame()

    val_df = df.copy()
    issues_per_row = [[] for _ in range(total_records)]
    
    # 1. Missing required fields audit
    critical_fields = ["work_id", "state", "district", "sanctioned_amount", "expenditure", "sanction_date"]
    field_summary = {}
    
    for col in Config.CANONICAL_COLUMNS:
        if col in val_df.columns:
            null_count = int(val_df[col].isna().sum()) + int((val_df[col] == "").sum())
            null_pct = round((null_count / total_records) * 100, 2)
            field_summary[col] = {
                "null_count": null_count,
                "null_pct": null_pct,
                "present": True
            }
        else:
            field_summary[col] = {
                "null_count": total_records,
                "null_pct": 100.0,
                "present": False
            }

    for idx, row in val_df.iterrows():
        for fld in critical_fields:
            val = row.get(fld, None)
            if pd.isna(val) or val == "" or val is None:
                issues_per_row[idx].append(f"Missing required field: {fld}")

    # 2. Duplicate Work IDs check
    if "work_id" in val_df.columns:
        dup_ids = val_df[val_df["work_id"].duplicated(keep=False)]["work_id"].dropna().unique()
        for idx, row in val_df.iterrows():
            if row.get("work_id") in dup_ids:
                issues_per_row[idx].append("Duplicate work_id identifier")

    # 3. Numeric validity (negative amounts)
    for col in ["sanctioned_amount", "estimated_cost", "authorized_amount", "expenditure"]:
        if col in val_df.columns:
            numeric_vals = pd.to_numeric(val_df[col], errors="coerce")
            neg_indices = val_df[numeric_vals < 0].index
            for idx in neg_indices:
                issues_per_row[idx].append(f"Negative value in {col}: {val_df.loc[idx, col]}")

    # 4. Progress percentage validity (0 <= progress <= 100)
    if "progress_percent" in val_df.columns:
        prog_vals = pd.to_numeric(val_df["progress_percent"], errors="coerce")
        invalid_prog = val_df[(prog_vals < 0) | (prog_vals > 100)].index
        for idx in invalid_prog:
            issues_per_row[idx].append(f"Invalid progress_percent ({val_df.loc[idx, 'progress_percent']}%): must be 0-100")

    # 5. Chronological validity: expected_completion_date >= sanction_date
    if "sanction_date" in val_df.columns and "expected_completion_date" in val_df.columns:
        s_dates = pd.to_datetime(val_df["sanction_date"], errors="coerce")
        e_dates = pd.to_datetime(val_df["expected_completion_date"], errors="coerce")
        
        # Check invalid date parse
        unparseable_s = val_df[val_df["sanction_date"].notna() & s_dates.isna()].index
        for idx in unparseable_s:
            issues_per_row[idx].append("Unparseable sanction_date format")
            
        unparseable_e = val_df[val_df["expected_completion_date"].notna() & e_dates.isna()].index
        for idx in unparseable_e:
            issues_per_row[idx].append("Unparseable expected_completion_date format")
            
        # Check inverted dates
        inverted = val_df[s_dates.notna() & e_dates.notna() & (e_dates < s_dates)].index
        for idx in inverted:
            issues_per_row[idx].append("Chronological inversion: expected_completion_date before sanction_date")

    # Tag records
    val_df["validation_issues"] = issues_per_row
    val_df["is_record_valid"] = [len(issues) == 0 for issues in issues_per_row]
    
    valid_count = int(val_df["is_record_valid"].sum())
    invalid_count = total_records - valid_count
    val_rate = round((valid_count / total_records) * 100, 2)
    
    # Collect aggregate warnings
    warnings = []
    if invalid_count > 0:
        warnings.append(f"{invalid_count} records ({round((invalid_count/total_records)*100, 1)}%) have structural or logical validation errors.")
    if field_summary.get("sanctioned_amount", {}).get("null_count", 0) > 0:
        warnings.append("Some records are missing sanctioned_amount values.")
    if len(val_df[val_df["status"].isna()]) > 0:
        warnings.append("Some records have undefined project execution status.")

    report = {
        "total_records": total_records,
        "valid_records": valid_count,
        "invalid_records": invalid_count,
        "validation_rate": val_rate,
        "field_summary": field_summary,
        "warnings": warnings,
        "issue_types_breakdown": pd.Series([issue for sublist in issues_per_row for issue in sublist]).value_counts().to_dict()
    }
    
    return report, val_df
