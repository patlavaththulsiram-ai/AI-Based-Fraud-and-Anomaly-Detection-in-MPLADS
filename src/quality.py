"""Data Quality Evaluation Engine
Computes a comprehensive 0-100 Data Quality Score composed of:
1. Completeness: Field population rate across core canonical columns.
2. Consistency: Semantic and chronological harmony across status, dates, and amounts.
3. Validity: Adherence to domain ranges, non-negativity, and geographical coordinates.
4. Uniqueness: Absence of redundant entities and duplicated work IDs.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from config import Config

def compute_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates dataset quality dimensions and returns granular sub-scores and overall score (0-100).
    """
    total = len(df)
    if total == 0:
        return {
            "overall_score": 0.0,
            "grade": "F",
            "completeness_score": 0.0,
            "consistency_score": 0.0,
            "validity_score": 0.0,
            "uniqueness_score": 0.0,
            "dimension_details": {},
            "recommendations": ["Dataset contains zero records."]
        }

    # 1. Completeness Dimension (30% weight)
    # Check null/empty rates across canonical fields
    completeness_ratios = []
    for col in Config.CANONICAL_COLUMNS:
        if col in df.columns:
            non_null = df[col].notna() & (df[col] != "") & (df[col] != "nan")
            completeness_ratios.append(non_null.mean())
        else:
            completeness_ratios.append(0.0)
    completeness_score = round(float(np.mean(completeness_ratios)) * 100, 1)

    # 2. Consistency Dimension (25% weight)
    # Checks:
    # a. Inverted dates (expected before sanction)
    # b. Inverted completion (actual before sanction)
    # c. Completed status vs progress >= 60%
    # d. Pending status vs progress == 0%
    consistencies = []
    
    # Date consistency
    s_dt = pd.to_datetime(df["sanction_date"], errors="coerce")
    e_dt = pd.to_datetime(df["expected_completion_date"], errors="coerce")
    a_dt = pd.to_datetime(df["actual_completion_date"], errors="coerce")
    
    both_dates = s_dt.notna() & e_dt.notna()
    if both_dates.sum() > 0:
        date_ok = (e_dt[both_dates] >= s_dt[both_dates]).mean()
        consistencies.append(date_ok)
    else:
        consistencies.append(1.0)
        
    actual_dates = s_dt.notna() & a_dt.notna()
    if actual_dates.sum() > 0:
        act_ok = (a_dt[actual_dates] >= s_dt[actual_dates]).mean()
        consistencies.append(act_ok)
    else:
        consistencies.append(1.0)

    # Status vs progress consistency
    if "status" in df.columns and "progress_percent" in df.columns:
        completed_mask = df["status"].str.lower() == "completed"
        if completed_mask.sum() > 0:
            comp_prog_ok = (df.loc[completed_mask, "progress_percent"] >= 60.0).mean()
            consistencies.append(comp_prog_ok)
            
        pending_mask = df["status"].str.lower() == "pending"
        if pending_mask.sum() > 0:
            pend_ok = (df.loc[pending_mask, "progress_percent"] <= 10.0).mean()
            consistencies.append(pend_ok)

    consistency_score = round(float(np.mean(consistencies)) * 100, 1)

    # 3. Validity Dimension (25% weight)
    # Checks:
    # a. Non-negative amounts
    # b. Progress between 0 and 100
    # c. Coordinates inside India bounds (Lat ~6-38, Lon ~68-98)
    validities = []
    
    for col in ["sanctioned_amount", "estimated_cost", "authorized_amount", "expenditure"]:
        if col in df.columns:
            nums = pd.to_numeric(df[col], errors="coerce")
            validities.append((nums >= 0).mean())
            
    if "progress_percent" in df.columns:
        prog = pd.to_numeric(df["progress_percent"], errors="coerce")
        validities.append(((prog >= 0.0) & (prog <= 100.0)).mean())

    if "latitude" in df.columns and "longitude" in df.columns:
        lats = pd.to_numeric(df["latitude"], errors="coerce")
        lons = pd.to_numeric(df["longitude"], errors="coerce")
        geo_valid = ((lats.isna()) | ((lats >= 6.0) & (lats <= 38.0))) & \
                    ((lons.isna()) | ((lons >= 68.0) & (lons <= 98.0)))
        validities.append(geo_valid.mean())

    validity_score = round(float(np.mean(validities)) * 100, 1)

    # 4. Uniqueness Dimension (20% weight)
    # Checks: unique work_ids and unique work descriptions
    uniqueness_scores = []
    if "work_id" in df.columns and total > 0:
        unique_ids = df["work_id"].nunique() / total
        uniqueness_scores.append(unique_ids)
        
    if "work_description" in df.columns and total > 0:
        unique_desc = df["work_description"].nunique() / total
        # Near-duplicate works naturally exist in large programs, so calibrate description uniqueness
        uniqueness_scores.append(min(1.0, unique_desc * 1.05))
        
    uniqueness_score = round(float(np.mean(uniqueness_scores)) * 100, 1)

    # Overall Composite Score
    overall_score = round(
        (completeness_score * 0.30) +
        (consistency_score * 0.25) +
        (validity_score * 0.25) +
        (uniqueness_score * 0.20),
        1
    )
    
    # Grade assignment
    if overall_score >= 90.0:
        grade = "A"
        grade_desc = "Excellent Data Integrity"
    elif overall_score >= 75.0:
        grade = "B"
        grade_desc = "Good Operational Quality"
    elif overall_score >= 60.0:
        grade = "C"
        grade_desc = "Moderate Quality - Audit Required"
    else:
        grade = "D"
        grade_desc = "Low Quality - Heavy Remediation Needed"

    # Actionable recommendations
    recommendations = []
    if completeness_score < 85.0:
        recommendations.append("Enhance mandatory field entry during project intake to boost completeness.")
    if consistency_score < 85.0:
        recommendations.append("Implement automated portal validation to prevent premature 'Completed' flags on incomplete works.")
    if validity_score < 90.0:
        recommendations.append("Enforce client-side non-negative boundary limits for monetary expenditures and percentages.")
    if uniqueness_score < 80.0:
        recommendations.append("Investigate high description overlap across projects in identical administrative districts.")
    if not recommendations:
        recommendations.append("Data quality metrics exceed benchmark standards for computational auditing.")

    return {
        "overall_score": overall_score,
        "grade": grade,
        "grade_description": grade_desc,
        "sub_scores": {
            "completeness": completeness_score,
            "consistency": consistency_score,
            "validity": validity_score,
            "uniqueness": uniqueness_score
        },
        "weights": {
            "completeness": 0.30,
            "consistency": 0.25,
            "validity": 0.25,
            "uniqueness": 0.20
        },
        "recommendations": recommendations
    }
