"""Feature Engineering Engine
Extracts financial, operational, temporal, and risk-sensitive attributes from cleaned MPLADS records.
Constructs both tabular feature frames for reporting and scaled numerical matrices for ML models.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from config import Config

# Feature definitions for ML model training and inference
ML_NUMERICAL_FEATURES = [
    "expenditure",
    "sanction_utilization",
    "cost_overrun_pct",
    "planned_duration_days",
    "actual_duration_days",
    "delay_days",
    "progress_vs_spend_gap",
    "completion_status_encoded",
    "category_encoded",
    "agency_work_count",
    "expenditure_zscore_within_category",
    "round_number_flag"
]

def check_round_number(val: float) -> int:
    """Detects suspicious round disbursals (e.g. 5.0, 10.0, 25.0, 50.0 or just-below thresholds)."""
    if pd.isna(val) or val <= 0:
        return 0
    # Exact round Lakhs
    if val in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]:
        return 1
    # Check just below limits (e.g., 4.99, 9.98, 24.95, 49.90)
    for threshold in [5.0, 10.0, 25.0, 50.0]:
        diff = threshold - val
        if 0.0 < diff <= 0.15:
            return 1
    # Integer check
    if abs(val - round(val)) < 0.001 and val >= 5.0:
        return 1
    return 0

def engineer_features(df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = True) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Extracts all engineered features from preprocessed MPLADS records.
    
    Returns:
        (featured_df, scaled_feature_matrix, fitted_scaler)
    """
    feat_df = df.copy()
    ref_date = datetime(2026, 3, 1)

    # 1. Financial Ratios
    # Sanction Utilization: expenditure / sanctioned_amount
    sanctioned = feat_df["sanctioned_amount"].replace(0, np.nan)
    feat_df["sanction_utilization"] = (feat_df["expenditure"] / sanctioned).fillna(0.0).round(4)
    # Clip extreme outliers to reasonable bounds for stability
    feat_df["sanction_utilization"] = feat_df["sanction_utilization"].clip(0.0, 5.0)

    # Cost Overrun Pct: (expenditure - estimated_cost) / estimated_cost
    estimated = feat_df["estimated_cost"].replace(0, np.nan)
    feat_df["cost_overrun_pct"] = ((feat_df["expenditure"] - feat_df["estimated_cost"]) / estimated).fillna(0.0).round(4)
    feat_df["cost_overrun_pct"] = feat_df["cost_overrun_pct"].clip(-1.0, 5.0)

    # 2. Temporal & Duration Metrics
    s_dt = pd.to_datetime(feat_df["sanction_date"], errors="coerce")
    e_dt = pd.to_datetime(feat_df["expected_completion_date"], errors="coerce")
    a_dt = pd.to_datetime(feat_df["actual_completion_date"], errors="coerce")

    # Planned Duration in Days
    planned_days = (e_dt - s_dt).dt.days
    feat_df["planned_duration_days"] = planned_days.fillna(365).clip(lower=30, upper=2500)

    # Actual / Elapsed Duration in Days
    # If completed, actual_completion - sanction_date; else ref_date - sanction_date
    actual_days = []
    for idx, row in feat_df.iterrows():
        act_date = a_dt.iloc[idx]
        sanc_date = s_dt.iloc[idx]
        if pd.notna(act_date) and pd.notna(sanc_date):
            diff = (act_date - sanc_date).days
        elif pd.notna(sanc_date):
            diff = (ref_date - sanc_date).days
        else:
            diff = 365
        actual_days.append(max(0, diff))
    feat_df["actual_duration_days"] = actual_days

    # Delay Days: max(0, completion_date_or_ref_date - expected_completion_date)
    delays = []
    for idx, row in feat_df.iterrows():
        exp_date = e_dt.iloc[idx]
        act_date = a_dt.iloc[idx]
        status = str(row.get("status", "")).lower()
        
        if pd.isna(exp_date):
            delays.append(0)
            continue
            
        if pd.notna(act_date):
            diff = (act_date - exp_date).days
            delays.append(max(0, diff))
        else:
            diff = (ref_date - exp_date).days
            if status in ["delayed", "ongoing"] and diff > 0:
                delays.append(diff)
            else:
                delays.append(0)
    feat_df["delay_days"] = delays

    # 3. Execution Gap: Progress vs Spend
    # progress_vs_spend_gap = utilization - (progress / 100.0)
    prog_ratio = (feat_df["progress_percent"].fillna(0.0) / 100.0)
    feat_df["progress_vs_spend_gap"] = (feat_df["sanction_utilization"] - prog_ratio).round(4)

    # 4. Frequency & Categorical Encodings
    # Agency work count
    agency_counts = feat_df["implementing_agency"].value_counts().to_dict()
    feat_df["agency_work_count"] = feat_df["implementing_agency"].map(agency_counts).fillna(1)

    # Status encoding: Completed=1, Ongoing=2, Delayed=3, Pending=4, other=0
    status_map = {
        "completed": 1,
        "ongoing": 2,
        "delayed": 3,
        "pending": 4
    }
    feat_df["completion_status_encoded"] = feat_df["status"].str.lower().map(status_map).fillna(0).astype(int)

    # Category encoding
    cat_order = ["Education", "Health", "Roads", "Water", "Electricity", "Community Assets"]
    cat_map = {cat.lower(): i + 1 for i, cat in enumerate(cat_order)}
    feat_df["category_encoded"] = feat_df["work_category"].str.lower().map(cat_map).fillna(0).astype(int)

    # Expenditure Z-Score within Category
    cat_stats = feat_df.groupby("work_category")["expenditure"].agg(["mean", "std"]).to_dict("index")
    zscores = []
    for idx, row in feat_df.iterrows():
        cat = row.get("work_category", "")
        exp = row.get("expenditure", 0.0)
        stats = cat_stats.get(cat, {"mean": 20.0, "std": 10.0})
        mean = stats["mean"]
        std = stats["std"] if (stats["std"] is not None and stats["std"] > 0) else 1.0
        zscores.append(round((exp - mean) / std, 4))
    feat_df["expenditure_zscore_within_category"] = zscores

    # Suspicious Round Number Flag
    feat_df["round_number_flag"] = feat_df["expenditure"].apply(check_round_number)

    # 5. Extract and Scale ML Matrix
    X_raw = feat_df[ML_NUMERICAL_FEATURES].fillna(0.0).values
    
    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        # Save scaler to models_store
        scaler_path = Config.MODELS_STORE_DIR / "feature_scaler.joblib"
        joblib.dump(scaler, scaler_path)
    else:
        X_scaled = scaler.transform(X_raw)

    return feat_df, X_scaled, scaler
