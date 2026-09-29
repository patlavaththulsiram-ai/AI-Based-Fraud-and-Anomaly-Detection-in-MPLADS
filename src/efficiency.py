"""Efficiency & Implementation Bottleneck Analysis Engine
Analyzes administrative velocity, project lifecycle delays, completion trajectories,
and physical-financial alignment across states, districts, and time periods.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

def analyze_efficiency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes key performance indicators for administrative efficiency and implementation bottlenecks.
    """
    total = len(df)
    if total == 0:
        return {}

    # 1. Project counts by execution status
    status_counts = df["status"].value_counts().to_dict()
    completed_count = int(status_counts.get("Completed", 0))
    ongoing_count = int(status_counts.get("Ongoing", 0))
    delayed_count = int(status_counts.get("Delayed", 0))
    pending_count = int(status_counts.get("Pending", 0))

    # 2. Delays metrics
    delays = df["delay_days"].fillna(0)
    avg_completion_delay = round(float(delays[delays > 0].mean()), 1) if (delays > 0).sum() > 0 else 0.0
    max_delay = int(delays.max()) if len(delays) > 0 else 0
    
    # Planned vs Actual duration for completed works
    completed_df = df[df["status"].str.lower() == "completed"]
    if len(completed_df) > 0:
        avg_planned_duration = round(float(completed_df["planned_duration_days"].mean()), 1)
        avg_actual_duration = round(float(completed_df["actual_duration_days"].mean()), 1)
        avg_duration_variance = round(avg_actual_duration - avg_planned_duration, 1)
    else:
        avg_planned_duration = round(float(df["planned_duration_days"].mean()), 1)
        avg_actual_duration = round(float(df["actual_duration_days"].mean()), 1)
        avg_duration_variance = 0.0

    # 3. Delay by State
    state_efficiency = []
    if "state" in df.columns:
        for state, grp in df.groupby("state"):
            s_total = len(grp)
            s_delayed = int((grp["status"].str.lower() == "delayed").sum())
            s_avg_delay = round(float(grp["delay_days"].mean()), 1)
            s_utilization = round(float(grp.get("sanction_utilization", pd.Series(0)).mean()) * 100, 1)
            s_avg_gap = round(float(grp.get("progress_vs_spend_gap", pd.Series(0)).mean()) * 100, 1)
            
            state_efficiency.append({
                "state": state,
                "total_projects": s_total,
                "delayed_count": s_delayed,
                "delay_rate_pct": round((s_delayed / s_total) * 100, 1),
                "avg_delay_days": s_avg_delay,
                "avg_utilization_pct": s_utilization,
                "avg_spend_progress_gap_pct": s_avg_gap
            })
        # Sort by delay rate descending
        state_efficiency.sort(key=lambda x: x["delay_rate_pct"], reverse=True)

    # 4. Completion Trend by Year
    yearly_trend = []
    s_dt = pd.to_datetime(df["sanction_date"], errors="coerce")
    valid_years = s_dt.dropna().dt.year.astype(int)
    unique_years = sorted(valid_years.unique())
    
    for yr in unique_years:
        mask = (valid_years == yr)
        sub = df[mask]
        y_total = len(sub)
        y_comp = int((sub["status"].str.lower() == "completed").sum())
        y_spend = round(float(sub["expenditure"].sum()), 2)
        y_sanc = round(float(sub["sanctioned_amount"].sum()), 2)
        yearly_trend.append({
            "year": int(yr),
            "sanctioned_projects": y_total,
            "completed_projects": y_comp,
            "completion_rate_pct": round((y_comp / max(1, y_total)) * 100, 1),
            "total_sanctioned_lakhs": y_sanc,
            "total_expenditure_lakhs": y_spend
        })

    # 5. Spend vs Progress Gap Distribution
    gap = df.get("progress_vs_spend_gap", pd.Series(0.0, index=df.index))
    gap_distribution = {
        "balanced": int(((gap >= -0.10) & (gap <= 0.10)).sum()),
        "moderate_lead": int(((gap > 0.10) & (gap <= 0.30)).sum()),
        "severe_lead": int((gap > 0.30).sum()),
        "favorable_progress": int((gap < -0.10).sum())
    }

    return {
        "status_distribution": {
            "completed": completed_count,
            "ongoing": ongoing_count,
            "delayed": delayed_count,
            "pending": pending_count
        },
        "delay_metrics": {
            "avg_completion_delay_days": avg_completion_delay,
            "max_delay_days": max_delay,
            "avg_planned_duration_days": avg_planned_duration,
            "avg_actual_duration_days": avg_actual_duration,
            "avg_duration_variance_days": avg_duration_variance
        },
        "state_efficiency": state_efficiency,
        "yearly_trend": yearly_trend,
        "gap_distribution": gap_distribution
    }
