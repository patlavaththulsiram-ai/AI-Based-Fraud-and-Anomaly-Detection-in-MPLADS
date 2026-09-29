"""Power BI Dataset Export Engine
Generates clean, standardized, UTF-8 encoded CSV tables optimized for direct
ingestion and relationship mapping inside Microsoft Power BI Desktop and Service.
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

def export_powerbi_datasets(scored_df: pd.DataFrame) -> Dict[str, str]:
    """
    Generates and saves the four required Power BI CSV exports to exports/ directory.
    
    Returns:
        dictionary mapping export name to absolute file path string.
    """
    Config.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = scored_df.copy()
    
    # 1. projects_scored.csv
    # Stable column selection for Power BI fact table
    export_cols = [
        "work_id", "mp_name", "constituency", "state", "district",
        "implementing_agency", "work_category", "work_description",
        "sanctioned_amount", "estimated_cost", "authorized_amount", "expenditure",
        "sanction_date", "expected_completion_date", "actual_completion_date",
        "status", "progress_percent", "latitude", "longitude",
        "sanction_utilization", "cost_overrun_pct", "planned_duration_days",
        "actual_duration_days", "delay_days", "progress_vs_spend_gap",
        "if_anomaly_score", "svm_anomaly_score", "rf_anomaly_prob",
        "rule_anomaly_score", "risk_score", "risk_level",
        "financial_risk", "delay_risk", "anomaly_risk", "duplicate_risk",
        "progress_risk", "data_quality_risk", "total_rules_triggered",
        "duplicate_risk_flag"
    ]
    present_cols = [c for c in export_cols if c in df.columns]
    projects_csv_path = Config.EXPORTS_DIR / "projects_scored.csv"
    df[present_cols].to_csv(projects_csv_path, index=False, encoding="utf-8-sig")

    # 2. anomalies.csv (Projects with Risk Level Medium, High, or Critical, or rules triggered)
    anomalies_mask = (df["risk_score"] >= 40.0) | (df.get("total_rules_triggered", 0) > 0)
    if "is_anomaly" in df.columns:
        anomalies_mask = anomalies_mask | (df["is_anomaly"] == 1)
    anomalies_df = df[anomalies_mask].copy()
    anomalies_csv_path = Config.EXPORTS_DIR / "anomalies.csv"
    anomalies_df[present_cols].to_csv(anomalies_csv_path, index=False, encoding="utf-8-sig")

    # 3. district_summary.csv (Aggregated dimension table for geospatial mapping)
    district_summary = df.groupby(["state", "district"]).agg(
        total_projects=("work_id", "count"),
        total_sanctioned_lakhs=("sanctioned_amount", "sum"),
        total_expenditure_lakhs=("expenditure", "sum"),
        avg_sanction_utilization=("sanction_utilization", "mean"),
        avg_risk_score=("risk_score", "mean"),
        critical_risk_projects=("risk_level", lambda s: (s == "Critical").sum()),
        high_risk_projects=("risk_level", lambda s: (s == "High").sum()),
        delayed_projects=("status", lambda s: (s.str.lower() == "delayed").sum()),
        avg_delay_days=("delay_days", "mean")
    ).reset_index()

    district_summary["total_sanctioned_lakhs"] = district_summary["total_sanctioned_lakhs"].round(2)
    district_summary["total_expenditure_lakhs"] = district_summary["total_expenditure_lakhs"].round(2)
    district_summary["avg_sanction_utilization"] = (district_summary["avg_sanction_utilization"] * 100).round(1)
    district_summary["avg_risk_score"] = district_summary["avg_risk_score"].round(1)
    district_summary["avg_delay_days"] = district_summary["avg_delay_days"].round(1)

    district_csv_path = Config.EXPORTS_DIR / "district_summary.csv"
    district_summary.to_csv(district_csv_path, index=False, encoding="utf-8-sig")

    # 4. alerts.csv (Triage operational table for Power BI cards & drillthroughs)
    alerts = []
    for idx, row in anomalies_df.iterrows():
        score = row.get("risk_score", 0.0)
        level = row.get("risk_level", "Medium")
        reasons = row.get("risk_reasons", ["Risk indicator requiring verification"])
        top_reason = reasons[0] if isinstance(reasons, list) and len(reasons) > 0 else "Risk indicator requiring verification"
        
        alerts.append({
            "alert_id": f"ALT-{row.get('work_id', idx)}",
            "work_id": row.get("work_id", ""),
            "date_flagged": row.get("sanction_date", "2024-01-01"),
            "mp_name": row.get("mp_name", ""),
            "constituency": row.get("constituency", ""),
            "district": row.get("district", ""),
            "state": row.get("state", ""),
            "work_category": row.get("work_category", ""),
            "expenditure_lakhs": row.get("expenditure", 0.0),
            "severity": level,
            "risk_score": score,
            "status": "Open",
            "primary_indicator": top_reason
        })
    alerts_df = pd.DataFrame(alerts)
    alerts_csv_path = Config.EXPORTS_DIR / "alerts.csv"
    alerts_df.to_csv(alerts_csv_path, index=False, encoding="utf-8-sig")

    print(f"Power BI datasets successfully exported to {Config.EXPORTS_DIR}")
    return {
        "projects_scored": str(projects_csv_path),
        "anomalies": str(anomalies_csv_path),
        "district_summary": str(district_csv_path),
        "alerts": str(alerts_csv_path)
    }
