"""Deterministic Rule-Based Anomaly & Policy Compliance Engine
Audits project records against statutory MPLADS implementation guidelines,
financial bounds, physical milestones, and timeline constraints.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

RULE_METADATA = {
    "rule_expenditure_gt_authorized": {
        "code": "R1",
        "name": "Expenditure Exceeds Authorized Allocation",
        "severity": "High",
        "description": "Cumulative disbursements exceed total funds released by the district authority."
    },
    "rule_chronic_delay": {
        "code": "R2",
        "name": "Chronic Timeline Delay",
        "severity": "Medium",
        "description": "Target completion date breached by more than 90 days without documented extension."
    },
    "rule_progress_spend_mismatch": {
        "code": "R3",
        "name": "Physical Progress vs Financial Outflow Mismatch",
        "severity": "Critical",
        "description": "Fund utilization exceeds 70% while physical ground progress remains below 30%."
    },
    "rule_cost_escalation": {
        "code": "R4",
        "name": "Significant Cost Escalation",
        "severity": "High",
        "description": "Expenditure exceeds initial engineering cost estimate by over 20%."
    },
    "rule_unusually_long_duration": {
        "code": "R5",
        "name": "Unusually Prolonged Workflow",
        "severity": "Medium",
        "description": "Planned duration exceeds 3 years or actual elapsed duration exceeds 3 times the planned schedule."
    },
    "rule_premature_completion": {
        "code": "R6",
        "name": "Premature Completion Declaration",
        "severity": "Critical",
        "description": "Work is officially marked as 'Completed' while physical progress is less than 60%."
    },
    "rule_suspicious_round_amount": {
        "code": "R7",
        "name": "Suspicious Disbursal Value Threshold",
        "severity": "Medium",
        "description": "Expenditure coincides with round transaction caps or just below statutory tender thresholds."
    },
    "rule_invalid_financial_data": {
        "code": "R8",
        "name": "Invalid or Negative Financial Entry",
        "severity": "High",
        "description": "Negative expenditure or cost figures, or inverted timeline chronology detected."
    }
}

def apply_rule_engine(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates rule criteria on featured MPLADS dataset.
    
    Returns:
        (df_with_rules, rule_summary_dict)
    """
    rdf = df.copy()
    
    # R1: Expenditure > Authorized Amount
    auth = rdf["authorized_amount"].fillna(0.0)
    exp = rdf["expenditure"].fillna(0.0)
    rdf["rule_expenditure_gt_authorized"] = (exp > (auth + 0.01)).astype(int)

    # R2: Chronic Delay (delay_days > 90 or status == 'Delayed')
    delays = rdf.get("delay_days", pd.Series(0, index=rdf.index))
    status_delayed = rdf["status"].astype(str).str.lower() == "delayed"
    rdf["rule_chronic_delay"] = ((delays > 90) | status_delayed).astype(int)

    # R3: Progress vs Spend Mismatch (Spend > 70% while Progress < 30%)
    utilization = rdf.get("sanction_utilization", pd.Series(0.0, index=rdf.index))
    prog = rdf["progress_percent"].fillna(0.0)
    rdf["rule_progress_spend_mismatch"] = ((utilization > 0.70) & (prog < 30.0)).astype(int)

    # R4: Cost Escalation (cost_overrun_pct > 0.20 or expenditure > estimated_cost)
    overrun = rdf.get("cost_overrun_pct", pd.Series(0.0, index=rdf.index))
    rdf["rule_cost_escalation"] = (overrun > 0.20).astype(int)

    # R5: Unusually Long Duration (> 1095 days or actual > 3x planned)
    planned = rdf.get("planned_duration_days", pd.Series(365, index=rdf.index))
    actual = rdf.get("actual_duration_days", pd.Series(365, index=rdf.index))
    rdf["rule_unusually_long_duration"] = ((planned > 1095) | (actual > (3 * planned))).astype(int)

    # R6: Premature Completion (Status == 'Completed' and progress < 60%)
    status_completed = rdf["status"].astype(str).str.lower() == "completed"
    rdf["rule_premature_completion"] = (status_completed & (prog < 60.0)).astype(int)

    # R7: Suspicious Round Disbursal Value
    round_flag = rdf.get("round_number_flag", pd.Series(0, index=rdf.index))
    rdf["rule_suspicious_round_amount"] = round_flag.astype(int)

    # R8: Invalid Financial Data (negative amounts or cost <= 0)
    est = rdf["estimated_cost"].fillna(0.0)
    sanc = rdf["sanctioned_amount"].fillna(0.0)
    rdf["rule_invalid_financial_data"] = ((exp < 0) | (est < 0) | (sanc < 0)).astype(int)

    # Total rules triggered per record
    rule_cols = list(RULE_METADATA.keys())
    rdf["total_rules_triggered"] = rdf[rule_cols].sum(axis=1)

    # Aggregate statistics
    rule_counts = {}
    for col, meta in RULE_METADATA.items():
        hit_count = int(rdf[col].sum())
        hit_pct = round((hit_count / max(1, len(rdf))) * 100, 2)
        rule_counts[col] = {
            "code": meta["code"],
            "name": meta["name"],
            "severity": meta["severity"],
            "description": meta["description"],
            "hit_count": hit_count,
            "hit_percentage": hit_pct
        }

    summary = {
        "total_audited_records": len(rdf),
        "records_with_any_violation": int((rdf["total_rules_triggered"] > 0).sum()),
        "violation_rate": round(((rdf["total_rules_triggered"] > 0).sum() / max(1, len(rdf))) * 100, 2),
        "rule_counts": rule_counts
    }

    return rdf, summary
