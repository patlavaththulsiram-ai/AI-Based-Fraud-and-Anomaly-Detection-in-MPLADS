"""Composite Risk Scoring & Explainability Engine
Combines multi-model scores (Isolation Forest, One-Class SVM, Random Forest)
and deterministic policy rules into an explainable 0-100 Risk Score.
Assigns risk tiers (Low, Medium, High, Critical), computes granular risk breakdowns,
and provides objective, plain-language 'Why Flagged' forensic explanations.

CRITICAL POLICY: All findings are framed strictly as 'risk indicator requiring verification',
never as 'fraud confirmed'.
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
from config import Config
from src.rules import RULE_METADATA

def get_risk_level(score: float, thresholds: Dict[str, Tuple[int, int]] = None) -> str:
    """Classifies 0-100 score into Low, Medium, High, or Critical tier."""
    if thresholds is None:
        thresholds = Config.RISK_THRESHOLDS
        
    s = float(score)
    if s <= thresholds["LOW"][1]:
        return "Low"
    elif s <= thresholds["MEDIUM"][1]:
        return "Medium"
    elif s <= thresholds["HIGH"][1]:
        return "High"
    else:
        return "Critical"

def generate_plain_language_reasons(row: pd.Series) -> List[str]:
    """
    Generates plain-language, legally defensible explanation bullets for flagged works.
    Mandatory compliance: Frames all points as risk indicators requiring verification.
    """
    reasons = []
    
    # Check triggered deterministic rules
    for rule_col, meta in RULE_METADATA.items():
        if row.get(rule_col, 0) == 1:
            if rule_col == "rule_expenditure_gt_authorized":
                exp = row.get("expenditure", 0)
                auth = row.get("authorized_amount", 0)
                reasons.append(
                    f"Financial risk indicator requiring verification: Reported expenditure (₹{exp:.2f}L) "
                    f"exceeds authorized funds released (₹{auth:.2f}L) by ₹{exp - auth:.2f}L."
                )
            elif rule_col == "rule_chronic_delay":
                delay = int(row.get("delay_days", 0))
                reasons.append(
                    f"Implementation timeline indicator requiring verification: Project target date breached "
                    f"with an accumulated delay of {delay} days."
                )
            elif rule_col == "rule_progress_spend_mismatch":
                util = row.get("sanction_utilization", 0) * 100
                prog = row.get("progress_percent", 0)
                reasons.append(
                    f"Milestone mismatch indicator requiring verification: Financial disbursal is at {util:.1f}% "
                    f"while recorded physical progress is only {prog:.1f}%."
                )
            elif rule_col == "rule_cost_escalation":
                overrun = row.get("cost_overrun_pct", 0) * 100
                reasons.append(
                    f"Budget variance indicator requiring verification: Total spending exceeds original "
                    f"engineering cost estimate by {overrun:.1f}%."
                )
            elif rule_col == "rule_unusually_long_duration":
                reasons.append(
                    "Workflow duration indicator requiring verification: Execution timeline spans significantly "
                    "beyond standard statutory durations for similar civic works."
                )
            elif rule_col == "rule_premature_completion":
                prog = row.get("progress_percent", 0)
                reasons.append(
                    f"Completion claim indicator requiring verification: Project is marked 'Completed' despite "
                    f"only {prog:.1f}% physical ground progress being logged."
                )
            elif rule_col == "rule_suspicious_round_amount":
                exp = row.get("expenditure", 0)
                reasons.append(
                    f"Procurement threshold indicator requiring verification: Disbursed value of ₹{exp:.2f}L "
                    f"coincides with round fiscal limits or tender-splitting bounds."
                )
            elif rule_col == "rule_invalid_financial_data":
                reasons.append(
                    "Data integrity indicator requiring verification: Detected negative or zero financial values."
                )

    # Duplicate work indicator
    if row.get("duplicate_risk_flag", 0) == 1:
        reasons.append(
            "Duplicate tender indicator requiring verification: High lexical and financial similarity "
            "detected with another civil work in the same administrative district."
        )

    # Machine learning anomaly confidence
    rf_prob = float(row.get("rf_anomaly_prob", 0.0))
    if rf_prob >= 75.0:
        reasons.append(
            f"Supervised model risk indicator requiring verification: High pattern concordance ({rf_prob:.1f}%) "
            f"with historical audit discrepancy patterns."
        )
    elif float(row.get("if_anomaly_score", 0.0)) >= 80.0:
        reasons.append(
            "Statistical outlier indicator requiring verification: Project features deviate significantly "
            "from normative baseline distributions (Isolation Forest)."
        )

    # Fallback for clean works
    if not reasons:
        reasons.append("Normative parameters: Project implementation metrics conform to expected MPLADS execution bounds.")

    return reasons

def compute_project_risk_scores(
    df: pd.DataFrame,
    model_scores: Dict[str, np.ndarray],
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Computes continuous 0-100 Risk Scores, categorical levels, sub-dimension breakdowns,
    and plain-language verification indicators.
    """
    if weights is None:
        weights = {
            "if": Config.WEIGHT_ISOLATION_FOREST,
            "svm": Config.WEIGHT_ONE_CLASS_SVM,
            "rf": Config.WEIGHT_RANDOM_FOREST,
            "rules": Config.WEIGHT_RULES
        }

    scored_df = df.copy()

    # Assign individual component scores
    scored_df["if_anomaly_score"] = model_scores["if_score"]
    scored_df["svm_anomaly_score"] = model_scores["svm_score"]
    scored_df["rf_anomaly_prob"] = model_scores["rf_prob"]

    # Rule score (0 to 100 based on number of triggered rules and severity)
    # 1 rule = ~25, 2 = ~50, 3 = ~75, 4+ = 100
    rule_counts = scored_df.get("total_rules_triggered", pd.Series(0, index=scored_df.index))
    rule_score = np.clip(rule_counts * 25.0, 0.0, 100.0)
    scored_df["rule_anomaly_score"] = rule_score.round(2)

    # Composite Risk Score Calculation (0-100)
    composite = (
        (weights["if"] * scored_df["if_anomaly_score"]) +
        (weights["svm"] * scored_df["svm_anomaly_score"]) +
        (weights["rf"] * scored_df["rf_anomaly_prob"]) +
        (weights["rules"] * scored_df["rule_anomaly_score"])
    )
    scored_df["risk_score"] = np.clip(composite, 0.0, 100.0).round(1)

    # Classify Risk Tier
    scored_df["risk_level"] = scored_df["risk_score"].apply(get_risk_level)

    # Compute Sub-dimension Risk Breakdowns (0-100 each)
    # 1. Financial Risk: driven by cost overrun, expenditure > auth, and utilization
    fin_risk = (
        np.clip(scored_df.get("cost_overrun_pct", 0) * 100, 0, 100) * 0.4 +
        scored_df.get("rule_expenditure_gt_authorized", 0) * 40.0 +
        scored_df.get("rule_suspicious_round_amount", 0) * 20.0
    )
    scored_df["financial_risk"] = np.clip(fin_risk, 0.0, 100.0).round(1)

    # 2. Delay Risk: driven by delay_days and duration rules
    delays = scored_df.get("delay_days", 0)
    delay_scaled = np.clip((delays / 365.0) * 100.0, 0, 100)
    scored_df["delay_risk"] = np.clip(
        delay_scaled * 0.6 + scored_df.get("rule_chronic_delay", 0) * 40.0,
        0.0, 100.0
    ).round(1)

    # 3. Anomaly Risk: driven by IF & SVM unsupervised models
    scored_df["anomaly_risk"] = (
        (scored_df["if_anomaly_score"] * 0.55) +
        (scored_df["svm_anomaly_score"] * 0.45)
    ).round(1)

    # 4. Duplicate Risk: driven by TF-IDF duplicate flag
    scored_df["duplicate_risk"] = (scored_df.get("duplicate_risk_flag", 0) * 85.0).round(1)

    # 5. Progress Risk: driven by progress vs spend gap and premature completion
    gap = scored_df.get("progress_vs_spend_gap", 0)
    gap_scaled = np.clip(gap * 100.0, 0, 100)
    scored_df["progress_risk"] = np.clip(
        gap_scaled * 0.5 + scored_df.get("rule_premature_completion", 0) * 50.0,
        0.0, 100.0
    ).round(1)

    # 6. Data Quality Risk
    quality_risk = (scored_df.get("rule_invalid_financial_data", 0) * 80.0)
    scored_df["data_quality_risk"] = np.clip(quality_risk, 0.0, 100.0).round(1)

    # Generate Explainable Reasons
    reasons_list = []
    for idx, row in scored_df.iterrows():
        reasons_list.append(generate_plain_language_reasons(row))
    scored_df["risk_reasons"] = reasons_list

    return scored_df
