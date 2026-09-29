"""Unit Tests for Stage 3: Deterministic Rule Engine
"""

import pandas as pd
import pytest
from src.rules import apply_rule_engine, RULE_METADATA

def test_rule_engine_detections():
    """Verify that every deterministic policy rule triggers accurately on edge cases."""
    test_cases = pd.DataFrame({
        "work_id": ["R1-TEST", "R2-TEST", "R3-TEST", "R4-TEST", "R5-TEST", "R6-TEST", "R7-TEST", "R8-TEST", "NORMAL-TEST"],
        "authorized_amount": [20.0, 50.0, 50.0, 40.0, 30.0, 40.0, 25.0, 30.0, 50.0],
        "expenditure": [25.0, 40.0, 40.0, 55.0, 25.0, 35.0, 25.0, -10.0, 40.0], # R1: 25>20, R4: 55>40 (overrun), R8: -10
        "estimated_cost": [20.0, 50.0, 50.0, 40.0, 30.0, 40.0, 25.0, -10.0, 45.0],
        "sanctioned_amount": [20.0, 50.0, 50.0, 40.0, 30.0, 40.0, 25.0, 30.0, 50.0],
        "delay_days": [0, 150, 0, 0, 0, 0, 0, 0, 10], # R2: 150 > 90
        "status": ["Ongoing", "Delayed", "Ongoing", "Ongoing", "Ongoing", "Completed", "Ongoing", "Ongoing", "Completed"],
        "sanction_utilization": [1.25, 0.80, 0.80, 1.375, 0.83, 0.875, 1.0, 0.0, 0.80], # R3: util 0.80 > 0.70 with prog 20 < 30
        "progress_percent": [50.0, 40.0, 20.0, 80.0, 50.0, 45.0, 100.0, 0.0, 100.0], # R6: Completed with 45 < 60
        "cost_overrun_pct": [0.25, 0.0, 0.0, 0.375, 0.0, 0.0, 0.0, 0.0, -0.11],
        "planned_duration_days": [365, 365, 365, 365, 1200, 365, 365, 365, 365], # R5: 1200 > 1095
        "actual_duration_days": [200, 400, 200, 200, 1300, 300, 200, 100, 350],
        "round_number_flag": [0, 0, 0, 0, 0, 0, 1, 0, 0] # R7: 1
    })

    rdf, summary = apply_rule_engine(test_cases)

    # R1: Expenditure > Authorized
    assert rdf.loc[rdf["work_id"] == "R1-TEST", "rule_expenditure_gt_authorized"].iloc[0] == 1
    # R2: Chronic Delay
    assert rdf.loc[rdf["work_id"] == "R2-TEST", "rule_chronic_delay"].iloc[0] == 1
    # R3: Progress vs Spend Mismatch
    assert rdf.loc[rdf["work_id"] == "R3-TEST", "rule_progress_spend_mismatch"].iloc[0] == 1
    # R4: Cost Escalation
    assert rdf.loc[rdf["work_id"] == "R4-TEST", "rule_cost_escalation"].iloc[0] == 1
    # R5: Unusually Long Duration
    assert rdf.loc[rdf["work_id"] == "R5-TEST", "rule_unusually_long_duration"].iloc[0] == 1
    # R6: Premature Completion Claim
    assert rdf.loc[rdf["work_id"] == "R6-TEST", "rule_premature_completion"].iloc[0] == 1
    # R7: Suspicious Round Amount
    assert rdf.loc[rdf["work_id"] == "R7-TEST", "rule_suspicious_round_amount"].iloc[0] == 1
    # R8: Invalid Financial Data
    assert rdf.loc[rdf["work_id"] == "R8-TEST", "rule_invalid_financial_data"].iloc[0] == 1
    # Normal project should trigger zero rules
    assert rdf.loc[rdf["work_id"] == "NORMAL-TEST", "total_rules_triggered"].iloc[0] == 0

    assert summary["total_audited_records"] == 9
    assert len(summary["rule_counts"]) == len(RULE_METADATA)
