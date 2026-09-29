"""Unit Tests for Stage 3: Risk Scoring, Explainability, Efficiency, and Power BI Exports
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config import Config
from src.generate_data import generate_mplads_dataset
from src.preprocess import preprocess_dataset
from src.features import engineer_features
from src.rules import apply_rule_engine
from src.duplicates import find_duplicate_works
from src.risk_score import compute_project_risk_scores, get_risk_level
from src.efficiency import analyze_efficiency
from src.export import export_powerbi_datasets

@pytest.fixture
def scored_dataset():
    """Generates and scores a sample dataset."""
    df = generate_mplads_dataset(num_records=200, anomaly_ratio=0.15)
    clean_df, _, _ = preprocess_dataset(df)
    feat_df, X_scaled, _ = engineer_features(clean_df, fit_scaler=True)
    ruled_df, _ = apply_rule_engine(feat_df)
    dup_df, _ = find_duplicate_works(ruled_df, threshold=0.80)
    
    # Mock model scores for deterministic evaluation
    n = len(dup_df)
    mock_scores = {
        "if_score": np.random.uniform(10, 90, n).round(2),
        "svm_score": np.random.uniform(10, 85, n).round(2),
        "rf_prob": np.random.uniform(5, 95, n).round(2)
    }
    scored = compute_project_risk_scores(dup_df, mock_scores)
    return scored

def test_risk_score_bounds_and_tiers(scored_dataset):
    """Verify composite risk score is strictly in [0, 100] and maps to valid tiers."""
    scores = scored_dataset["risk_score"]
    assert (scores >= 0.0).all()
    assert (scores <= 100.0).all()
    
    valid_levels = {"Low", "Medium", "High", "Critical"}
    assert set(scored_dataset["risk_level"].unique()).issubset(valid_levels)
    
    # Test specific tier mapping helper
    assert get_risk_level(25) == "Low"
    assert get_risk_level(45) == "Medium"
    assert get_risk_level(65) == "High"
    assert get_risk_level(85) == "Critical"

def test_sub_dimension_risk_breakdowns(scored_dataset):
    """Verify all 6 risk dimensions are bounded [0, 100]."""
    dimensions = ["financial_risk", "delay_risk", "anomaly_risk", "duplicate_risk", "progress_risk", "data_quality_risk"]
    for dim in dimensions:
        assert dim in scored_dataset.columns
        vals = scored_dataset[dim]
        assert (vals >= 0.0).all()
        assert (vals <= 100.0).all()

def test_explainable_reasons_compliance(scored_dataset):
    """Verify plain-language reasons strictly comply with policy wording requirements."""
    for reasons in scored_dataset["risk_reasons"]:
        assert isinstance(reasons, list)
        assert len(reasons) > 0
        for reason in reasons:
            assert isinstance(reason, str)
            # Mandatory compliance: Never say "fraud confirmed"
            assert "fraud confirmed" not in reason.lower()

def test_efficiency_metrics(scored_dataset):
    """Verify efficiency metrics calculations."""
    eff = analyze_efficiency(scored_dataset)
    assert "status_distribution" in eff
    assert "delay_metrics" in eff
    assert "state_efficiency" in eff
    assert "yearly_trend" in eff
    assert "gap_distribution" in eff
    assert eff["status_distribution"]["completed"] >= 0

def test_export_powerbi_datasets(scored_dataset, tmp_path):
    """Verify Power BI export CSV files are generated with valid headers."""
    exports = export_powerbi_datasets(scored_dataset)
    
    for name, filepath_str in exports.items():
        p = Path(filepath_str)
        assert p.exists(), f"Missing export file: {filepath_str}"
        df_exp = pd.read_csv(p)
        assert len(df_exp) > 0 or name == "alerts"
        # Check standard columns
        if name in ["projects_scored", "anomalies"]:
            assert "work_id" in df_exp.columns
            assert "risk_score" in df_exp.columns
            assert "risk_level" in df_exp.columns
        elif name == "district_summary":
            assert "district" in df_exp.columns
            assert "avg_risk_score" in df_exp.columns
        elif name == "alerts":
            assert "alert_id" in df_exp.columns
            assert "severity" in df_exp.columns
