"""Unit Tests for Stage 2: Validation, Preprocessing, Quality Scoring, and Feature Engineering
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config import Config
from src.generate_data import generate_mplads_dataset
from src.validate import validate_dataset
from src.preprocess import preprocess_dataset
from src.quality import compute_quality_score
from src.features import engineer_features, check_round_number, ML_NUMERICAL_FEATURES

@pytest.fixture
def sample_dataset():
    """Generates a small test dataset."""
    return generate_mplads_dataset(num_records=200, anomaly_ratio=0.15)

def test_validation_engine(sample_dataset):
    """Verify validation engine catches negative values, date inversions, and counts records."""
    report, val_df = validate_dataset(sample_dataset)
    
    assert report["total_records"] == 200
    assert report["valid_records"] > 0
    assert report["invalid_records"] >= 0
    assert "field_summary" in report
    assert "is_record_valid" in val_df.columns
    
    # Inject specific invalid rows and check detection
    corrupt_df = pd.DataFrame({
        "work_id": ["TEST-NEG", "TEST-DATE", "TEST-PROG"],
        "state": ["Telangana", "Kerala", "Gujarat"],
        "district": ["Hyderabad", "Kochi", "Surat"],
        "sanctioned_amount": [-10.0, 20.0, 30.0],
        "estimated_cost": [10.0, 20.0, 30.0],
        "authorized_amount": [10.0, 20.0, 30.0],
        "expenditure": [10.0, 20.0, 30.0],
        "sanction_date": ["2023-01-01", "2023-06-01", "2023-01-01"],
        "expected_completion_date": ["2023-12-31", "2022-01-01", "2023-12-31"], # inverted date
        "status": ["Ongoing", "Ongoing", "Completed"],
        "progress_percent": [50.0, 50.0, 150.0] # > 100%
    })
    
    c_report, c_val_df = validate_dataset(corrupt_df)
    assert c_report["invalid_records"] == 3
    assert not c_val_df["is_record_valid"].any()

def test_preprocessing_and_quarantine(sample_dataset):
    """Verify preprocessing imputes missing values, isolates quarantine, and generates logs."""
    clean_df, quarantine_df, logs = preprocess_dataset(sample_dataset)
    
    assert len(clean_df) + len(quarantine_df) <= len(sample_dataset)
    assert len(logs) >= 5
    
    # Check log contents
    steps = [l["step"] for l in logs]
    assert "Ingestion" in steps
    assert "Data Quarantine" in steps
    assert "Text Normalization" in steps
    
    # Clean df must not have negative expenditure
    assert (clean_df["expenditure"] >= 0).all()
    # Clean df must not have progress > 100
    assert (clean_df["progress_percent"] <= 100.0).all()

def test_quality_score(sample_dataset):
    """Verify data quality scoring bounds and sub-score metrics."""
    quality = compute_quality_score(sample_dataset)
    
    assert 0.0 <= quality["overall_score"] <= 100.0
    for sub in ["completeness", "consistency", "validity", "uniqueness"]:
        assert 0.0 <= quality["sub_scores"][sub] <= 100.0
    assert quality["grade"] in ["A", "B", "C", "D", "F"]
    assert isinstance(quality["recommendations"], list)

def test_feature_engineering_formulas():
    """Verify engineered financial, temporal, and gap formulas."""
    df_test = pd.DataFrame({
        "work_id": ["W1", "W2"],
        "state": ["Telangana", "Maharashtra"],
        "district": ["Hyderabad", "Pune"],
        "implementing_agency": ["PWD", "PWD"],
        "work_category": ["Roads", "Education"],
        "work_description": ["Road work", "Classroom"],
        "sanctioned_amount": [50.0, 20.0],
        "estimated_cost": [40.0, 25.0],
        "authorized_amount": [45.0, 20.0],
        "expenditure": [40.0, 30.0],
        "sanction_date": ["2023-01-01", "2023-01-01"],
        "expected_completion_date": ["2023-07-01", "2023-04-01"],
        "actual_completion_date": ["2023-06-15", ""],
        "status": ["Completed", "Ongoing"],
        "progress_percent": [100.0, 50.0]
    })
    
    feat_df, X_scaled, scaler = engineer_features(df_test, fit_scaler=True)
    
    # Sanction utilization: 40/50 = 0.8, 30/20 = 1.5
    assert feat_df["sanction_utilization"].iloc[0] == 0.8
    assert feat_df["sanction_utilization"].iloc[1] == 1.5
    
    # Cost overrun pct: (40-40)/40 = 0.0, (30-25)/25 = 0.2
    assert feat_df["cost_overrun_pct"].iloc[0] == 0.0
    assert feat_df["cost_overrun_pct"].iloc[1] == 0.2
    
    # Progress vs spend gap:
    # W1: 0.8 - (100/100) = -0.2
    # W2: 1.5 - (50/100) = 1.0
    assert feat_df["progress_vs_spend_gap"].iloc[0] == -0.2
    assert feat_df["progress_vs_spend_gap"].iloc[1] == 1.0
    
    # Scaled matrix check
    assert X_scaled.shape == (2, len(ML_NUMERICAL_FEATURES))
    assert not np.isnan(X_scaled).any()

def test_round_number_detector():
    """Verify detection of round disbursals and threshold numbers."""
    assert check_round_number(5.0) == 1
    assert check_round_number(10.0) == 1
    assert check_round_number(25.0) == 1
    assert check_round_number(49.95) == 1
    assert check_round_number(4.99) == 1
    assert check_round_number(13.47) == 0
