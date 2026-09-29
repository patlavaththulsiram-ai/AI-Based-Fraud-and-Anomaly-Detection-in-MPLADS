"""Unit Tests for Stage 1: Data Generation, Loading, and Schema Mapping
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from config import Config
from src.generate_data import generate_mplads_dataset, STATES_DATA, CATEGORIES
from src.load_data import load_dataset, map_columns_to_canonical, detect_data_source

def test_generate_data_structure():
    """Verify generated dataset contains canonical columns and correct anomaly distribution."""
    sample_size = 500
    df = generate_mplads_dataset(num_records=sample_size, anomaly_ratio=0.10)
    
    assert len(df) == sample_size
    assert "is_anomaly" in df.columns
    assert "anomaly_type" in df.columns
    
    # Check canonical columns exist
    for col in Config.CANONICAL_COLUMNS:
        assert col in df.columns, f"Missing canonical column: {col}"
        
    # Check states and categories coverage
    generated_states = set(df["state"].unique())
    assert len(generated_states) >= 10, f"Expected at least 10 states, got {len(generated_states)}"
    
    generated_categories = set(df["work_category"].unique())
    assert set(CATEGORIES).issubset(generated_categories)
    
    # Check anomaly distribution
    anomaly_count = df["is_anomaly"].sum()
    expected_anomalies = int(sample_size * 0.10)
    assert abs(anomaly_count - expected_anomalies) <= 5
    
    # Verify expected anomaly types exist
    types = set(df[df["is_anomaly"] == 1]["anomaly_type"].unique())
    assert "cost_overrun" in types
    assert "expenditure_exceeds_sanction" in types
    assert "premature_completion" in types
    assert "chronic_delay" in types

def test_map_columns_to_canonical():
    """Verify intelligent alias mapping for government data exports."""
    raw_data = {
        "workid": ["W1", "W2"],
        "honble_mp": ["MP A", "MP B"],
        "state_name": ["Telangana", "Karnataka"],
        "sanction_amt": [25.0, 50.0],
        "funds_utilized": [20.0, 45.0],
        "work_status": ["Completed", "Ongoing"],
        "physical_progress": [100.0, 75.0],
        "random_extra_col": ["XYZ", "ABC"]
    }
    raw_df = pd.DataFrame(raw_data)
    mapped_df, mapping, unmapped = map_columns_to_canonical(raw_df)
    
    assert mapped_df["work_id"].iloc[0] == "W1"
    assert mapped_df["mp_name"].iloc[0] == "MP A"
    assert mapped_df["state"].iloc[0] == "Telangana"
    assert mapped_df["sanctioned_amount"].iloc[0] == 25.0
    assert mapped_df["expenditure"].iloc[0] == 20.0
    assert mapped_df["status"].iloc[0] == "Completed"
    assert mapped_df["progress_percent"].iloc[0] == 100.0
    assert "random_extra_col" in unmapped

def test_load_dataset_csv_and_json(tmp_path):
    """Verify loading from CSV and JSON files, and correct data source detection."""
    # Test CSV
    csv_file = tmp_path / "test_mplads.csv"
    df_test = pd.DataFrame({
        "work_id": ["TEST-01"],
        "mp_name": ["Sri A"],
        "state": ["Telangana"],
        "sanctioned_amount": [10.0],
        "expenditure": [8.0],
        "status": ["Ongoing"]
    })
    df_test.to_csv(csv_file, index=False)
    
    loaded_df, meta = load_dataset(csv_file)
    assert meta["total_rows"] == 1
    assert meta["data_source"] == "Uploaded"
    assert "work_id" in loaded_df.columns
    
    # Test JSON
    json_file = tmp_path / "test_data.json"
    with open(json_file, "w") as f:
        json.dump([{"work_id": "J1", "mp_name": "MP J", "sanctioned_amount": 15.0}], f)
        
    loaded_json_df, json_meta = load_dataset(json_file)
    assert json_meta["total_rows"] == 1
    assert loaded_json_df["work_id"].iloc[0] == "J1"

def test_detect_data_source():
    """Verify synthetic vs uploaded badge detection."""
    df_syn = pd.DataFrame({"is_anomaly": [0, 1], "anomaly_type": ["None", "cost_overrun"]})
    assert detect_data_source("test_upload.csv", df_syn) == "Synthetic"
    
    df_up = pd.DataFrame({"work_id": [1, 2]})
    assert detect_data_source("test_upload.csv", df_up) == "Uploaded"
    assert detect_data_source("data/synthetic/data.csv", df_up) == "Synthetic"
