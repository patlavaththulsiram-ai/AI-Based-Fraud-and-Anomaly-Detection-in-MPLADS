"""Unit Tests for Stage 3: Duplicate Work Detection Engine
"""

import pandas as pd
import pytest
from src.duplicates import find_duplicate_works

def test_duplicate_work_detection():
    """Verify TF-IDF cosine similarity detects duplicate and near-duplicate works."""
    df_works = pd.DataFrame({
        "work_id": ["W-01", "W-02", "W-03", "W-04"],
        "district": ["Hyderabad", "Hyderabad", "Hyderabad", "Warangal"],
        "work_category": ["Education", "Education", "Roads", "Education"],
        "work_description": [
            "Construction of 2 additional smart classrooms at ZP High School, Kondapur",
            "Construction of 2 additional modern classrooms at ZP High School, Kondapur", # near duplicate of W-01
            "Laying of CC road pavement in Ward 12",
            "Construction of 2 additional smart classrooms at ZP High School, Kondapur" # identical text but in Warangal (different district)
        ],
        "sanctioned_amount": [25.0, 24.5, 30.0, 25.0],
        "expenditure": [24.0, 23.5, 29.0, 24.0]
    })

    dup_df, pairs = find_duplicate_works(df_works, threshold=0.80)

    # W-01 and W-02 must be flagged as duplicates
    assert len(pairs) >= 1
    top_pair = pairs[0]
    assert {top_pair["work_id_1"], top_pair["work_id_2"]} == {"W-01", "W-02"}
    assert top_pair["similarity_pct"] >= 80.0
    
    # Check flags in dataframe
    assert dup_df.loc[dup_df["work_id"] == "W-01", "duplicate_risk_flag"].iloc[0] == 1
    assert dup_df.loc[dup_df["work_id"] == "W-02", "duplicate_risk_flag"].iloc[0] == 1
    assert dup_df.loc[dup_df["work_id"] == "W-03", "duplicate_risk_flag"].iloc[0] == 0
