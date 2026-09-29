"""Unit Tests for Stage 3: ML Models & Evaluation Metrics
"""

import numpy as np
import pytest
from src.generate_data import generate_mplads_dataset
from src.preprocess import preprocess_dataset
from src.features import engineer_features
from src.models import train_and_tune_models, load_saved_models, compute_model_scores
from src.evaluate import evaluate_models

@pytest.fixture(scope="module")
def trained_suite():
    """Generates synthetic dataset and trains the 3 authorized models."""
    df = generate_mplads_dataset(num_records=400, anomaly_ratio=0.15)
    clean_df, _, _ = preprocess_dataset(df)
    feat_df, X_scaled, scaler = engineer_features(clean_df, fit_scaler=True)
    y_labels = feat_df["is_anomaly"].fillna(0).astype(int).values
    
    train_results = train_and_tune_models(X_scaled, y_labels)
    eval_results = evaluate_models(
        train_results["models"],
        train_results["split_data"]["X_test"],
        train_results["split_data"]["y_test"]
    )
    return {
        "X_scaled": X_scaled,
        "y_labels": y_labels,
        "train_results": train_results,
        "eval_results": eval_results
    }

def test_model_training_and_serialization(trained_suite):
    """Verify all 3 authorized models are trained, tuned, and serialized to disk."""
    models = trained_suite["train_results"]["models"]
    assert "isolation_forest" in models
    assert "one_class_svm" in models
    assert "random_forest" in models
    
    # Check feature importances exist
    importances = trained_suite["train_results"]["feature_importances"]
    assert len(importances) > 0
    assert sum(importances.values()) > 0.90 # Sum of importances ~ 1.0
    
    # Check saved models can be loaded
    loaded = load_saved_models()
    assert loaded is not None
    assert "isolation_forest" in loaded
    assert "one_class_svm" in loaded
    assert "random_forest" in loaded

def test_model_evaluation_metrics(trained_suite):
    """Verify precision, recall, F1, and confusion matrix are generated for all 3 models."""
    eval_res = trained_suite["eval_results"]
    metrics = eval_res["metrics"]
    
    for model_key in ["isolation_forest", "one_class_svm", "random_forest"]:
        assert model_key in metrics
        m = metrics[model_key]
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1_score"] <= 1.0
        assert 0.0 <= m["accuracy"] <= 1.0
        assert len(m["confusion_matrix"]) == 2
        
    # Random Forest supervised model should have strong F1 on synthetic data
    assert metrics["random_forest"]["f1_score"] >= 0.70
    assert len(eval_res["comparison_table"]) == 3

def test_compute_model_scores_bounds(trained_suite):
    """Verify normalized model scores are bounded within [0, 100]."""
    scores = compute_model_scores(trained_suite["X_scaled"], trained_suite["train_results"]["models"])
    
    for key in ["if_score", "svm_score", "rf_prob"]:
        arr = scores[key]
        assert len(arr) == len(trained_suite["X_scaled"])
        assert (arr >= 0.0).all(), f"{key} has values < 0"
        assert (arr <= 100.0).all(), f"{key} has values > 100"
