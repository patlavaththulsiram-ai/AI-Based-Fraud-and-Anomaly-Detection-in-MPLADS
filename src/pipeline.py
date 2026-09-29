"""End-to-End Orchestrator Pipeline
Executes data validation, preprocessing, quality audit, feature extraction,
rule evaluation, duplicate detection, ML model inference/training,
composite risk scoring, efficiency diagnostics, and Power BI dataset export.
Implements intelligent caching for sub-second UI response times.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import joblib

from config import Config
from src.validate import validate_dataset
from src.preprocess import preprocess_dataset
from src.quality import compute_quality_score
from src.features import engineer_features
from src.rules import apply_rule_engine
from src.duplicates import find_duplicate_works
from src.models import train_and_tune_models, load_saved_models, compute_model_scores
from src.evaluate import evaluate_models
from src.risk_score import compute_project_risk_scores
from src.efficiency import analyze_efficiency
from src.export import export_powerbi_datasets

CACHE_FILE = Config.CACHE_DIR / "pipeline_cache.joblib"

def run_pipeline(
    raw_df: pd.DataFrame,
    data_source: str = "Synthetic",
    force_retrain: bool = False
) -> Dict[str, Any]:
    """
    Executes the full audit pipeline on an input DataFrame and caches the structured results.
    """
    start_time = time.time()
    print(f"Starting MPLADS AI pipeline execution on {len(raw_df)} records (source: {data_source})...")

    # 1. Validation on raw input
    print("Step 1/10: Running validation audit...")
    val_report, val_df = validate_dataset(raw_df)

    # 2. Preprocessing & Quarantine
    print("Step 2/10: Preprocessing and segregating corrupt records...")
    clean_df, quarantine_df, cleaning_logs = preprocess_dataset(raw_df)

    # 3. Data Quality Evaluation
    print("Step 3/10: Computing 0-100 Data Quality Score...")
    quality_report = compute_quality_score(raw_df)

    # 4. Feature Engineering
    print("Step 4/10: Engineering financial, temporal, and spatial features...")
    feat_df, X_scaled, scaler = engineer_features(clean_df, fit_scaler=True)

    # 5. Deterministic Rules Engine
    print("Step 5/10: Applying policy and anomaly rules engine...")
    ruled_df, rule_summary = apply_rule_engine(feat_df)

    # 6. Duplicate Works Detection (TF-IDF Cosine Similarity)
    print("Step 6/10: Scanning for duplicate works (TF-IDF threshold >= 80%)...")
    dup_df, duplicate_pairs = find_duplicate_works(ruled_df, threshold=0.80)

    # 7. Model Training & Evaluation
    # Check if ground truth labels exist
    has_labels = ("is_anomaly" in dup_df.columns) and (dup_df["is_anomaly"].nunique() > 1)
    
    saved_models_dict = load_saved_models()
    
    if has_labels or force_retrain or saved_models_dict is None:
        print("Step 7/10: Training & tuning ML models (IsolationForest, OneClassSVM, RandomForest)...")
        if has_labels:
            y_labels = dup_df["is_anomaly"].fillna(0).astype(int).values
        else:
            # Fallback pseudo-labels from rules for initial unsupervised/semi-supervised calibration
            y_labels = (dup_df["total_rules_triggered"] >= 2).astype(int).values
            
        train_result = train_and_tune_models(X_scaled, y_labels)
        models = train_result["models"]
        feature_importances = train_result["feature_importances"]
        
        # Evaluate on test set
        eval_result = evaluate_models(
            models,
            train_result["split_data"]["X_test"],
            train_result["split_data"]["y_test"]
        )
    else:
        print("Step 7/10: Loading serialized ML models from models_store/...")
        models = saved_models_dict
        feature_importances = saved_models_dict.get("best_params", {}).get("feature_importances", {})
        eval_result = {
            "metrics": {},
            "comparison_table": [],
            "note": "Models evaluated on baseline holdout dataset."
        }

    # Compute model scores across all records
    print("Step 8/10: Computing normalized individual model risk vectors...")
    model_scores = compute_model_scores(X_scaled, models)

    # 8. Composite Risk Scoring & Explainability
    print("Step 9/10: Calculating composite 0-100 risk scores and explainable reasons...")
    scored_df = compute_project_risk_scores(dup_df, model_scores)

    # 9. Efficiency Diagnostics
    print("Step 10/10: Generating administrative efficiency analytics & Power BI exports...")
    efficiency_metrics = analyze_efficiency(scored_df)

    # 10. Power BI Dataset Exports
    exported_files = export_powerbi_datasets(scored_df)

    elapsed_time = round(time.time() - start_time, 2)
    print(f"Pipeline execution completed successfully in {elapsed_time} seconds.")

    # Package cache payload
    pipeline_cache = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed_time,
        "data_source": data_source,
        "total_projects": len(scored_df),
        "validation_report": val_report,
        "quarantine_count": len(quarantine_df),
        "cleaning_logs": cleaning_logs,
        "quality_report": quality_report,
        "rule_summary": rule_summary,
        "duplicate_pairs": duplicate_pairs,
        "evaluation_results": eval_result,
        "feature_importances": feature_importances,
        "efficiency_metrics": efficiency_metrics,
        "exported_files": exported_files,
        "scored_df": scored_df,
        "quarantine_df": quarantine_df
    }

    # Save to disk cache
    try:
        Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline_cache, CACHE_FILE)
    except Exception as e:
        print(f"Warning: could not write disk cache: {e}")

    return pipeline_cache

def get_cached_pipeline_results() -> Optional[Dict[str, Any]]:
    """Retrieves cached pipeline results from disk if available."""
    if CACHE_FILE.exists():
        try:
            return joblib.load(CACHE_FILE)
        except Exception:
            return None
    return None
