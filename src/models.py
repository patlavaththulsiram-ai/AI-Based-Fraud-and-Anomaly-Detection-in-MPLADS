
"""
Machine Learning Anomaly & Fraud Risk Detection Engine

Models:
1. IsolationForest - Unsupervised anomaly detection
2. OneClassSVM - Unsupervised anomaly detection
3. RandomForestClassifier - Supervised classification

Uses:
- Stratified 80/20 train-test split
- Resource-efficient fixed hyperparameters
- Single-worker model training
- Model serialization
- Normalized anomaly risk scores

Note:
Random Forest learns the supplied labels. If labels are
generated from business rules, its predictions represent
rule-derived classifications, not confirmed fraud.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import joblib

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

from config import Config
from src.features import ML_NUMERICAL_FEATURES


# =========================================================
# 1. TRAIN ALL THREE ML MODELS
# =========================================================

def train_and_tune_models(
    X_scaled: np.ndarray,
    y_labels: np.ndarray
) -> Dict[str, Any]:

    """
    Train three resource-efficient MPLADS ML models.

    Returns:
        Trained models, feature importances, best parameters,
        and stratified train-test split information.
    """

    Config.MODELS_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_scaled = np.asarray(
        X_scaled,
        dtype=np.float64
    )

    y_labels = np.asarray(
        y_labels,
        dtype=int
    )

    # Validate input
    if X_scaled.ndim != 2:
        raise ValueError(
            "Feature matrix must be two-dimensional."
        )

    if len(X_scaled) != len(y_labels):
        raise ValueError(
            "Feature and label counts do not match."
        )

    if len(X_scaled) < 10:
        raise ValueError(
            "At least 10 records are required."
        )

    if not np.isfinite(X_scaled).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    if not np.isin(y_labels, [0, 1]).all():
        raise ValueError(
            "Labels must contain only 0 and 1."
        )

    class_counts = np.bincount(
        y_labels,
        minlength=2
    )

    if class_counts.min() < 2:
        raise ValueError(
            "At least two records per class are required."
        )

    # =====================================================
    # 2. STRATIFIED 80/20 TRAIN-TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test, idx_train, idx_test = (
        train_test_split(
            X_scaled,
            y_labels,
            np.arange(len(y_labels)),
            test_size=0.20,
            random_state=Config.RANDOM_STATE,
            stratify=y_labels
        )
    )

    print(
        f"Training set: {len(X_train)} samples "
        f"({y_train.sum()} anomalies)"
    )

    print(
        f"Test set: {len(X_test)} samples "
        f"({y_test.sum()} anomalies)"
    )

    # =====================================================
    # 3. ISOLATION FOREST
    # =====================================================

    print(
        "Training lightweight Isolation Forest..."
    )

    contamination_rate = max(
        0.05,
        min(0.20, float(y_train.mean()))
    )

    best_if = IsolationForest(
        n_estimators=100,
        contamination=contamination_rate,
        max_samples="auto",
        random_state=Config.RANDOM_STATE,
        n_jobs=1
    )

    best_if.fit(X_train)

    print(
        "Isolation Forest training completed."
    )

    # =====================================================
    # 4. ONE-CLASS SVM
    # =====================================================

    print(
        "Training lightweight One-Class SVM..."
    )

    best_ocsvm = OneClassSVM(
        kernel="rbf",
        nu=0.05,
        gamma="scale",
        cache_size=100
    )

    best_ocsvm.fit(X_train)

    print(
        "One-Class SVM training completed."
    )

    # =====================================================
    # 5. RANDOM FOREST CLASSIFIER
    # =====================================================

    print(
        "Training lightweight Random Forest Classifier..."
    )

    best_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=Config.RANDOM_STATE,
        n_jobs=1
    )

    best_rf.fit(
        X_train,
        y_train
    )

    print(
        "Random Forest training completed."
    )

    # =====================================================
    # 6. FEATURE IMPORTANCE
    # =====================================================

    if len(best_rf.feature_importances_) != len(
        ML_NUMERICAL_FEATURES
    ):
        raise ValueError(
            "Feature importance count does not match "
            "ML_NUMERICAL_FEATURES."
        )

    feature_importances = dict(
        zip(
            ML_NUMERICAL_FEATURES,
            [
                round(float(value), 4)
                for value in best_rf.feature_importances_
            ]
        )
    )

    sorted_importances = dict(
        sorted(
            feature_importances.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    # =====================================================
    # 7. SAVE MODEL PARAMETERS
    # =====================================================

    best_params = {

        "isolation_forest": {
            "n_estimators": 100,
            "contamination": contamination_rate,
            "max_samples": "auto",
            "n_jobs": 1
        },

        "one_class_svm": {
            "kernel": "rbf",
            "nu": 0.05,
            "gamma": "scale",
            "cache_size": 100
        },

        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "n_jobs": 1
        },

        "feature_importances": sorted_importances
    }

    # =====================================================
    # 8. SERIALIZE MODELS
    # =====================================================

    joblib.dump(
        best_if,
        Config.MODELS_STORE_DIR / "isolation_forest.joblib"
    )

    joblib.dump(
        best_ocsvm,
        Config.MODELS_STORE_DIR / "one_class_svm.joblib"
    )

    joblib.dump(
        best_rf,
        Config.MODELS_STORE_DIR / "random_forest.joblib"
    )

    with open(
        Config.MODELS_STORE_DIR / "best_params.json",
        "w"
    ) as file:

        json.dump(
            best_params,
            file,
            indent=2
        )

    print(
        "All three ML models successfully trained and saved."
    )

    # =====================================================
    # 9. RETURN PIPELINE RESULTS
    # =====================================================

    return {

        "models": {

            "isolation_forest": best_if,

            "one_class_svm": best_ocsvm,

            "random_forest": best_rf

        },

        "best_params": best_params,

        "feature_importances": sorted_importances,

        "split_data": {

            "X_train": X_train,

            "X_test": X_test,

            "y_train": y_train,

            "y_test": y_test,

            "idx_train": idx_train,

            "idx_test": idx_test

        }

    }


# =========================================================
# 10. LOAD SAVED MODELS
# =========================================================

def load_saved_models() -> Optional[Dict[str, Any]]:

    """
    Load previously trained models from models_store.
    Returns None if the required model files are missing.
    """

    if_path = (
        Config.MODELS_STORE_DIR /
        "isolation_forest.joblib"
    )

    oc_path = (
        Config.MODELS_STORE_DIR /
        "one_class_svm.joblib"
    )

    rf_path = (
        Config.MODELS_STORE_DIR /
        "random_forest.joblib"
    )

    params_path = (
        Config.MODELS_STORE_DIR /
        "best_params.json"
    )

    model_paths = [
        if_path,
        oc_path,
        rf_path
    ]

    if not all(path.exists() for path in model_paths):

        return None

    try:

        best_params = {}

        if params_path.exists():

            with open(
                params_path,
                "r"
            ) as file:

                best_params = json.load(file)

        return {

            "isolation_forest": joblib.load(if_path),

            "one_class_svm": joblib.load(oc_path),

            "random_forest": joblib.load(rf_path),

            "best_params": best_params

        }

    except Exception as exc:

        print(
            f"Error loading saved ML models: {exc}"
        )

        return None


# =========================================================
# 11. COMPUTE MODEL RISK SCORES
# =========================================================

def compute_model_scores(
    X_scaled: np.ndarray,
    models: Dict[str, Any]
) -> Dict[str, np.ndarray]:

    """
    Calculate normalized 0-100 anomaly scores.

    Lower decision function values indicate more anomalous
    records for Isolation Forest and One-Class SVM.

    Random Forest returns class-1 probability.
    """

    X_scaled = np.asarray(
        X_scaled,
        dtype=np.float64
    )

    if X_scaled.ndim != 2 or len(X_scaled) == 0:

        raise ValueError(
            "Cannot calculate model scores for an empty dataset."
        )

    required_models = [
        "isolation_forest",
        "one_class_svm",
        "random_forest"
    ]

    for model_name in required_models:

        if model_name not in models:

            raise ValueError(
                f"Missing required model: {model_name}"
            )

    # =====================================================
    # ISOLATION FOREST SCORE
    # =====================================================

    if_model = models["isolation_forest"]

    if_raw = np.asarray(
        if_model.decision_function(X_scaled)
    ).reshape(-1)

    if_min = if_raw.min()
    if_max = if_raw.max()

    if if_max > if_min:

        if_score = (
            (if_max - if_raw) /
            (if_max - if_min)
        ) * 100.0

    else:

        if_score = np.zeros(len(X_scaled))

    # =====================================================
    # ONE-CLASS SVM SCORE
    # =====================================================

    svm_model = models["one_class_svm"]

    svm_raw = np.asarray(
        svm_model.decision_function(X_scaled)
    ).reshape(-1)

    svm_min = svm_raw.min()
    svm_max = svm_raw.max()

    if svm_max > svm_min:

        svm_score = (
            (svm_max - svm_raw) /
            (svm_max - svm_min)
        ) * 100.0

    else:

        svm_score = np.zeros(len(X_scaled))

    # =====================================================
    # RANDOM FOREST PROBABILITY
    # =====================================================

    rf_model = models["random_forest"]

    probabilities = rf_model.predict_proba(
        X_scaled
    )

    classes = list(rf_model.classes_)

    if 1 not in classes:

        rf_prob = np.zeros(len(X_scaled))

    else:

        anomaly_index = classes.index(1)

        rf_prob = (
            probabilities[:, anomaly_index] * 100.0
        )

    # =====================================================
    # RETURN SCORES
    # =====================================================

    return {

        "if_score": np.round(
            if_score,
            2
        ),

        "svm_score": np.round(
            svm_score,
            2
        ),

        "rf_prob": np.round(
            rf_prob,
            2
        )

    }
