"""Machine Learning Anomaly & Fraud Risk Detection Engine
Trains and serializes the three authorized models:
1. IsolationForest (Unsupervised tree ensemble)
2. OneClassSVM (Unsupervised RBF kernel support vector machine)
3. RandomForestClassifier (Supervised balanced ensemble)
Performs Stratified 80/20 Train-Test split and GridSearchCV optimization.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import joblib

from config import Config
from src.features import ML_NUMERICAL_FEATURES

def _unsupervised_scorer(estimator, X, y=None):
    """Mean decision function scorer for unsupervised model tuning."""
    return float(np.mean(estimator.decision_function(X)))

def train_and_tune_models(X_scaled: np.ndarray, y_labels: np.ndarray) -> Dict[str, Any]:
    """
    Trains and tunes IsolationForest, OneClassSVM, and RandomForestClassifier.
    
    Returns:
        dictionary containing trained models, test split indices, predictions, and best parameters.
    """
    Config.MODELS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y_labels, np.arange(len(y_labels)),
        test_size=0.20,
        random_state=Config.RANDOM_STATE,
        stratify=y_labels
    )
    
    print(f"Training set: {len(X_train)} samples ({y_train.sum()} anomalies)")
    print(f"Test set: {len(X_test)} samples ({y_test.sum()} anomalies)")

    # 1. Isolation Forest (Unsupervised)
    print("Training Isolation Forest with hyperparameter tuning...")
    contamination_rate = max(0.05, min(0.20, float(y_train.mean())))
    if_grid = GridSearchCV(
        IsolationForest(random_state=Config.RANDOM_STATE),
        param_grid={
            "n_estimators": [100, 150],
            "contamination": [0.08, contamination_rate]
        },
        scoring=_unsupervised_scorer,
        cv=2,
        n_jobs=-1
    )
    if_grid.fit(X_train)
    best_if = if_grid.best_estimator_

    # 2. One-Class SVM (Unsupervised RBF Kernel)
    print("Training One-Class SVM with hyperparameter tuning...")
    oc_grid = GridSearchCV(
        OneClassSVM(kernel="rbf"),
        param_grid={
            "nu": [0.05, 0.10],
            "gamma": ["scale", "auto"]
        },
        scoring=_unsupervised_scorer,
        cv=2,
        n_jobs=-1
    )
    oc_grid.fit(X_train)
    best_ocsvm = oc_grid.best_estimator_

    # 3. Random Forest Classifier (Supervised with GridSearchCV)
    print("Training Random Forest Classifier with GridSearchCV...")
    rf_base = RandomForestClassifier(
        random_state=Config.RANDOM_STATE,
        class_weight="balanced"
    )
    rf_grid = GridSearchCV(
        rf_base,
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        },
        scoring="f1",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE),
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    # Extract Feature Importances
    feature_importances = dict(zip(ML_NUMERICAL_FEATURES, [round(float(v), 4) for v in best_rf.feature_importances_]))
    # Sort descending
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    best_params = {
        "isolation_forest": if_grid.best_params_,
        "one_class_svm": oc_grid.best_params_,
        "random_forest": rf_grid.best_params_,
        "feature_importances": sorted_importances
    }

    # Save models to models_store/
    joblib.dump(best_if, Config.MODELS_STORE_DIR / "isolation_forest.joblib")
    joblib.dump(best_ocsvm, Config.MODELS_STORE_DIR / "one_class_svm.joblib")
    joblib.dump(best_rf, Config.MODELS_STORE_DIR / "random_forest.joblib")
    
    with open(Config.MODELS_STORE_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("All models successfully trained, tuned, and serialized.")

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

def load_saved_models() -> Optional[Dict[str, Any]]:
    """Loads serialized models from disk if present."""
    if_path = Config.MODELS_STORE_DIR / "isolation_forest.joblib"
    oc_path = Config.MODELS_STORE_DIR / "one_class_svm.joblib"
    rf_path = Config.MODELS_STORE_DIR / "random_forest.joblib"
    params_path = Config.MODELS_STORE_DIR / "best_params.json"
    
    if all(p.exists() for p in [if_path, oc_path, rf_path]):
        best_params = {}
        if params_path.exists():
            with open(params_path, "r") as f:
                best_params = json.load(f)
        return {
            "isolation_forest": joblib.load(if_path),
            "one_class_svm": joblib.load(oc_path),
            "random_forest": joblib.load(rf_path),
            "best_params": best_params
        }
    return None

def compute_model_scores(X_scaled: np.ndarray, models: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Computes normalized 0-100 anomaly & fraud risk scores for all three models across the full dataset.
    
    Returns:
        {
            "if_score": 0-100 array,
            "svm_score": 0-100 array,
            "rf_prob": 0-100 array
        }
    """
    # 1. Isolation Forest: decision_function (lower is more anomalous)
    if_model = models["isolation_forest"]
    if_raw = if_model.decision_function(X_scaled)
    # Invert and normalize to 0-100
    # if_raw is typically in range [-0.3, 0.3]
    if_min, if_max = if_raw.min(), if_raw.max()
    if if_max > if_min:
        if_score = ((if_max - if_raw) / (if_max - if_min)) * 100.0
    else:
        if_score = np.zeros(len(X_scaled))

    # 2. One-Class SVM: decision_function (negative is outlier, positive is inlier)
    svm_model = models["one_class_svm"]
    svm_raw = svm_model.decision_function(X_scaled)
    svm_min, svm_max = svm_raw.min(), svm_raw.max()
    if svm_max > svm_min:
        svm_score = ((svm_max - svm_raw) / (svm_max - svm_min)) * 100.0
    else:
        svm_score = np.zeros(len(X_scaled))

    # 3. Random Forest: predicted probability for class 1 (Fraud / Anomaly)
    rf_model = models["random_forest"]
    rf_proba = rf_model.predict_proba(X_scaled)[:, 1] * 100.0

    return {
        "if_score": np.round(if_score, 2),
        "svm_score": np.round(svm_score, 2),
        "rf_prob": np.round(rf_proba, 2)
    }
