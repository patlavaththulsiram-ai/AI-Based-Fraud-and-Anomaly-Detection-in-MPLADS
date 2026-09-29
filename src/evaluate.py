"""Model Evaluation & Benchmark Comparison Engine
Computes Precision, Recall, F1-Score, Accuracy, and Confusion Matrices
for IsolationForest, OneClassSVM, and RandomForest on the 20% test set.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score
)

def evaluate_models(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates IsolationForest, OneClassSVM, and RandomForest against ground truth test labels.
    
    Returns:
        comparison_dict with metrics, confusion matrices, and comparison table.
    """
    results = {}
    
    # Ground truth: 1 = anomaly, 0 = normal
    y_true = np.array(y_test, dtype=int)

    # 1. Isolation Forest
    # Predict returns -1 (outlier) or 1 (inlier). Convert to 1 (anomaly) and 0 (normal)
    if_model = models["isolation_forest"]
    if_raw_pred = if_model.predict(X_test)
    y_pred_if = np.where(if_raw_pred == -1, 1, 0)
    
    cm_if = confusion_matrix(y_true, y_pred_if).tolist()
    results["isolation_forest"] = {
        "model_name": "Isolation Forest",
        "type": "Unsupervised Ensemble",
        "precision": round(float(precision_score(y_true, y_pred_if, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred_if, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred_if, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred_if)), 4),
        "confusion_matrix": cm_if,
        "true_positives": int(cm_if[1][1]) if len(cm_if) > 1 else 0,
        "false_positives": int(cm_if[0][1]) if len(cm_if) > 1 else 0,
        "true_negatives": int(cm_if[0][0]) if len(cm_if) > 1 else 0,
        "false_negatives": int(cm_if[1][0]) if len(cm_if) > 1 else 0
    }

    # 2. One-Class SVM
    # Predict returns -1 (outlier) or 1 (inlier)
    svm_model = models["one_class_svm"]
    svm_raw_pred = svm_model.predict(X_test)
    y_pred_svm = np.where(svm_raw_pred == -1, 1, 0)
    
    cm_svm = confusion_matrix(y_true, y_pred_svm).tolist()
    results["one_class_svm"] = {
        "model_name": "One-Class SVM",
        "type": "Unsupervised Kernel Boundary",
        "precision": round(float(precision_score(y_true, y_pred_svm, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred_svm, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred_svm, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred_svm)), 4),
        "confusion_matrix": cm_svm,
        "true_positives": int(cm_svm[1][1]) if len(cm_svm) > 1 else 0,
        "false_positives": int(cm_svm[0][1]) if len(cm_svm) > 1 else 0,
        "true_negatives": int(cm_svm[0][0]) if len(cm_svm) > 1 else 0,
        "false_negatives": int(cm_svm[1][0]) if len(cm_svm) > 1 else 0
    }

    # 3. Random Forest Classifier
    rf_model = models["random_forest"]
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    cm_rf = confusion_matrix(y_true, y_pred_rf).tolist()
    roc_auc = round(float(roc_auc_score(y_true, y_proba_rf)), 4) if len(np.unique(y_true)) > 1 else 1.0
    
    results["random_forest"] = {
        "model_name": "Random Forest Classifier",
        "type": "Supervised Balanced Ensemble",
        "precision": round(float(precision_score(y_true, y_pred_rf, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred_rf, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred_rf, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred_rf)), 4),
        "roc_auc": roc_auc,
        "confusion_matrix": cm_rf,
        "true_positives": int(cm_rf[1][1]) if len(cm_rf) > 1 else 0,
        "false_positives": int(cm_rf[0][1]) if len(cm_rf) > 1 else 0,
        "true_negatives": int(cm_rf[0][0]) if len(cm_rf) > 1 else 0,
        "false_negatives": int(cm_rf[1][0]) if len(cm_rf) > 1 else 0
    }

    # Build Comparison Summary Table
    comparison_table = [
        {
            "model": "Isolation Forest",
            "category": "Unsupervised",
            "precision": f"{results['isolation_forest']['precision']*100:.1f}%",
            "recall": f"{results['isolation_forest']['recall']*100:.1f}%",
            "f1_score": f"{results['isolation_forest']['f1_score']*100:.1f}%",
            "accuracy": f"{results['isolation_forest']['accuracy']*100:.1f}%",
            "f1_raw": results["isolation_forest"]["f1_score"]
        },
        {
            "model": "One-Class SVM",
            "category": "Unsupervised",
            "precision": f"{results['one_class_svm']['precision']*100:.1f}%",
            "recall": f"{results['one_class_svm']['recall']*100:.1f}%",
            "f1_score": f"{results['one_class_svm']['f1_score']*100:.1f}%",
            "accuracy": f"{results['one_class_svm']['accuracy']*100:.1f}%",
            "f1_raw": results["one_class_svm"]["f1_score"]
        },
        {
            "model": "Random Forest Classifier",
            "category": "Supervised",
            "precision": f"{results['random_forest']['precision']*100:.1f}%",
            "recall": f"{results['random_forest']['recall']*100:.1f}%",
            "f1_score": f"{results['random_forest']['f1_score']*100:.1f}%",
            "accuracy": f"{results['random_forest']['accuracy']*100:.1f}%",
            "f1_raw": results["random_forest"]["f1_score"]
        }
    ]

    return {
        "metrics": results,
        "comparison_table": comparison_table,
        "note": "Evaluation metrics are measured against ground-truth labels on the 20% holdout test partition."
    }
