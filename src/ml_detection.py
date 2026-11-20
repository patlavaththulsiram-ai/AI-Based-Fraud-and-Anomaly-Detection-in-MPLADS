"""Dataset-scoped anomaly detection and supervised model evaluation."""

import logging
import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from config import Config
from src.features import ML_NUMERICAL_FEATURES, engineer_features

logger = logging.getLogger(__name__)

_REQUIRED_SOURCE_COLUMNS = {
    "sanctioned_amount",
    "estimated_cost",
    "expenditure",
    "sanction_date",
    "expected_completion_date",
    "actual_completion_date",
    "progress_percent",
    "implementing_agency",
    "status",
    "work_category"
}

_MODEL_ADVANTAGES = {
    "isolation_forest": "Tree-based isolation identifies observations that are easy to separate.",
    "one_class_svm": "RBF boundary detection can identify non-linear deviations from the learned cohort.",
    "random_forest": "Balanced supervised ensemble learns patterns from verified labelled examples."
}


def _empty_result(note: str) -> Dict[str, Any]:
    rows = [
        {
            "model": "Isolation Forest",
            "model_key": "isolation_forest",
            "category": "Unsupervised",
            "status": note,
            "precision": "—",
            "recall": "—",
            "f1_score": "—",
            "accuracy": "—",
            "records_evaluated": "—",
            "anomalies_detected": "—",
            "anomaly_pct": "—",
            "key_advantage": _MODEL_ADVANTAGES["isolation_forest"]
        },
        {
            "model": "One-Class SVM",
            "model_key": "one_class_svm",
            "category": "Unsupervised",
            "status": note,
            "precision": "—",
            "recall": "—",
            "f1_score": "—",
            "accuracy": "—",
            "records_evaluated": "—",
            "anomalies_detected": "—",
            "anomaly_pct": "—",
            "key_advantage": _MODEL_ADVANTAGES["one_class_svm"]
        },
        {
            "model": "Random Forest Classifier",
            "model_key": "random_forest",
            "category": "Supervised",
            "status": note,
            "precision": "—",
            "recall": "—",
            "f1_score": "—",
            "accuracy": "—",
            "records_evaluated": "—",
            "anomalies_detected": "—",
            "anomaly_pct": "—",
            "key_advantage": _MODEL_ADVANTAGES["random_forest"]
        }
    ]
    return {
        "comparison_table": rows,
        "metrics": {"random_forest": None},
        "feature_importances": {},
        "eval_note": note,
        "random_forest_status": note,
        "top_anomalies": [],
        "flagged_records": [],
        "chart_data": [],
        "models": {}
    }


def _format_metric(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1])
    }


def _valid_binary_label(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        normalized = str(value).strip().lower()
        if normalized in {"true", "yes", "1"}:
            return 1
        if normalized in {"false", "no", "0"}:
            return 0
        return None
    return int(numeric) if numeric in (0, 1) else None


def _comparison_row(
    name: str,
    key: str,
    category: str,
    advantage: str,
    status: str,
    evaluated: Optional[int] = None,
    anomalies: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    metrics = metrics or {}
    anomaly_pct = (
        anomalies / evaluated * 100
        if evaluated and anomalies is not None
        else None
    )
    return {
        "model": name,
        "model_key": key,
        "category": category,
        "status": status,
        "precision": _format_metric(metrics.get("precision")),
        "recall": _format_metric(metrics.get("recall")),
        "f1_score": _format_metric(metrics.get("f1_score")),
        "accuracy": _format_metric(metrics.get("accuracy")),
        "records_evaluated": evaluated if evaluated is not None else "—",
        "anomalies_detected": anomalies if anomalies is not None else "—",
        "anomaly_pct": f"{anomaly_pct:.1f}%" if anomaly_pct is not None else "—",
        "key_advantage": advantage
    }


def evaluate_dataset(
    dataframe: pd.DataFrame,
    ground_truth: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Fit and evaluate models using only the caller's selected-dataset records."""
    if dataframe is None or dataframe.empty:
        return _empty_result("No project records are available for model evaluation.")

    missing_columns = sorted(_REQUIRED_SOURCE_COLUMNS.difference(dataframe.columns))
    if missing_columns:
        return _empty_result(
            "Model evaluation requires these project fields: " + ", ".join(missing_columns) + "."
        )

    if "work_id" not in dataframe.columns:
        return _empty_result("The selected dataset has no work_id field for record-level results.")

    feature_input = dataframe.reset_index(drop=True).copy()
    for column in (
        "sanctioned_amount",
        "estimated_cost",
        "expenditure",
        "progress_percent"
    ):
        feature_input[column] = pd.to_numeric(feature_input[column], errors="coerce")
    for column in ("status", "work_category", "implementing_agency"):
        feature_input[column] = feature_input[column].astype("string").fillna("")

    work_ids = [
        str(work_id) if pd.notna(work_id) else ""
        for work_id in feature_input["work_id"]
    ]
    labelled_positions = []
    labelled_targets = []
    for position, work_id in enumerate(work_ids):
        raw_label = ground_truth.get(work_id) if ground_truth is not None else None
        label = _valid_binary_label(raw_label)
        if label is not None:
            labelled_positions.append(position)
            labelled_targets.append(label)

    split = None
    split_note = "Ground-truth labels are unavailable; only unsupervised anomaly counts can be reported."
    if labelled_targets:
        y_labelled = np.asarray(labelled_targets, dtype=int)
        class_counts = np.bincount(y_labelled, minlength=2)
        if class_counts.min() >= 2:
            test_count = max(2, int(math.ceil(len(y_labelled) * 0.2)))
            if test_count < len(y_labelled) and len(y_labelled) - test_count >= 2:
                try:
                    train_positions, test_positions = train_test_split(
                        np.asarray(labelled_positions, dtype=int),
                        test_size=test_count,
                        random_state=Config.RANDOM_STATE,
                        stratify=y_labelled
                    )
                    split = (
                        train_positions,
                        test_positions,
                        np.asarray([_valid_binary_label(ground_truth[work_ids[i]]) for i in train_positions], dtype=int),
                        np.asarray([_valid_binary_label(ground_truth[work_ids[i]]) for i in test_positions], dtype=int)
                    )
                except ValueError as exc:
                    split_note = f"Stratified holdout split is not supported by these labels: {exc}"
            else:
                split_note = "Not enough labelled records for separate train and test partitions."
        else:
            split_note = (
                "Random Forest evaluation requires at least two labelled examples in each class; "
                f"found {int(class_counts[0])} normal and {int(class_counts[1])} anomalous."
            )

    if split is None:
        fit_positions = np.arange(len(feature_input), dtype=int)
        reference_df = None
    else:
        fit_positions = split[0]
        reference_df = feature_input.iloc[fit_positions]

    try:
        engineered, _, _ = engineer_features(
            feature_input,
            fit_scaler=True,
            persist_scaler=False,
            reference_df=reference_df
        )
    except Exception as exc:
        logger.exception("Feature engineering failed for selected dataset.")
        return _empty_result(f"Feature engineering failed: {exc}")

    feature_frame = engineered.reindex(columns=ML_NUMERICAL_FEATURES)
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    usable_features = feature_frame.iloc[fit_positions].columns[
        feature_frame.iloc[fit_positions].notna().any()
    ].tolist()
    if not usable_features:
        return _empty_result("No usable numeric engineered features are available.")

    feature_frame = feature_frame[usable_features]
    imputer = SimpleImputer(strategy="median")
    try:
        imputer.fit(feature_frame.iloc[fit_positions])
        imputed_features = imputer.transform(feature_frame)
        if imputed_features.shape[1] == 0 or not np.isfinite(imputed_features).all():
            return _empty_result("The selected dataset has no finite values for model features.")
    except Exception as exc:
        logger.exception("Feature imputation failed for selected dataset.")
        return _empty_result(f"Feature imputation failed: {exc}")

    if split is None:
        fit_features = imputed_features
        y_test = None
    else:
        train_positions, _, _, y_test = split
        fit_features = imputed_features[train_positions]

    if len(fit_features) < 2:
        return _empty_result("At least two project records are required for unsupervised models.")

    scaler = StandardScaler()
    try:
        fit_scaled = scaler.fit_transform(fit_features)
        all_scaled = scaler.transform(imputed_features)
    except Exception as exc:
        logger.exception("Feature scaling failed for selected dataset.")
        return _empty_result(f"Feature scaling failed: {exc}")

    results = {}
    anomaly_record_ids = set()
    flagged_records = []
    project_name_column = next(
        (
            column for column in ("project_name", "project_title", "work_name", "title")
            if column in feature_input.columns
        ),
        "work_description" if "work_description" in feature_input.columns else None
    )
    unsupervised_specs = (
        ("isolation_forest", "Isolation Forest", IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )),
        ("one_class_svm", "One-Class SVM", OneClassSVM(
            kernel="rbf",
            nu=0.05,
            gamma="scale"
        ))
    )
    for key, name, estimator in unsupervised_specs:
        try:
            estimator.fit(fit_scaled)
            predictions = (estimator.predict(all_scaled) == -1).astype(int)
            anomaly_scores = -np.asarray(estimator.decision_function(all_scaled)).reshape(-1)
            detected = int(predictions.sum())
            metrics = (
                _classification_metrics(y_test, predictions[split[1]])
                if y_test is not None
                else None
            )
            if metrics is not None and key == "isolation_forest":
                results["isolation_forest"] = metrics
            elif metrics is not None:
                results["one_class_svm"] = metrics

            for position, prediction in enumerate(predictions):
                if prediction:
                    anomaly_record_ids.add(work_ids[position])
                    raw_reasons = feature_input.iloc[position].get("top_risk_drivers")
                    if isinstance(raw_reasons, (list, tuple, np.ndarray)):
                        reason = "; ".join(
                            str(value) for value in raw_reasons
                            if value is not None and not pd.isna(value) and str(value).strip()
                        )
                    elif isinstance(raw_reasons, str):
                        reason = raw_reasons.strip()
                        if reason.casefold() == "nan":
                            reason = ""
                    else:
                        reason = ""
                    project_name = (
                        feature_input.iloc[position].get(project_name_column)
                        if project_name_column
                        else None
                    )
                    if project_name is not None and pd.isna(project_name):
                        project_name = None
                    risk_level = feature_input.iloc[position].get("risk_level")
                    if risk_level is not None and pd.isna(risk_level):
                        risk_level = None
                    flagged_records.append({
                        "project_id": work_ids[position],
                        "project_name": str(project_name).strip() if project_name is not None else "",
                        "anomaly_score": float(anomaly_scores[position]),
                        "model": name,
                        "risk_level": str(risk_level).strip() if risk_level is not None else "",
                        "reason": reason
                    })

            results[key] = {
                "status": "Evaluated",
                "records_evaluated": len(work_ids),
                "anomalies_detected": detected,
                "metrics": metrics,
                "predictions": predictions,
                "anomaly_scores": anomaly_scores
            }
        except Exception as exc:
            logger.exception("%s failed for selected dataset.", name)
            results[key] = {
                "status": f"Model failed: {exc}",
                "records_evaluated": None,
                "anomalies_detected": None,
                "metrics": None,
                "predictions": None,
                "anomaly_scores": None
            }

    random_forest_metrics = None
    feature_importances = {}
    if split is None:
        rf_status = split_note
    else:
        train_positions, test_positions, y_train, y_test = split
        train_counts = np.bincount(y_train, minlength=2)
        if train_counts.min() == 0:
            rf_status = "Both labelled classes must be present in the training partition."
        else:
            try:
                base_model = RandomForestClassifier(
                    random_state=Config.RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1
                )
                min_class_count = int(train_counts.min())
                if min_class_count >= 2:
                    cv_folds = min(3, min_class_count)
                    search = GridSearchCV(
                        base_model,
                        param_grid={
                            "n_estimators": [100, 200],
                            "max_depth": [None, 10],
                            "min_samples_split": [2, 5]
                        },
                        scoring="f1",
                        cv=StratifiedKFold(
                            n_splits=cv_folds,
                            shuffle=True,
                            random_state=Config.RANDOM_STATE
                        ),
                        n_jobs=-1
                    )
                    search.fit(imputed_features[train_positions], y_train)
                    fitted_model = search.best_estimator_
                    rf_status = f"Evaluated; GridSearchCV used {cv_folds}-fold stratified cross-validation."
                else:
                    fitted_model = base_model.fit(imputed_features[train_positions], y_train)
                    rf_status = "Evaluated without GridSearchCV; training class counts were too small for cross-validation."

                rf_predictions = fitted_model.predict(imputed_features[test_positions])
                random_forest_metrics = _classification_metrics(y_test, rf_predictions)
                if len(np.unique(y_test)) == 2:
                    probabilities = fitted_model.predict_proba(imputed_features[test_positions])[:, 1]
                    random_forest_metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
                else:
                    random_forest_metrics["roc_auc"] = None
                feature_importances = {
                    feature: float(importance)
                    for feature, importance in sorted(
                        zip(usable_features, fitted_model.feature_importances_),
                        key=lambda item: item[1],
                        reverse=True
                    )
                }
                results["random_forest"] = {
                    "status": rf_status,
                    **random_forest_metrics
                }
            except Exception as exc:
                logger.exception("Random Forest training/evaluation failed for selected dataset.")
                rf_status = f"Model failed: {exc}"
                results["random_forest"] = {"status": rf_status}

    comparison_table = []
    for key, name, category in (
        ("isolation_forest", "Isolation Forest", "Unsupervised"),
        ("one_class_svm", "One-Class SVM", "Unsupervised")
    ):
        model_result = results[key]
        comparison_table.append(_comparison_row(
            name=name,
            key=key,
            category=category,
            advantage=_MODEL_ADVANTAGES[key],
            status=model_result["status"],
            evaluated=model_result["records_evaluated"],
            anomalies=model_result["anomalies_detected"],
            metrics=model_result.get("metrics")
        ))
    comparison_table.append(_comparison_row(
        name="Random Forest Classifier",
        key="random_forest",
        category="Supervised",
        advantage=_MODEL_ADVANTAGES["random_forest"],
        status=rf_status,
        evaluated=len(y_test) if split is not None else None,
        metrics=random_forest_metrics
    ))
    chart_data = [
        {
            "model": row["model"],
            "anomaly_count": row["anomalies_detected"],
            "anomaly_percentage": float(row["anomalies_detected"]) / len(work_ids) * 100
        }
        for row in comparison_table[:2]
        if isinstance(row["anomalies_detected"], int) and work_ids
    ]

    if split is not None:
        eval_note = (
            f"Unsupervised metrics and Random Forest metrics use the selected dataset's "
            f"{len(y_test)}-record stratified holdout. GridSearchCV is used only when the training "
            "partition has enough examples in both classes."
        )
    else:
        eval_note = (
            f"{split_note} Isolation Forest and One-Class SVM were evaluated on the selected "
            f"dataset's {len(imputed_features)} records. Detected records are potential anomalies "
            "requiring verification, not confirmed fraud."
        )

    top_anomalies = (
        engineered[engineered["work_id"].astype(str).isin(anomaly_record_ids)]
        .sort_values("risk_score", ascending=False)
        .head(25)
        .to_dict(orient="records")
        if "risk_score" in engineered.columns
        else engineered[engineered["work_id"].astype(str).isin(anomaly_record_ids)]
        .head(25)
        .to_dict(orient="records")
    )
    return {
        "comparison_table": comparison_table,
        "metrics": {"random_forest": random_forest_metrics},
        "feature_importances": feature_importances,
        "eval_note": eval_note,
        "random_forest_status": rf_status,
        "top_anomalies": top_anomalies,
        "flagged_records": sorted(
            flagged_records,
            key=lambda record: record["anomaly_score"],
            reverse=True
        ),
        "chart_data": chart_data,
        "models": results
    }
