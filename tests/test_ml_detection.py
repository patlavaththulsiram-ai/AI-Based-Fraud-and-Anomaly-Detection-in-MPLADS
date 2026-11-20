import pytest

from config import Config
from src.generate_data import generate_mplads_dataset
from src.features import ML_NUMERICAL_FEATURES
from src.ingest_service import _ground_truth_label
from src.ml_detection import evaluate_dataset
from src.preprocess import preprocess_dataset


@pytest.fixture(scope="module")
def labelled_frame(tmp_path_factory):
    original_synthetic_dir = Config.DATA_SYNTHETIC_DIR
    Config.DATA_SYNTHETIC_DIR = tmp_path_factory.mktemp("ml-detection-data")
    try:
        source = generate_mplads_dataset(num_records=100, anomaly_ratio=0.20)
        cleaned, _, _ = preprocess_dataset(source)
        labels = dict(zip(cleaned["work_id"].astype(str), cleaned["is_anomaly"]))
        return cleaned, labels
    finally:
        Config.DATA_SYNTHETIC_DIR = original_synthetic_dir


def test_unlabelled_dataset_reports_actual_unsupervised_counts(labelled_frame):
    dataframe, _ = labelled_frame
    result = evaluate_dataset(dataframe)

    assert len(result["comparison_table"]) == 3
    for row in result["comparison_table"][:2]:
        assert row["records_evaluated"] == len(dataframe)
        assert isinstance(row["anomalies_detected"], int)
        assert row["anomaly_pct"] == (
            f"{row['anomalies_detected'] / len(dataframe) * 100:.1f}%"
        )
        assert row["precision"] == "—"
        assert row["recall"] == "—"
        assert row["f1_score"] == "—"
        assert row["accuracy"] == "—"
    assert [row["anomaly_count"] for row in result["chart_data"]] == [
        row["anomalies_detected"] for row in result["comparison_table"][:2]
    ]
    assert len(result["flagged_records"]) == sum(
        row["anomalies_detected"] for row in result["comparison_table"][:2]
    )
    assert all(
        {"project_id", "project_name", "anomaly_score", "model", "risk_level", "reason"}
        <= set(record)
        for record in result["flagged_records"]
    )
    assert result["metrics"]["random_forest"] is None
    assert not result["feature_importances"]
    assert "Ground-truth labels are unavailable" in result["eval_note"]


def test_labelled_dataset_uses_real_holdout_evaluation_and_importance(labelled_frame):
    dataframe, labels = labelled_frame
    result = evaluate_dataset(dataframe, labels)

    assert len(result["comparison_table"]) == 3
    rf_row = result["comparison_table"][2]
    rf_metrics = result["metrics"]["random_forest"]
    assert rf_metrics is not None
    assert rf_row["records_evaluated"] > 0
    assert len(rf_metrics["confusion_matrix"]) == 2
    assert sum(map(sum, rf_metrics["confusion_matrix"])) == rf_row["records_evaluated"]
    assert 0 <= rf_metrics["precision"] <= 1
    assert 0 <= rf_metrics["recall"] <= 1
    assert 0 <= rf_metrics["f1_score"] <= 1
    assert 0 <= rf_metrics["accuracy"] <= 1
    assert result["feature_importances"]
    assert abs(sum(result["feature_importances"].values()) - 1.0) < 1e-6
    assert set(result["feature_importances"]).issubset(ML_NUMERICAL_FEATURES)


def test_small_or_one_class_labelled_data_skips_supervised_metrics(labelled_frame):
    dataframe, labels = labelled_frame
    normal_only_labels = {work_id: 0 for work_id in labels}
    result = evaluate_dataset(dataframe, normal_only_labels)

    assert result["metrics"]["random_forest"] is None
    assert "at least two labelled examples in each class" in result["comparison_table"][2]["status"]
    assert result["comparison_table"][0]["records_evaluated"] == len(dataframe)


def test_missing_required_feature_columns_return_clear_evaluation_note(labelled_frame):
    dataframe, _ = labelled_frame
    result = evaluate_dataset(dataframe.drop(columns=["expenditure"]))

    assert "expenditure" in result["eval_note"]
    assert all(row["records_evaluated"] == "—" for row in result["comparison_table"])


def test_empty_dataset_returns_unavailable_counts_without_failing(labelled_frame):
    dataframe, _ = labelled_frame
    result = evaluate_dataset(dataframe.iloc[0:0])

    assert "No project records" in result["eval_note"]
    assert result["chart_data"] == []
    assert result["flagged_records"] == []


def test_missing_feature_values_are_handled(labelled_frame):
    dataframe, _ = labelled_frame
    incomplete = dataframe.copy()
    for column in (
        "sanctioned_amount",
        "estimated_cost",
        "expenditure",
        "progress_percent",
        "sanction_date",
        "expected_completion_date",
        "actual_completion_date",
    ):
        incomplete.loc[incomplete.index[0], column] = None

    result = evaluate_dataset(incomplete)

    assert all(
        row["records_evaluated"] == len(incomplete)
        for row in result["comparison_table"][:2]
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), (1, 1), (False, 0), (True, 1), ("yes", 1), ("no", 0), (None, None), (2, None), ("unknown", None)]
)
def test_ingestion_only_persists_explicit_binary_ground_truth(value, expected):
    assert _ground_truth_label(value) == expected
