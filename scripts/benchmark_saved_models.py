import sys
import time
import hashlib
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config
from src.preprocess import preprocess_dataset
from src.features import engineer_features
from src.models import load_saved_models

DATASET = ROOT / "data" / "synthetic" / "mplads_synthetic_5000.csv"
MODEL_DIR = Config.MODELS_STORE_DIR

FILES = [
    MODEL_DIR / "isolation_forest.joblib",
    MODEL_DIR / "one_class_svm.joblib",
    MODEL_DIR / "random_forest.joblib",
    MODEL_DIR / "feature_scaler.joblib",
]

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("\n=== SAVED MODEL INFERENCE BENCHMARK ===")

for path in [DATASET, *FILES]:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")

before = {p.name: file_hash(p) for p in FILES}

print("\n1. Loading dataset...")
raw_df = pd.read_csv(DATASET)
print("Raw records:", len(raw_df))

print("\n2. Preprocessing dataset...")
start = time.perf_counter()
clean_df, quarantine_df, cleaning_log = preprocess_dataset(raw_df)
preprocess_time = time.perf_counter() - start

print("Clean records:", len(clean_df))
print("Quarantined records:", len(quarantine_df))
print("Preprocessing time:", round(preprocess_time, 4), "seconds")

if clean_df.empty:
    raise ValueError("No valid records available.")

print("\n3. Loading saved scaler and models...")
start = time.perf_counter()

scaler = joblib.load(MODEL_DIR / "feature_scaler.joblib")
models = load_saved_models()

load_time = time.perf_counter() - start

if models is None:
    raise RuntimeError("Could not load all three saved models.")

print("Model loading time:", round(load_time, 4), "seconds")
print("Scaler features:", scaler.n_features_in_)

print("\n4. Engineering features without fitting scaler...")
start = time.perf_counter()

featured_df, X_scaled, _ = engineer_features(
    clean_df,
    scaler=scaler,
    fit_scaler=False,
    persist_scaler=False,
    reference_df=clean_df
)

feature_time = time.perf_counter() - start

if X_scaled.shape[1] != scaler.n_features_in_:
    raise ValueError("Feature count does not match saved scaler.")

print("Feature matrix:", X_scaled.shape)
print("Feature generation time:", round(feature_time, 4), "seconds")

print("\n5. Running saved-model inference...")

results = {}

for name in [
    "isolation_forest",
    "one_class_svm",
    "random_forest"
]:
    model = models[name]

    # Warm-up prediction
    model.predict(X_scaled[:min(10, len(X_scaled))])

    start = time.perf_counter()
    predictions = model.predict(X_scaled)
    elapsed = time.perf_counter() - start

    if name == "random_forest":
        anomalies = int(np.sum(predictions == 1))
    else:
        anomalies = int(np.sum(predictions == -1))

    results[name] = {
        "records": len(predictions),
        "anomalies": anomalies,
        "seconds": elapsed,
        "records_per_second": len(predictions) / elapsed
    }

    print("\nModel:", name)
    print("Records:", len(predictions))
    print("Potential anomaly predictions:", anomalies)
    print("Prediction time:", round(elapsed, 4), "seconds")
    print("Throughput:", round(len(predictions) / elapsed, 2),
          "records/second")

print("\n6. Checking saved model file integrity...")

after = {p.name: file_hash(p) for p in FILES}

for name in before:
    status = "UNCHANGED" if before[name] == after[name] else "MODIFIED"
    print(name, ":", status)

if before != after:
    raise RuntimeError("A saved model or scaler file changed!")

print("\n=== BENCHMARK SUMMARY ===")
print("Raw dataset records:", len(raw_df))
print("Processed records:", len(X_scaled))
print("Saved models tested:", len(results))
print("All saved model files unchanged: YES")
print("Retraining performed: NO")

print("\nInference benchmark completed successfully.")
