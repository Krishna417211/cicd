"""
Simple monitoring: verifies the production model loads and checks for
basic data drift between the training data and any new data.

This version is CI-friendly (no running servers required).
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

print("=" * 60)
print("Model Monitoring")
print("=" * 60)

# ----------------------------------------
# 1. Model health check
# ----------------------------------------
MODEL_PATH = "models/taxi_model.keras"

if not os.path.exists(MODEL_PATH):
    print("[DOWN] Model file not found")
    raise SystemExit(1)

model = tf.keras.models.load_model(MODEL_PATH)
print(f"[OK]   Model loads. Expects input shape {model.input_shape}")

# Quick prediction sanity check (4 features)
sample = np.array([[1, 2.5, 10.0, 2.0]], dtype=np.float32)
pred = model.predict(sample, verbose=0)
if np.isfinite(pred[0][0]):
    print(f"[OK]   Sample prediction works: {float(pred[0][0]):.2f}")
else:
    print("[WARNING] Prediction is not a finite number")

# ----------------------------------------
# 2. Basic data drift check
#    Compare mean of each feature vs the cleaned training data.
# ----------------------------------------
DATA_PATH = "data/taxi_clean.csv"
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    means = df.drop(columns=["total_amount"]).mean().to_dict()
    print("\nFeature averages (baseline):")
    for k, v in means.items():
        print(f"   {k:<18}: {v:.3f}")

    # If a "new_data.csv" exists, compare drift
    if os.path.exists("data/new_data.csv"):
        new = pd.read_csv("data/new_data.csv")
        print("\nDrift check vs new_data.csv:")
        for col in means:
            if col in new.columns:
                baseline = means[col]
                current = new[col].mean()
                drift_pct = abs(current - baseline) / (abs(baseline) + 1e-9) * 100
                flag = "DRIFT" if drift_pct > 30 else "ok"
                print(f"   {col:<18}: {drift_pct:5.1f}% [{flag}]")
    else:
        print("\nNo data/new_data.csv found - skipping drift comparison.")
else:
    print("\nNo cleaned data found - skipping drift check.")

# ----------------------------------------
# 3. Monitoring services health check
#    Checks whether Prometheus and Grafana URLs are reachable.
#    These only run when the docker-compose monitoring stack is up,
#    so we report status without failing the pipeline if they are down.
# ----------------------------------------
try:
    import requests
except ImportError:
    requests = None

print("\n" + "-" * 60)
print("Monitoring services")
print("-" * 60)

services = {
    "Prometheus": "http://localhost:9090/-/healthy",
    "Grafana": "http://localhost:3000/api/health",
}

if requests is None:
    print("requests not installed - skipping URL checks")
else:
    for name, url in services.items():
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"[OK]   {name:<12} reachable at {url}")
            else:
                print(f"[WARN] {name:<12} returned status {r.status_code}")
        except Exception:
            print(f"[DOWN] {name:<12} not reachable "
                  f"(is the docker-compose stack running?)")

print("\n" + "=" * 60)
print("Monitoring completed")
print("=" * 60)