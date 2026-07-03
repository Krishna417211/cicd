import json
import subprocess
import mlflow
from mlflow.tracking import MlflowClient

# ----------------------------------------
# MLflow Configuration
# ----------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "TaxiFareModel"

client = MlflowClient()

print("=" * 60)
print("Comparing Registered Model Versions")
print("=" * 60)

# ----------------------------------------
# Get all registered versions
# FIX: the model name must be quoted inside the filter string.
# ----------------------------------------
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

if len(versions) < 2:
    # Not enough versions yet: the current one is the best by default.
    print("Only one version found. Treating it as the best model.")
    only = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
    run = client.get_run(only.run_id)
    metrics = run.data.metrics
    winner = only
    winner_metrics = metrics
else:
    versions = sorted(versions, key=lambda x: int(x.version), reverse=True)

    latest = versions[0]
    previous = versions[1]

    latest_run = client.get_run(latest.run_id)
    previous_run = client.get_run(previous.run_id)

    latest_metrics = latest_run.data.metrics
    previous_metrics = previous_run.data.metrics

    latest_mse = latest_metrics.get("mse", float("inf"))
    previous_mse = previous_metrics.get("mse", float("inf"))

    print(f"\nLatest   -> version {latest.version}, mse {latest_mse:.4f}")
    print(f"Previous -> version {previous.version}, mse {previous_mse:.4f}")

    # Lower MSE wins
    if latest_mse < previous_mse:
        winner = latest
        winner_metrics = latest_metrics
    else:
        winner = previous
        winner_metrics = previous_metrics

# ----------------------------------------
# Get Current Git Commit (for traceability)
# ----------------------------------------
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    git_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
except Exception:
    git_commit = "unknown"
    git_branch = "unknown"

# ----------------------------------------
# Save the winner to best_model.json
# ----------------------------------------
best_model = {
    "model_name": MODEL_NAME,
    "version": winner.version,
    "run_id": winner.run_id,
    "mse": winner_metrics.get("mse"),
    "mae": winner_metrics.get("mae"),
    "r2_score": winner_metrics.get("r2_score"),
    "git_commit": git_commit,
    "git_branch": git_branch,
}

with open("best_model.json", "w") as f:
    json.dump(best_model, f, indent=4)

print("\n" + "=" * 60)
print(f"Best model: version {winner.version} (mse {winner_metrics.get('mse'):.4f})")
print("Saved to best_model.json")
print("=" * 60)