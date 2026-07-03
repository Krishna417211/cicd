import json
import mlflow
from mlflow.tracking import MlflowClient

# ----------------------------------------
# MLflow Configuration
# ----------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# ----------------------------------------
# Read the winner chosen by compare_model.py
# ----------------------------------------
with open("best_model.json", "r") as f:
    best = json.load(f)

MODEL_NAME = best["model_name"]
BEST_VERSION = best["version"]

print("=" * 60)
print("Promoting Best Model to Production")
print("=" * 60)
print(f"Model   : {MODEL_NAME}")
print(f"Version : {BEST_VERSION}")

# ----------------------------------------
# Archive any current Production model
# ----------------------------------------
for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
    if v.current_stage == "Production":
        print(f"Archiving current Production model (version {v.version})")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v.version,
            stage="Archived",
        )

# ----------------------------------------
# Promote the winner
# ----------------------------------------
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=BEST_VERSION,
    stage="Production",
)

print("\n" + "=" * 60)
print(f"Version {BEST_VERSION} is now in Production")
print("=" * 60)