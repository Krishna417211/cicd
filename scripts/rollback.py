"""
Roll the Production model back to a previous version.

Usage:
    python scripts/rollback.py <version_number>

Example:
    python scripts/rollback.py 3
"""
import sys
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

MODEL_NAME = "TaxiFareModel"

print("=" * 60)
print("Model Rollback")
print("=" * 60)

if len(sys.argv) != 2:
    print("Usage: python scripts/rollback.py <version_number>")
    sys.exit(1)

target_version = sys.argv[1]

# ----------------------------------------
# Archive the current Production model
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
# Promote the target (older) version back to Production
# ----------------------------------------
print(f"Rolling back to version {target_version}")
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=target_version,
    stage="Production",
)

print("\n" + "=" * 60)
print(f"Rolled back. Version {target_version} is now in Production")
print("=" * 60)