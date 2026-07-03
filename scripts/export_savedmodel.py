"""
Export the trained .keras model into TensorFlow SavedModel format,
which is what TensorFlow Serving requires.

TF Serving expects this folder layout:
    models/taxi_model/1/        <- version number folder
        saved_model.pb
        variables/
        assets/

The number "1" is the model version. TF Serving serves the highest number.
"""
import os
import tensorflow as tf

SRC = "models/taxi_model.keras"
EXPORT_DIR = "models/taxi_model/1"   # version 1

print("=" * 60)
print("Exporting model for TensorFlow Serving")
print("=" * 60)

if not os.path.exists(SRC):
    raise SystemExit(f"{SRC} not found. Train the model first.")

model = tf.keras.models.load_model(SRC)

os.makedirs(EXPORT_DIR, exist_ok=True)

# Keras 3 uses model.export() to write a TF SavedModel.
model.export(EXPORT_DIR)

print(f"\nSavedModel written to: {EXPORT_DIR}")
print("Folder contents:")
for item in os.listdir(EXPORT_DIR):
    print("   ", item)

print("\n" + "=" * 60)
print("Export complete. TF Serving can now serve this model.")
print("=" * 60)