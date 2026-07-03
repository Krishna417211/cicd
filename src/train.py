import random
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from tensorflow.keras.callbacks import EarlyStopping

from mlflow.models.signature import infer_signature

# -----------------------------
# Reproducibility
# -----------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# MLflow Configuration
# -----------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("taxi_experiment")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/taxi_clean.csv")

X = df.drop(columns=["total_amount"])
y = df["total_amount"]

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# -----------------------------
# Build Model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

# -----------------------------
# Train
# -----------------------------
with mlflow.start_run() as run:

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    # Predictions
    y_pred = model.predict(x_test, verbose=0).flatten()

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # -----------------------------
    # Log Parameters
    # -----------------------------
    mlflow.log_param("epochs", 50)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss", "mse")

    # -----------------------------
    # Log Metrics
    # -----------------------------
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)

    # numpy example produces TensorSpec, not ColSpec
    input_example = x_train.iloc[:5].to_numpy().astype("float32")

    signature = infer_signature(
        x_train.to_numpy().astype("float32"),
        model.predict(x_train[:5], verbose=0),
    )

    model_info = mlflow.tensorflow.log_model(
        model=model,
        name="model",
        registered_model_name="TaxiFareModel",
        signature=signature,
        input_example=input_example,
    )

    # -----------------------------
    # Save Model
    # -----------------------------
    model.save("models/taxi_model.keras")

    print("\n==============================")
    print("Training Completed")
    print("==============================")
    print("Run ID :", run.info.run_id)
    print("Model URI :", model_info.model_uri)
    print(f"R2 Score : {r2:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"MAE      : {mae:.4f}")
    print("==============================")
    print("Model registered in MLflow")
    print("Model saved to models/taxi_model.keras")