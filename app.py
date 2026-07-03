from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from prometheus_client import Counter, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

import tensorflow as tf
import numpy as np
import time

app = FastAPI()

# -----------------------------
# Load TensorFlow Model
# -----------------------------
model = tf.keras.models.load_model("models/taxi_model.keras")

# -----------------------------
# Prometheus Metrics
# -----------------------------
prediction_requests = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

# -----------------------------
# Input Schema
# Model was trained on 4 features in this order:
#   passenger_count, trip_distance, fare_amount, tip_amount
# -----------------------------
class TaxiInput(BaseModel):
    passenger_count: int
    trip_distance: float
    fare_amount: float
    tip_amount: float

# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "API Running"}

# -----------------------------
# Prediction Route
# -----------------------------
@app.post("/predict")
def predict(data: TaxiInput):

    prediction_requests.inc()
    start = time.time()

    features = np.array([[
        data.passenger_count,
        data.trip_distance,
        data.fare_amount,
        data.tip_amount,
    ]], dtype=np.float32)

    prediction = model.predict(features, verbose=0)

    prediction_latency.observe(time.time() - start)

    return {"predicted_total_amount": float(prediction[0][0])}

# -----------------------------
# Metrics Route
# -----------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

# -----------------------------
# Automatic FastAPI Metrics
# -----------------------------
Instrumentator().instrument(app).expose(app)