from __future__ import annotations

from functools import lru_cache
import logging

import mlflow
import pandas as pd
from flask import Flask, jsonify, request


MODEL_URI = "models:/WineQualityBestModel/Production"

app = Flask(__name__)


@lru_cache(maxsize=1)
def get_model():
    """Load and cache the production model from MLflow Model Registry."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
    return mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/")
def health() -> str:
    return "MLflow Model Server Running"


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON payload"}), 400

    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        return jsonify({"error": "Payload must be a JSON object or list of objects"}), 400

    try:
        input_df = pd.DataFrame(rows)
        predictions = get_model().predict(input_df)
    except Exception as exc:  # pragma: no cover - runtime validation path
        return jsonify({"error": str(exc)}), 400

    return jsonify({"predictions": predictions.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
