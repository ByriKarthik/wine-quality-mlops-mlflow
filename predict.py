from __future__ import annotations

import logging

import mlflow
import pandas as pd


MODEL_URI = "models:/WineQualityBestModel/Production"


def build_sample_input() -> pd.DataFrame:
    """Create one sample row matching the wine quality feature schema."""
    return pd.DataFrame(
        [
            {
                "fixed acidity": 7.4,
                "volatile acidity": 0.70,
                "citric acid": 0.00,
                "residual sugar": 1.9,
                "chlorides": 0.076,
                "free sulfur dioxide": 11.0,
                "total sulfur dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            }
        ]
    )


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
    model = mlflow.pyfunc.load_model(MODEL_URI)
    sample_input = build_sample_input()
    prediction = model.predict(sample_input)
    print("Sample input:")
    print(sample_input.to_string(index=False))
    print(f"Prediction: {prediction.tolist()}")


if __name__ == "__main__":
    main()
