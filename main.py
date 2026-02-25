import os
import random
import warnings

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from src.data_loader import load_data
from src.trainer import train_and_log_models


RANDOM_STATE = 42
EXPERIMENT_NAME = "Wine_Quality_MLOps"


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_experiment() -> None:
    """Configure SQLite tracking and ensure experiment metadata is present."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=EXPERIMENT_NAME,
            tags={
                "project": "Wine Quality Prediction MLOps",
                "mlflow.note.content": "Wine Quality MLOps experiment with model registry and promotion.",
            },
        )
    else:
        experiment_id = experiment.experiment_id
        client.set_experiment_tag(experiment_id, "project", "Wine Quality Prediction MLOps")
        client.set_experiment_tag(
            experiment_id,
            "mlflow.note.content",
            "Wine Quality MLOps experiment with model registry and promotion.",
        )

    mlflow.set_experiment(EXPERIMENT_NAME)


def main() -> None:
    set_global_seed(RANDOM_STATE)
    warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

    configure_experiment()

    train_x, test_x, train_y, test_y = load_data(random_state=RANDOM_STATE)
    train_and_log_models(train_x, test_x, train_y, test_y)


if __name__ == "__main__":
    main()
