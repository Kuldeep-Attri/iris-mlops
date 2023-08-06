import json
from pathlib import Path

import mlflow

import config


def mlflow_config():
    """
    Configure MLflow tracking.

    This function sets up the configuration for MLflow tracking by creating a local directory for model registry,
    setting the MLFLOW_TRACKING_URI to use the local directory, and returning the configured mlflow object.

    Returns:
        mlflow.tracking.MlflowClient: The configured mlflow object for tracking experiments and models.
    """
    MODEL_REGISTRY = config.MODEL_REGISTRY
    Path(MODEL_REGISTRY).mkdir(parents=True, exist_ok=True)
    MLFLOW_TRACKING_URI = "file://" + str(MODEL_REGISTRY.absolute())
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    return mlflow


def save_json(data: dict, file_name: str, save_dir: str = config.OUTPUT_DIR):
    """
    Save a dictionary as a JSON file.

    Args:
        data: data to save.
        file_name: Name of the JSON file.
        save_dir: Directory to save the JSON file.

    Returns: None
    """

    data_path = Path(save_dir) / file_name
    with open(data_path, "w") as f:
        json.dump(data, f)


def load_json(file_name: str, save_dir: str = config.OUTPUT_DIR) -> dict:
    """
    Load a JSON file.

    Args:
        file_name: Name of the JSON file.
        save_dir: Directory of the JSON file.

    Returns: Dictionary with the data.
    """

    data_path = Path(save_dir) / file_name
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cached JSON from {data_path} does not exist."
        )

    with open(data_path, "r") as f:
        return json.load(f)
