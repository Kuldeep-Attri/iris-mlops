import warnings
from datetime import datetime as dt
from itertools import product
from typing import Optional, Tuple

warnings.simplefilter("ignore")

import dvc.api
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import typer
import yaml
from mlflow.tracking.client import MlflowClient

import config
from config import logger
from models import SimpleNeuralNetwork
from prepare_data import stratify_split
from train import test_model, train_n_validate_model
from utils import mlflow_config

# Constants
TUNING_EXPERIMENT_NAME = "iris-mlops-tuning"
TUNING_MODEL_NAME = "iris-tuned-model"
PRODUCTION_MODEL_NAME = "iris-production-model"

# Initialize Typer CLI app
app = typer.Typer()


def register_production_model(
    mlflow, production_model_run_id: str
) -> Optional[str]:
    """
    Register the best model from tuning as the production model if it outperforms the current production model.

    Parameters:
        mlflow: MLflow client instance.

    Returns:
        None
    """
    mlflow_client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

    runs = mlflow.search_runs(
        experiment_names=[TUNING_EXPERIMENT_NAME],
        order_by=["metrics.val_loss ASC"],
        max_results=1,
    )

    if not runs.empty:
        new_model_run = runs.iloc[0]
        new_model_run_id = new_model_run.run_id

        if production_model_run_id != "ZZZ":
            production_model_performance = mlflow.get_run(
                production_model_run_id
            ).data.metrics["val_loss"]
            new_model_performance = new_model_run["metrics.val_loss"]

            if new_model_performance < production_model_performance:
                new_model_version = mlflow.register_model(
                    model_uri=f"runs:/{new_model_run_id}/model",
                    name=PRODUCTION_MODEL_NAME,
                )
                mlflow_client.transition_model_version_stage(
                    name=PRODUCTION_MODEL_NAME,
                    version=new_model_version._version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(
                    "The new model outperformed the production model, so updated the production model."
                )
                return new_model_run_id
            else:
                logger.info(
                    "The new model did not outperform the production model, so no updates."
                )
                return
        else:
            new_model_version = mlflow.register_model(
                model_uri=f"runs:/{new_model_run_id}/model",
                name=PRODUCTION_MODEL_NAME,
            )
            mlflow_client.transition_model_version_stage(
                name=PRODUCTION_MODEL_NAME,
                version=new_model_version._version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info("Registered the first production model.")
            return new_model_run_id
    else:
        logger.error(
            "No runs found. Please ensure you have run an experiment."
        )
        return


def log_tuning_with_mlflow(
    mlflow,
    i: int,
    model: nn.Module,
    train_losses: np.array,
    val_losses: np.array,
    accuracy: float,
    num_epochs: int,
    learning_rate: float,
    l1_dim: int,
    l2_dim: int,
    act: str,
):
    """
    Log tuning details using MLFlow.
    """

    with mlflow.start_run(
        run_name=f"run-tuning-{i+1}",
        description=f"Models generated during hyper-parameter tunings.",
    ) as run:
        params = {
            "data_url": dvc.api.get_url(
                path=str(config.DATA_DIR / config.DATA_FILE)
            ),
            "input_dim": config.INPUT_DIM,
            "output_dim": config.NUM_CLASSES,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "layer1_dimension": l1_dim,
            "layer2_dimension": l2_dim,
            "activation": act,
        }
        mlflow.log_params(params=params)

        mlflow.pytorch.log_model(model, TUNING_MODEL_NAME)

        for i in range(len(list(train_losses))):
            mlflow.log_metrics({"train_loss": list(train_losses)[i]}, step=i)
            mlflow.log_metrics({"val_loss": list(val_losses)[i]}, step=i)
        mlflow.log_metric("test_accuracy", accuracy)


@app.command()
def tune():
    """
    Hyperparameter tuning function.

    This function performs hyperparameter tuning for a neural network model using the combinations of parameters
    specified in the TUNING_CONFIG. It trains, validates, tests, and tracks the model's performance using MLFlow.

    Returns:
        None
    """

    with open(
        str(config.ROOT_DIR / "iris_model_registry.yaml"), "r"
    ) as config_file:
        model_registry_config = yaml.safe_load(config_file)

    production_model_run_id = model_registry_config["mlflow"]["production"][
        "latest_model_run_id"
    ]

    combinations = product(
        config.TUNING_CONFIG["num_epochs"],
        config.TUNING_CONFIG["learning_rates"],
        config.TUNING_CONFIG["layer1_dims"],
        config.TUNING_CONFIG["layer2_dims"],
        config.TUNING_CONFIG["activation_functions"],
    )

    logger.info(
        f'Preparing data at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )
    X_train, y_train, X_val, y_val, X_test, y_test = stratify_split(
        file_name=config.DATA_FILE, data_dir=config.DATA_DIR
    )
    logger.info(
        f'Prepared data at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )

    mlflow = mlflow_config()
    mlflow.set_experiment(TUNING_EXPERIMENT_NAME)

    logger.info(
        f'Tuning & Tracking models at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )

    for i, combination in enumerate(combinations):
        num_epochs, learning_rate, l1_dim, l2_dim, act = combination

        model = SimpleNeuralNetwork(
            input_dim=config.INPUT_DIM,
            output_dim=config.NUM_CLASSES,
            layer1_dim=l1_dim,
            layer2_dim=l2_dim,
            act=act,
        )

        model, train_losses, val_losses = train_n_validate_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )

        accuracy = test_model(model=model, X_test=X_test, y_test=y_test)

        log_tuning_with_mlflow(
            mlflow=mlflow,
            i=i,
            model=model,
            train_losses=train_losses,
            val_losses=val_losses,
            accuracy=accuracy,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            l1_dim=l1_dim,
            l2_dim=l2_dim,
            act=act,
        )

    logger.info(
        f'Tuned & Tracked model at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )

    new_production_model_run_id = register_production_model(
        mlflow=mlflow, production_model_run_id=production_model_run_id
    )
    if new_production_model_run_id:
        with open(
            str(config.ROOT_DIR / "iris_model_registry.yaml"), "w"
        ) as config_file:
            model_registry_config["mlflow"]["production"][
                "latest_model_run_id"
            ] = new_production_model_run_id
            yaml.dump(
                model_registry_config, config_file, default_flow_style=False
            )


if __name__ == "__main__":
    app()
