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
from mlflow.tracking.client import MlflowClient

import config
from config import logger
from models import SimpleNeuralNetwork
from prepare_data import stratify_split
from utils import mlflow_config

# Constants
TUNING_EXPERIMENT_NAME = "iris-mlops-tuning-update"
TUNING_MODEL_NAME = "iris-tuned-model"
PRODUCTION_MODEL_NAME = "iris-production-model"

# Initialize Typer CLI app
app = typer.Typer()


def register_production_model(mlflow):
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

        try:
            production_model_info = mlflow.models.get_model_info(
                f"models:/{PRODUCTION_MODEL_NAME}/Production"
            )
            production_model_run_id = production_model_info.run_id
            production_model_performance = mlflow.get_run(
                production_model_run_id
            ).data.metrics["val_loss"]
            new_model_performance = new_model_run.data.metrics["val_loss"]

            if new_model_performance < production_model_performance:
                new_model_version = mlflow.register_model(
                    model_uri=f"runs:/{new_model_run_id}/model",
                    name=PRODUCTION_MODEL_NAME,
                )
                mlflow_client.transition_model_version_stage(
                    name=PRODUCTION_MODEL_NAME,
                    version=new_model_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(
                    "The new model outperformed the production model, so updated the production model."
                )
            else:
                logger.info(
                    "The new model did not outperform the production model, so no updates."
                )
        except:
            new_model_version = mlflow.register_model(
                model_uri=f"runs:/{new_model_run_id}/model",
                name=PRODUCTION_MODEL_NAME,
            )
            print(new_model_version)
            mlflow_client.transition_model_version_stage(
                name=PRODUCTION_MODEL_NAME,
                version=new_model_version._version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info("Registered the first production model.")
    else:
        logger.error(
            "No runs found. Please ensure you have run an experiment."
        )


def train_n_validate_model(
    model: torch.nn.Module,
    X_train: torch.FloatTensor,
    y_train: torch.LongTensor,
    X_val: torch.FloatTensor,
    y_val: torch.LongTensor,
    num_epochs: int,
    learning_rate: float,
) -> Tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    """
    Train and validate a PyTorch model.

    This function trains a given PyTorch model using the provided training data and validates it using the validation data.
    It computes and returns the training and validation losses for each epoch.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained and validated.
        X_train (torch.FloatTensor): The training input data as a PyTorch tensor.
        y_train (torch.LongTensor): The training target labels as a PyTorch tensor.
        X_val (torch.FloatTensor): The validation input data as a PyTorch tensor.
        y_val (torch.LongTensor): The validation target labels as a PyTorch tensor.
        num_epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for optimization.

    Returns:
        Tuple[torch.nn.Module, np.ndarray, np.ndarray]: A tuple containing the trained model,
        the training losses for each epoch, and the validation losses for each epoch.
    """

    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        output_train = model(X_train)
        loss_train = criterion(output_train, y_train)
        loss_train.backward()
        optimizer.step()

        loss_train = loss_train.item()
        train_losses[epoch] = loss_train

        model.eval()
        output_val = model(X_val)
        loss_val = criterion(output_val, y_val)

        loss_val = loss_val.item()
        val_losses[epoch] = loss_val

    return model, train_losses, val_losses


def test_model(
    model: nn.Module, X_test: torch.FloatTensor, y_test: torch.LongTensor
) -> float:
    """
    Test a PyTorch model and calculate its accuracy.

    This function evaluates a trained PyTorch model using the provided test data and calculates its accuracy.
    The accuracy is calculated as the ratio of correctly predicted samples to the total number of test samples.

    Args:
        model (nn.Module): The trained PyTorch model to be tested.
        X_test (torch.FloatTensor): The test input data as a PyTorch tensor.
        y_test (torch.LongTensor): The test target labels as a PyTorch tensor.

    Returns:
        float: The accuracy of the model on the test data.
    """

    model.eval()

    output_test = model(X_test)
    _, predicted_test = torch.max(output_test, 1)
    correct_predictions_test = (predicted_test == y_test).sum().item()
    total_test_samples = y_test.size(0) * 1.0

    accuracy = round(correct_predictions_test / total_test_samples, 4) * 100.0
    return accuracy


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

    # combinations = product(
    #     config.TUNING_CONFIG["num_epochs"],
    #     config.TUNING_CONFIG["learning_rates"],
    #     config.TUNING_CONFIG["layer1_dims"],
    #     config.TUNING_CONFIG["layer2_dims"],
    #     config.TUNING_CONFIG["activation_functions"],
    # )
    combinations = product(
        config.TUNING_CONFIG["num_epochs"][:1],
        config.TUNING_CONFIG["learning_rates"][:1],
        config.TUNING_CONFIG["layer1_dims"][:1],
        config.TUNING_CONFIG["layer2_dims"][:1],
        config.TUNING_CONFIG["activation_functions"][:1],
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

    register_production_model(mlflow=mlflow)


if __name__ == "__main__":
    app()
