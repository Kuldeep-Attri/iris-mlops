from datetime import datetime as dt
from pathlib import Path
from typing import Tuple

import dvc.api
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from config import logger
from models import SimpleNeuralNetwork
from prepare_data import stratify_split
from utils import mlflow_config


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

    logger.info(
        f'Test Accuracy: {accuracy} % at {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )
    return accuracy


def train():
    """
    Train neural network model function.

    This function prepares data, trains a neural network model, validates its performance,
    tests it, and tracks the training process using MLFlow.

    Returns:
        None
    """

    logger.info(
        f'Preparing data at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )
    X_train, y_train, X_val, y_val, X_test, y_test = stratify_split(
        file_name=config.DATA_FILE, data_dir=config.DATA_DIR
    )
    logger.info(
        f'Prepared data at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )

    model = SimpleNeuralNetwork(
        input_dim=config.INPUT_DIM, output_dim=config.NUM_CLASSES
    )
    learning_rate, num_epochs = config.LR, config.NUM_EPOCHS

    mlflow = mlflow_config()
    mlflow.set_experiment("iris-mlops-training")

    logger.info(
        f'Training data at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
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
    logger.info(
        f'Trained data at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )

    logger.info(
        f'Testing data at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )
    accuracy = test_model(model=model, X_test=X_test, y_test=y_test)
    logger.info(
        f'Tested data at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )

    logger.info(
        f'Tracking training with MLFlow at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )
    with mlflow.start_run(
        run_name=f"run-training",
        description=f"Models generated during trainings",
    ) as run:
        params = {
            "data_url": dvc.api.get_url(
                path=str(config.DATA_DIR / config.DATA_FILE)
            ),
            "input_dim": config.INPUT_DIM,
            "output_dim": config.NUM_CLASSES,
            "num_epochs": config.NUM_EPOCHS,
            "learning_rate": config.LR,
        }
        mlflow.log_params(params=params)

        mlflow.pytorch.log_model(model, f"iris-trained-model")

        for i in range(len(list(train_losses))):
            mlflow.log_metrics({"train_loss": list(train_losses)[i]}, step=i)
            mlflow.log_metrics({"val_loss": list(val_losses)[i]}, step=i)
        mlflow.log_metric("test_accuracy", accuracy)

    logger.info(
        f'Tracked training with MLFlow at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )


if __name__ == "__main__":
    train()
