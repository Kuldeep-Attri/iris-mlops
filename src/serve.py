from typing import List

import mlflow
import torch
from fastapi import FastAPI

from utils import mlflow_config

app = FastAPI()

mlflow = mlflow_config()
MODEL = mlflow.pytorch.load_model(f"models:/iris-production-model/1")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the PyTorch Model API"}


@app.post("/predict/")
async def predict(input_data: List[float]):
    """
    Predict the class label based on the input data.

    Args:
        input_data (List[float]): The input data for prediction.

    Returns:
        dict: A dictionary containing the predicted class label.
    """
    sample = torch.FloatTensor([input_data])
    output = MODEL(sample)
    prediction = torch.argmax(output)
    prediction = prediction.item()

    return {"prediction": prediction}
