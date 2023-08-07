import sys
from typing import List

sys.path.append("./src")

import mlflow
import torch
import yaml
from fastapi import FastAPI

import config
from utils import mlflow_config

app = FastAPI()

mlflow = mlflow_config()

PRODUCTION_MODEL = None
TUNING_MODEL_NAME = "iris-tuned-model"

with open(
    str(config.ROOT_DIR / "iris_model_registry.yaml"), "r"
) as config_file:
    model_registry_config = yaml.safe_load(config_file)

PRODUCTION_MODEL_RUN_ID = model_registry_config["mlflow"]["production"][
    "latest_model_run_id"
]

# Check if we have loaded a production model or not.
if PRODUCTION_MODEL_RUN_ID != "ZZZ":
    PRODUCTION_MODEL = mlflow.pytorch.load_model(
        f"runs:/{PRODUCTION_MODEL_RUN_ID}/{TUNING_MODEL_NAME}"
    )
    print(PRODUCTION_MODEL)
else:
    print("No model registered for production.")


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
    output = PRODUCTION_MODEL(sample)
    prediction = torch.argmax(output)
    prediction = prediction.item()

    return {"prediction": prediction}
