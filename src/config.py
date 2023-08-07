import logging
import sys
from pathlib import Path

import mlflow

# Set SEED
SEED = 3

# Config Model
INPUT_DIM = 4
NUM_CLASSES = 3
LR = 0.01
NUM_EPOCHS = 100

# Tuning Parameters
TUNING_CONFIG = {
    "num_epochs": [10, 50, 100],
    "learning_rates": [0.05, 0.01, 0.001],
    "layer1_dims": [64, 128, 256],
    "layer2_dims": [32, 64, 128],
    "activation_functions": ["relu"],
}

# Config Directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
LOGS_DIR = Path(ROOT_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(ROOT_DIR, "data")
DATA_FILE = "raw_data.csv"
PROCESSED_DATA_FILE = "processed_data.csv"

OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Create logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

# Logger
logging.config.dictConfig(logging_config)
logger = logging.getLogger()
