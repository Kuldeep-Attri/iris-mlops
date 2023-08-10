from datetime import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import config
from config import logger
from etl.extract import extract_data
from etl.load import load_data
from etl.transform import transform_data


def etl_data(
    file_name: str = config.DATA_FILE,
    data_dir: str = config.DATA_DIR,
):
    """
    Perform ETL (Extract, Transform, Load) pipeline on data.

    This function orchestrates the ETL process by sequentially extracting data, transforming it, and then loading
    the transformed data. It logs timestamps and important steps along the way.

    Args:
        file_name (str, optional): The name of the source data file. Defaults to the value specified in the module settings.
        data_dir (str, optional): The directory containing the source data file. Defaults to the value specified
            in the module settings.

    Returns:
        pd.DataFrame: The extracted, transformed and loaded data.
    """

    logger.info(
        f'Extracting data from source csv at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )
    df = extract_data(file_name=file_name, data_dir=data_dir)
    logger.info(
        f'Extracted data from source csv at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )

    logger.info(
        f'Transforming data at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )
    df = transform_data(df=df)
    logger.info(
        f'Transformed data at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )

    logger.info(
        f'Loading data at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )
    df = load_data(df=df)
    logger.info(
        f'Loaded data at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )

    logger.info(
        f'Finished Running Data Pipeline at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )

    return df


def stratify_split(
    file_name: str = config.DATA_FILE,
    data_dir: str = config.DATA_DIR,
    target_column_name: str = "target",
    train_fraction: float = 0.7,
) -> Tuple[
    torch.FloatTensor,
    torch.LongTensor,
    torch.FloatTensor,
    torch.LongTensor,
    torch.FloatTensor,
    torch.LongTensor,
]:
    """
    Perform Stratified Train-Validation-Test Split on ETL-transformed data.

    This function performs a stratified train-validation-test split on the ETL-transformed data.
    It extracts, transforms, and loads the data using the ETL process before splitting it into subsets.

    Args:
        file_name (str, optional): The name of the source data file. Defaults to the value specified in the module settings.
        data_dir (str, optional): The directory containing the source data file. Defaults to the value specified
            in the module settings.
        target_column_name (str, optional): The name of the column containing the target labels. Defaults to 'target'.
        train_fraction (float, optional): The fraction of the data to be used for training. Defaults to 0.7.

    Returns:
        Tuple[
            torch.FloatTensor,  # X_train
            torch.LongTensor,   # y_train
            torch.FloatTensor,  # X_val
            torch.LongTensor,   # y_val
            torch.FloatTensor,  # X_test
            torch.LongTensor    # y_test
        ]: A tuple containing PyTorch Tensors for X_train, y_train, X_val, y_val, X_test, and y_test.
    """

    df = etl_data(file_name=file_name, data_dir=data_dir)

    X = df.drop(columns=[target_column_name]).values
    y = df[target_column_name].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1 - train_fraction),
        stratify=y,
        random_state=config.SEED,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=config.SEED,
    )

    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test
