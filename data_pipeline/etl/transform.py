from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def remove_columns(df: pd.DataFrame, columns: List = ["id"]) -> pd.DataFrame:
    """
    Remove specified columns from the input DataFrame and return a new DataFrame with remaining columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (List, optional): List of columns to be removed from the DataFrame. Defaults to ["id"].

    Returns:
        pd.DataFrame: New DataFrame with the specified columns removed.
    """

    data = df.copy()
    return data.drop(columns=columns)


def rename_columns(
    df: pd.DataFrame,
    column_names: List = [
        "sepal_lenght",
        "sepal_width",
        "petal_lenght",
        "petal_width",
        "target",
    ],
) -> pd.DataFrame:
    """
    Rename columns of the input DataFrame and return a new DataFrame with updated column names.

    This function creates a new DataFrame by assigning new column names to the input DataFrame's columns.
    The input DataFrame remains unchanged.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_names (List, optional): List of new column names to be assigned. Defaults to common column names.

    Returns:
        pd.DataFrame: A new DataFrame with the updated column names.
    """

    data = df.copy()
    data.columns = column_names

    return data


def encode_label(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode labels in the 'target' column of the input DataFrame and return a new DataFrame.

    This function creates a new DataFrame by encoding labels in the 'target' column to numerical values.
    The mapping used for encoding is:
    - 'Iris-setosa' -> 0
    - 'Iris-versicolor' -> 1
    - 'Iris-virginica' -> 2

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'target' column with label strings.

    Returns:
        pd.DataFrame: A new DataFrame with labels in the 'target' column encoded.
    """

    data = df.copy()

    data["target"] = data["target"].map(
        {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    )

    return data


# def data_normalization()
