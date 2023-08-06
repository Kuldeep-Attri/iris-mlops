from pathlib import Path

import pandas as pd

import config


def load_data(
    df: pd.DataFrame,
    processed_data_file: str = config.PROCESSED_DATA_FILE,
    data_dir: str = config.DATA_DIR,
) -> pd.DataFrame:
    """
    Save a Pandas DataFrame as a CSV file.

    This function takes a Pandas DataFrame and saves it as a CSV file with the provided filename.
    By default, the file is saved in the specified directory and with the filename from the module settings.

    Args:
        df (pd.DataFrame): The Pandas DataFrame to be saved.
        processed_data_file (str, optional): The filename to save the CSV file. Defaults to the value
            specified in the module settings.
        data_dir (str, optional): The directory where the CSV file will be saved. Defaults to the value
            specified in the module settings.

    Returns:
        pd.DataFrame: The input Pandas DataFrame.
    """

    output_file = Path(data_dir, processed_data_file)
    df.to_csv(output_file, index=False)
    return df
