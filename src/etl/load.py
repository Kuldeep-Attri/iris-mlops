import pandas as pd
import settings


def load_data(
    df: pd.DataFrame, processed_data_file: str = settings.PROCESSED_DATA_FILE
) -> pd.DataFrame:
    """
    Save a Pandas DataFrame as a CSV file.

    This function takes a Pandas DataFrame and saves it as a CSV file with the provided filename.
    By default, the file is saved with the specified filename from the module settings.

    Args:
        df (pd.DataFrame): The Pandas DataFrame to be saved.
        processed_data_file (str, optional): The path and filename to save the CSV file.
            Defaults to the value specified in the module settings.

    Returns:
        Input Pandas DataFrame
    """

    df.to_csv(processed_data_file, index=False)
    return df
