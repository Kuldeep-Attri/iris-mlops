from datetime import datetime as dt

import settings
from etl.extract import extract_data
from etl.load import load_data
from etl.transform import transform_data
from utils import get_logger

logger = get_logger("data_pipeline")


def run(
    file_name: str = settings.DATA_FILE,
    data_dir: str = settings.DATA_DIR,
):
    """
    Run the data pipeline including extraction, transformation, and loading steps.

    This function orchestrates the data pipeline by performing the following steps:
    1. Extract data from a source CSV file.
    2. Transform the extracted data.
    3. Load the transformed data.

    Args:
        file_name (str, optional): The name of the source CSV file. Defaults to settings.DATA_FILE.
        data_dir (str, optional): The directory path for the data files. Defaults to settings.DATA_DIR.

    Returns:
        None
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

    return


if __name__ == "__main__":
    run()
