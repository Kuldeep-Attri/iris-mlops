from datetime import datetime as dt

import settings
from etl.extract import extract_data
from utils import get_logger

logger = get_logger("data_pipeline")


def run():
    logger.info(
        f'Extracting data from source csv at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )
    data = extract_data(
        file_name=settings.DATA_FILE, data_dir=settings.DATA_DIR
    )
    logger.info(
        f'Extracted data from source csv at: {dt.now().strftime("%Y-%m-%d %H:%M:%S")} JST'
    )

    logger.info(data.head(3))


if __name__ == "__main__":
    run()
