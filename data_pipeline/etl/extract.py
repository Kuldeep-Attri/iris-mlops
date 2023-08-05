from pathlib import Path
from typing import Optional

import pandas as pd
import utils
from yarl import URL

logger = utils.get_logger("extract")


def extract_data(file_name: str, data_dir: str) -> Optional[pd.DataFrame]:
    """Extract data in pd.DataFrame format from input csv file.

    Args:
        file_name (str): Input csv file name.
        data_dir (str): Location of the input csv file.

    Raises:
        FileNotFoundError: Input file location does not exist.

    Returns:
        Optional[pd.DataFrame]: A pd.Dataframe containing the data.
    """

    data_path = Path(data_dir) / file_name
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file at location {data_path} does not exist."
        )
    else:
        df = pd.read_csv(data_path)
        return df
