import pandas as pd
from utils import upload_df_to_cloud_storage


def load_data(
    df: pd.DataFrame,
    bucket_name: str = "kuldeep-iris-dataset",
    destination_blob_name: str = "iris-dataset/processed_data.csv",
) -> None:
    """
    Upload a Pandas DataFrame as a CSV file to a specified Google Cloud Storage bucket and blob.

    This function takes a Pandas DataFrame and uploads it as a CSV file to the specified
    Google Cloud Storage bucket and destination blob name.

    Args:
        df (pd.DataFrame): The Pandas DataFrame to be uploaded.
        bucket_name (str, optional): The name of the Google Cloud Storage bucket.
            Defaults to "kuldeep-iris-dataset".
        destination_blob_name (str, optional): The destination blob name (path and filename) in the bucket.
            Defaults to "iris-dataset/processed_data.csv".

    Returns:
        None
    """

    upload_df_to_cloud_storage(
        df=df,
        bucket_name=bucket_name,
        destination_blob_name=destination_blob_name,
    )

    return
