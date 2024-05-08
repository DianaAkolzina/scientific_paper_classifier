import io
import os
import pandas as pd
from google.cloud import storage

BUCKET_NAME = os.getenv("BUCKET_NAME")

def get_data_from_gcp(BUCKET_NAME, file_name):
    # Create a storage client
    storage_client = storage.Client()
    # Get the bucket
    bucket = storage_client.get_bucket(BUCKET_NAME)
    # Get the blob (file) from the bucket
    blob = bucket.get_blob(file_name)
    # Read the CSV file into a DataFrame
    df = pd.read_csv(io.BytesIO(blob.download_as_bytes()), sep=",")
    return df
