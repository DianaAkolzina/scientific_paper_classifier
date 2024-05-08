import io
import pandas as pd
from google.cloud import storage

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

def save_data_to_gcp(BUCKET_NAME, file_name, df):
    # Create a storage client
    storage_client = storage.Client()
    # Get the bucket
    bucket = storage_client.get_bucket(BUCKET_NAME)
    # Create a blob (file) object
    blob = bucket.blob(file_name)
    # Convert the DataFrame to a CSV bytes object
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    # Upload the CSV bytes object to GCS
    blob.upload_from_file(csv_buffer, content_type='text/csv')
    print(f"DataFrame saved to GCS bucket: {BUCKET_NAME}/cleaned_data/{file_name}")
