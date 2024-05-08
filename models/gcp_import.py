from data.getdata import get_data_from_gcp
import io
import os
import pandas as pd
from google.cloud import storage

BUCKET_NAME = os.getenv("BUCKET_NAME")

# add new datasets to filename_list
file_name_list = ['cleaned_data/arxiv_cleaned.csv', 'cleaned_data/collective_evolution_cleaned.csv'\
                ,'cleaned_data/david_cleaned.csv', 'cleaned_data/ftcheck_cleaned.csv'\
                , 'cleaned_data/isha_cleaned.csv', 'cleaned_data/naturalnews_cleaned.csv'\
                , 'cleaned_data/astro_lipton_cleaned.csv']

#load datasets
df_dict={}

for index, file in enumerate(file_name_list):
    df_dict[index] = get_data_from_gcp(BUCKET_NAME, file)

# get concatenated dataset
def concat_dfs(df_dictionary):
    for index, key in enumerate(df_dictionary):
        if index == len(df_dictionary)-2:
            df = pd.concat((df_dictionary[index], df_dictionary[index+1]), join='inner')
        else:
            return df

#save combined dataset to gcp
def save_data_to_gcp(df, file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    csv_data = io.BytesIO()
    df.to_csv(csv_data, index=False)
    blob = bucket.blob(file_name)
    csv_data.seek(0)
    blob.upload_from_file(csv_data, content_type='text/csv')
    print(f"DataFrame saved to GCS bucket: {BUCKET_NAME}/cleaned_data/{file_name}")

print("concatenating dfs .....")
combined_df = concat_dfs(df_dict)
print("dfs are concatenated")

save_data_to_gcp(combined_df, 'cleaned_data/concatenated_df.csv')
print("concatenated dfs saved to gcp")
