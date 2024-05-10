#Imports
import os
from google.cloud import storage
import pandas as pd
import io
from data.getdata import get_data_from_gcp
from models.Transfer_learning_sciBERT import scibert_model_test
from models.preprocess import preprocessing_pipeline, preprocessing_pipeline_sample
from models.gcp_import import concat_dfs, save_data_to_gcp
from sklearn.model_selection import train_test_split

#Data collecting
BUCKET_NAME = os.getenv("BUCKET_NAME")
filename = 'cleaned_data/Updated_df_300.csv'
df = get_data_from_gcp(BUCKET_NAME, filename)

#load datasets
file_name_list = ['cleaned_data/arxiv_cleaned.csv',
                    , 'cleaned_data/arxiv_2_cleaned.csv'\
                    , 'cleaned_data/arxiv_3_cleaned.csv'\
                    , 'cleaned_data/arxiv_4_cleaned_cleaned.csv'\
                    , 'cleaned_data/astro_lipton_cleaned.csv'\
                    , 'cleaned_data/collective_evolution_cleaned.csv'\
                    , 'cleaned_data/collective_evolution_2_cleaned.csv'\
                    , 'cleaned_data/david_cleaned.csv'\
                    , 'cleaned_data/ftcheck_cleaned.csv'\
                    , 'cleaned_data/isha_cleaned.csv'\
                    , 'cleaned_data/naturalnews_cleaned.csv']

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

# Train, test, split
X_train, X_test, y_train, y_test = train_test_split(combined_df y, test_size=0.2, random_state=42)

# Preprocess the data
text_column = 'Processed Text'
author_column = 'Author'
label_column = 'Label'
X_train_processed = preprocessing_pipeline(X_train, text_column, author_column, label_column)

X_test_processed = preprocessing_pipeline_sample(X_test)


#Call the SciBERT model
scibert_model_test(X_train_processed)
scibert_model_test(X_test_processed)
