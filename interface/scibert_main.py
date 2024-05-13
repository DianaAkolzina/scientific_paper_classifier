import os
from data.getdata import get_data_from_gcp
from models.Transfer_learning_sciBERT import scibert_model_test
from models.preprocess import preprocessing_pipeline, preprocessing_pipeline_sample
from models.gcp_import import concat_dfs, save_data_to_gcp
from sklearn.model_selection import train_test_split
import pandas as pd

#service_account_email = os.getenv("service_account_email")
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_email
def main():
    BUCKET_NAME=os.getenv("BUCKET_NAME")
    #load datasets
    file_name_list = ['cleaned_data/arxiv_cleaned.csv'\
                    , 'cleaned_data/arxiv_2_cleaned.csv'\
                    , 'cleaned_data/arxiv_3_cleaned.csv'\
                    , 'cleaned_data/arxiv_4_cleaned.csv'\
                    , 'cleaned_data/astro_lipton_cleaned.csv'\
                    , 'cleaned_data/collective_evolution_cleaned.csv'\
                    , 'cleaned_data/collective_evolution_2_cleaned.csv'\
                    , 'cleaned_data/david_cleaned.csv'\
                    , 'cleaned_data/ftcheck_cleaned.csv'\
                    , 'cleaned_data/isha_cleaned.csv'\
                    , 'cleaned_data/naturalnews_cleaned.csv']
    # Initialize an empty dictionary to store dataframes
    df_dict = {}

    # Loop through file names and get data from GCP
    for index, file in enumerate(file_name_list):
        df_dict[index] = get_data_from_gcp(BUCKET_NAME, file)

    print("Concatenating dataframes...")
    combined_df = concat_dfs(df_dict)
    print("Combined DataFrame:")
    print(combined_df)

    # Save concatenated DataFrame to GCP
    save_data_to_gcp(combined_df, 'cleaned_data/concatenated_df.csv')
    print("Concatenated DataFrame saved to GCP")
    print(combined_df.keys())

    #Add column names?
    column_names = ['Title', 'Author', 'Published Date', 'Word Count',
       'Most Frequent Word', 'Link', 'Label',
       'Primary Category', 'All Categories', 'Processed Text']
    combined_df.columns = column_names

    # Create X and y
    X = combined_df['Processed Text']
    y = combined_df['Label']
    print("X & y created")

    # Train, test, split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("X_train, X_test, y_train, y_test created")

    # Preprocess the data
    text_column = 'Processed Text'
    author_column = 'Author'
    label_column = 'Label'
    X_train_processed = preprocessing_pipeline(X_train, text_column, author_column, label_column)
    print("X_train processed")

    X_test_processed = preprocessing_pipeline_sample(X_test)
    print("X_test processed")

    # Call the SciBERT model for testing
    scibert_model_test(X_train_processed)
    scibert_model_test(X_test_processed)

if __name__ == "__main__":
    main()
