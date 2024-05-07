import pandas as pd
import glob
import os

directory_path = "data/raw/"

csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

def print_dataframe_info(file_path):
    try:

        df = pd.read_csv(file_path)

        print(f"File: {file_path}")

        print("Columns Info:")
        print(df.columns)

        print("First few rows:")
        print(df.head())
        print(df.columns)
        print("Description:")
        print(df.describe(include='all'))

        print("Missing values per column:")
        print(df.isnull().sum())

        print("length")
        print(len(df))
        print("-" * 50)

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

for file in csv_files:
    print_dataframe_info(file)
