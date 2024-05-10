import os
import pandas as pd
from data.getdata import get_data_from_gcp
from models.baseline_model import initialize_model, train_svm_model, evaluate_model
from models.preprocess import preprocessing_pipeline, preprocessing_pipeline_sample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

def get_data(data_size):
    # Pull the data
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    filename = f'cleaned_data/Updated_df_{data_size}.csv'
    df = get_data_from_gcp(BUCKET_NAME, filename)
    return df

def preprocess_data(df):
    # Preprocess the data
    text_column = 'Processed Text'
    author_column = 'Author'
    label_column = 'Label'
    preprocessed_data = preprocessing_pipeline(df, text_column, author_column, label_column)

    # Reset DataFrame and type labels
    combined_df = preprocessed_data
    combined_df.reset_index(drop=True, inplace=True)
    combined_df['Label'] = combined_df['Label'].astype('category')

    # Balance the dataset
    label_counts = combined_df['Label'].value_counts()
    min_category_size = label_counts.min()
    balanced_df = pd.concat([
        combined_df[combined_df['Label'] == label].sample(n=min_category_size, random_state=42)
        for label in combined_df['Label'].cat.categories
    ])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

def train_evaluate_model(balanced_df):
    X = balanced_df['Processed Text']
    y = balanced_df['Label']

    # Initialize and fit the TF-IDF Vectorizer
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X)

    # Train, test, split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Initialize, train and evaluate the SVM model
    svm_model = initialize_model()
    trained_model = train_svm_model(svm_model, X_train, y_train)

    class_names = ['1', '0']
    evaluate_model(trained_model, X_test, y_test, class_names)
    return trained_model

import joblib
from google.cloud import storage

def save_model_to_gcp(model, bucket_name, destination_blob_name):
    """Saves the model to a Google Cloud Storage bucket."""

    model_filename = 'model.joblib'
    joblib.dump(model, model_filename)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(model_filename)

    os.remove(model_filename)

    print(f"Model saved to GCS bucket {bucket_name} under {destination_blob_name}")


def load_model_from_gcp(bucket_name, source_blob_name):
    """Loads the model from a Google Cloud Storage bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    model_filename = 'downloaded_model.joblib'
    blob.download_to_filename(model_filename)

    model = joblib.load(model_filename)

    os.remove(model_filename)

    return model


def predict_baseline(new_text = '''
    China’s centralized efforts to contain the epidemic
    '''):

    model = load_model_from_gcp(BUCKET_NAME, 'models/baseline_model.joblib')
    processed_new_text = preprocessing_pipeline_sample(new_text)
    new_text_tfidf = tfidf.transform([processed_new_text])
    predicted_label = model.predict(new_text_tfidf)
    # print("Predicted Label:", predicted_label[0])
    return predicted_label[0]



if __name__== "__main__":
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    print("getting data...")
    data_main = get_data(data_size = 300)

    print("preprocessing data...")
    preprocess_main = preprocess_data(data_main)

    print("training model...")
    model_trained = train_evaluate_model(preprocess_main)

    print("saving model")
    save_model_to_gcp(model_trained, BUCKET_NAME, 'models/baseline_model.joblib')

    print("model saved")
