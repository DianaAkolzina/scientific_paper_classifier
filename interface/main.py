import os
from data.getdata import get_data_from_gcp
from models.baseline_model import vectorize_data, initialize_model, train_svm_model, evaluate_model
from models.preprocess import preprocessing_pipeline
from sklearn.model_selection import train_test_split

#Pull the data
BUCKET_NAME = os.getenv("BUCKET_NAME")
filename = ''

df = get_data_from_gcp(BUCKET_NAME, filename)

#Preprocess the data
text_column = df['Main Body']
author_column = df['Author']
label_column = df['Label']

preprocessed_data = preprocessing_pipeline(df, text_column, author_column, label_column)

#Split data into features and target
X = preprocessed_data['Processed Text']
y = preprocessed_data['Label']

#Vectorize the data
X_tfidf = vectorize_data(X)

#Train, test, split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

#Initialize, train and evaluate a model
svm_model = initialize_model()
trained_model = train_svm_model(svm_model, X_train, y_train)
evaluate_model(trained_model, X_test, y_test)
