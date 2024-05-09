import os
import pandas as pd
from data.getdata import get_data_from_gcp
from models.baseline_model import initialize_model, train_svm_model, evaluate_model
from models.preprocess import preprocessing_pipeline, preprocessing_pipeline_sample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Pull the data
BUCKET_NAME = os.getenv("BUCKET_NAME")
filename = 'cleaned_data/Updated_df_3000.csv'
df = get_data_from_gcp(BUCKET_NAME, filename)

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

new_text = '''
China’s centralized efforts to contain the epidemic
'''


processed_new_text = preprocessing_pipeline_sample(new_text)
new_text_tfidf = tfidf.transform([processed_new_text])
predicted_label = trained_model.predict(new_text_tfidf)
print("Predicted Label:", predicted_label[0])
