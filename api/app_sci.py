import os
import pandas as pd
import torch
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pickle
import joblib
from google.cloud import storage
from transformers import BertTokenizer, BertModel
from torch import nn
import logging
import spacy
from langdetect import detect, LangDetectException
import string
import re
from collections import Counter
from unidecode import unidecode
import io
from sklearn.feature_extraction.text import CountVectorizer
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.gcp.internal.clients import bigquery
from apache_beam.io.gcp.bigquery_tools import TableReference
from apache_beam.options.pipeline_options import SetupOptions, WorkerOptions
from apache_beam.io.gcp.bigquery import ReadFromBigQuery, WriteToBigQuery


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory="templates")

app = FastAPI()

app.state = type('', (), {})()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = unidecode(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d', ' ', text)
    return text

def lemmatize_text(text):
    doc = nlp(text)
    ents = set([ent.text for ent in doc.ents])
    filtered_tokens = []
    for token in doc:
        if not token.is_stop and len(token.lemma_) > 2 and token.lemma_.isalpha() \
           and token.pos_ not in ['PRON', 'DET'] and token.text not in ents:
            filtered_tokens.append(token.lemma_)
    return ' '.join(filtered_tokens)

def extract_middle_text(text, word_count=400):
    words = text.split()
    total_words = len(words)
    if total_words <= word_count:
        return text
    start_index = (total_words - word_count) // 2
    return ' '.join(words[start_index:start_index + word_count])

def preprocessing_pipeline_sample(text):
    text = preprocess_text(text)
    text = lemmatize_text(text)
    text = extract_middle_text(text)
    return text

class SciBERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(SciBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        linear1_output = self.linear1(pooled_output)
        relu_output = self.relu(linear1_output)
        final_output = self.linear2(relu_output)
        return final_output

def save_vectorizer_to_gcs(vectorizer, bucket_name, vectorizer_blob_name):
    vectorizer_filename = 'vectorizer.pkl'

    with open(vectorizer_filename, 'wb') as file:
        pickle.dump(vectorizer, file)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    vectorizer_blob = bucket.blob(vectorizer_blob_name)
    vectorizer_blob.upload_from_filename(vectorizer_filename)

    os.remove(vectorizer_filename)

    print(f"Vectorizer saved to GCS bucket {bucket_name} under {vectorizer_blob_name}")

def load_vectorizer_from_gcs(bucket_name, vectorizer_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    vectorizer_blob = bucket.blob(vectorizer_blob_name)

    vectorizer_filename = 'downloaded_vectorizer.pkl'

    vectorizer_blob.download_to_filename(vectorizer_filename)

    with open(vectorizer_filename, 'rb') as file:
        vectorizer = pickle.load(file)

    os.remove(vectorizer_filename)

    return vectorizer

def load_model_from_gcp(bucket_name, blob_name):
    logger.info("Loading model from GCP bucket: %s", bucket_name)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    model_filename = 'downloaded_model'
    blob.download_to_filename(model_filename)

    if blob_name.endswith('.joblib'):
        model = joblib.load(model_filename)
    elif blob_name.endswith('.pth'):
        model_class = SciBERTClassifier
        num_labels = 2
        model = model_class(num_labels)
        model.load_state_dict(torch.load(model_filename))
    else:
        raise ValueError("Unsupported model file format")

    os.remove(model_filename)
    logger.info("Model loaded successfully")
    return model

BUCKET_NAME = os.getenv("BUCKET_NAME")
logger.info("Loading models from GCP bucket: %s", BUCKET_NAME)
app.state.baseline_model = load_model_from_gcp(BUCKET_NAME, 'models/baseline_model.joblib')
app.state.scibert_model = load_model_from_gcp(BUCKET_NAME, 'models/scibert_model.pth')

def predict_baseline(new_text):
    model = app.state.baseline_model
    tfidf = load_vectorizer_from_gcs(BUCKET_NAME, 'models/x_tfidf.pkl')
    processed_new_text = preprocessing_pipeline_sample(new_text)
    new_text_tfidf = tfidf.transform([processed_new_text])
    predicted_label = model.predict(new_text_tfidf)
    print("Predicted Label (Baseline):", predicted_label[0])
    return predicted_label[0], processed_new_text

def predict_scibert(new_text):
    model = app.state.scibert_model
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
    max_length = 256

    def predict_label(text, model, tokenizer):
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        predicted_label = torch.argmax(outputs).item()
        return predicted_label

    processed_new_text = preprocessing_pipeline_sample(new_text)
    words = processed_new_text.split()
    chunk_size = 400
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    predicted_labels = [predict_label(chunk, model, tokenizer) for chunk in chunks]
    class_names = ['pseudoscientific', 'scientific']
    print("Predicted Labels (SciBERT):", [class_names[label] for label in predicted_labels])
    return predicted_labels, processed_new_text

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify")
async def classify(request: Request, text: str = Form(...)):
    logger.info("Received POST request with text: %s", text)
    baseline_model = app.state.baseline_model
    scibert_model = app.state.scibert_model
    assert baseline_model is not None, "Baseline model not loaded"
    assert scibert_model is not None, "SciBERT model not loaded"

    logger.info("Predicting with Baseline Model...")
    baseline_prediction, processed_text_baseline = predict_baseline(text)

    logger.info("Predicting with SciBERT Model...")
    scibert_predictions, processed_text_scibert = predict_scibert(text)

    if baseline_prediction == 0:
        baseline_classifier = "scientific"
    else:
        baseline_classifier = "pseudoscientific"

    scibert_classifier_counts = Counter([int(pred) for pred in scibert_predictions])
    scibert_probability = scibert_classifier_counts[0] / len(scibert_predictions)

    logger.info("Prediction completed. Baseline: %s, SciBERT: %s", baseline_classifier, scibert_predictions)

    words = processed_text_scibert.split()
    word_counts = Counter(words)
    bigrams = list(zip(words[:-1], words[1:]))
    bigram_counts = Counter(bigrams)

    frequent_words = [word for word, count in word_counts.most_common(3) if count > 2]
    frequent_bigrams = [' '.join(bigram) for bigram, count in bigram_counts.most_common(3) if count > 2]

    num_words = len(words)
    num_unique_words = len(set(words))
    num_sentences = processed_text_scibert.count('.') + 1

    return {
        "baseline_prediction": baseline_classifier,
        "scibert_predictions": [str(pred) for pred in scibert_predictions],
        "scibert_probability": scibert_probability,
        "text": text,
        "frequent_words": frequent_words,
        "frequent_bigrams": frequent_bigrams,
        "num_words": num_words,
        "num_unique_words": num_unique_words,
        "num_sentences": num_sentences
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
