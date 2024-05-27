import os
import pandas as pd
import torch
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import joblib
from transformers import BertTokenizer, BertModel
from torch import nn
import logging
import spacy
from langdetect import detect, LangDetectException
import string
import re
from collections import Counter
from unidecode import unidecode
import matplotlib.pyplot as plt
import numpy as np
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory="templates")

app = FastAPI()

app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# Add state attribute to the app
app.state = type('State', (), {})()

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
        self.lstm = nn.LSTM(768, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_attentions=True)
        pooled_output = outputs[1]
        attention_weights = outputs[2]
        pooled_output = self.dropout(pooled_output)
        lstm_output, _ = self.lstm(pooled_output.unsqueeze(0))
        linear1_output = self.linear1(lstm_output.squeeze(0))
        logits = self.linear2(linear1_output)
        return logits, attention_weights

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
            outputs, attention_weights = model(input_ids, attention_mask)
        predicted_label = torch.argmax(outputs[0]).item()
        return predicted_label, attention_weights

    processed_new_text = preprocessing_pipeline_sample(new_text)
    words = processed_new_text.split()
    chunk_size = 400
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    predicted_labels = []
    attention_weights_list = []
    for chunk in chunks:
        label, attention_weights = predict_label(chunk, model, tokenizer)
        predicted_labels.append(label)
        attention_weights_list.append(attention_weights)
    class_names = ['pseudoscientific', 'scientific']
    logger.info("Predicted Labels (SciBERT): %s", [class_names[label] for label in predicted_labels])
    return predicted_labels, processed_new_text, attention_weights_list, chunks, tokenizer

def load_model_from_local(model_path):
    logger.info("Loading model from local path: %s", model_path)
    if model_path.endswith('.joblib') or model_path.endswith('.pkl'):
        model = joblib.load(model_path)
    elif model_path.endswith('.pth'):
        model_class = SciBERTClassifier
        num_labels = 2
        model = model_class(num_labels)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path), strict=False)
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    else:
        raise ValueError("Unsupported model file format")
    logger.info("Model loaded successfully")
    return model

def load_vectorizer_from_local(vectorizer_path):
    logger.info("Loading vectorizer from local path: %s", vectorizer_path)
    vectorizer = joblib.load(vectorizer_path)
    logger.info("Vectorizer loaded successfully")
    return vectorizer

# Get the absolute path of the parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the file paths relative to the parent directory
baseline_model_path = os.path.join(parent_dir, 'models', 'svm_model.pkl')
scibert_model_path = os.path.join(parent_dir, 'models', 'scibert_model.pth')
tfidf_vectorizer_path = os.path.join(parent_dir, 'models', 'tfidf_vectorizer.pkl')

logger.info("Loading models from local storage")
app.state.baseline_model = load_model_from_local(baseline_model_path)
app.state.scibert_model = load_model_from_local(scibert_model_path)

def predict_baseline(new_text):
    model = app.state.baseline_model
    tfidf = load_vectorizer_from_local(tfidf_vectorizer_path)
    processed_new_text = preprocessing_pipeline_sample(new_text)
    new_text_tfidf = tfidf.transform([processed_new_text])
    predicted_label = model.predict(new_text_tfidf)
    logger.info("Predicted Label (Baseline): %s", predicted_label[0])
    return predicted_label[0], processed_new_text

def visualize_attention(text, attention_weights, tokenizer, top_n=5):
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids'].squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    attention_weights = attention_weights[-1].squeeze().mean(dim=0).cpu().numpy()

    top_indices = np.argsort(attention_weights)[-top_n:].tolist()
    top_tokens = [tokens[i] for i in top_indices]
    top_attention_weights = [attention_weights[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(np.array(top_attention_weights).reshape(1, -1), cmap='Blues', aspect='auto')

    im.set_clim(0, max(top_attention_weights))

    ax.set_xticks(range(len(top_tokens)))
    ax.set_xticklabels(top_tokens, rotation=90)
    ax.set_yticks([])
    ax.set_title("Top Attention Weights Visualization")
    fig.colorbar(im, ax=ax, orientation='vertical')
    fig.tight_layout()
    plt.show()

@app.get("/model-overview")
def model_overview(request: Request):
    return templates.TemplateResponse("model_overview.html", {"request": request})

@app.get("/data-overview")
def data_overview(request: Request):
    return templates.TemplateResponse("data_overview.html", {"request": request})

@app.get("/scibert-overview")
def scibert_overview(request: Request):
    return templates.TemplateResponse("scibert_overview.html", {"request": request})

@app.get("/sources-credentials")
def sources_credentials(request: Request):
    return templates.TemplateResponse("sources_credentials.html", {"request": request})

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
    scibert_predictions, processed_text_scibert, attention_weights_list, chunks, tokenizer = predict_scibert(text)

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

    attention_weights = None
    tokens = None
    if attention_weights_list and chunks:
        try:
            attention_weights = attention_weights_list[0][-1].squeeze().mean(dim=0).cpu().numpy().tolist()
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(chunks[0], add_special_tokens=True))
            visualize_attention(chunks[0], attention_weights_list[0], tokenizer)
        except (IndexError, TypeError) as e:
            logger.error("Error occurred while processing attention weights: %s", str(e))

    return {
        "baseline_prediction": baseline_classifier,
        "scibert_predictions": [str(pred) for pred in scibert_predictions],
        "scibert_probability": scibert_probability,
        "text": text,
        "frequent_words": frequent_words,
        "frequent_bigrams": frequent_bigrams,
        "num_words": num_words,
        "num_unique_words": num_unique_words,
        "num_sentences": num_sentences,
        "attention_weights": attention_weights,
        "tokens": tokens
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
