import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from interface.main import load_model_from_gcp
from models.preprocess import preprocessing_pipeline_sample
from models.baseline_model import vectorize_data
import joblib
from google.cloud import storage

app = FastAPI()
# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model = load_model_from_gcp(BUCKET_NAME, 'models/baseline_model.joblib')

# http://127.0.0.1:8001/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(
        new_text: str
    ):
    """
    Make a prediction for a given text
    """

    model = app.state.model
    assert model is not None

    processed_new_text = preprocessing_pipeline_sample(new_text)
    new_text_tfidf = vectorize_data([processed_new_text])
    predicted_label = model.predict(new_text_tfidf)

    if predicted_label[0]==0:
        scientific_classifier = "scientific"
    else:
        scientific_classifier = "pseudoscientific"

    return dict({'Your text is' : scientific_classifier})



@app.get("/")
def root():
    return {
    'greeting': 'Hello'
}
