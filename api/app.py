import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os
from interface.main_api import load_model_from_gcp, predict_baseline
from models.preprocess import preprocessing_pipeline_sample
from models.baseline_model import vectorize_data
import joblib
from google.cloud import storage

app = FastAPI()
# Allowing all middleware is optional, but good practice for dev purposes
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

BUCKET_NAME = os.getenv("BUCKET_NAME")
app.state.model = load_model_from_gcp(BUCKET_NAME, 'models/baseline_model.joblib')

# http://127.0.0.1:8001/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
# http://127.0.0.1:8002/predict?new_text=Make a prediction for a given text
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

    return {'Your text is' : scientific_classifier}



@app.get("/")
def root():
    return {
    'greeting': 'Hello'
}

from pydantic import BaseModel
class UserPost(BaseModel):
    new_text: str

@app.post("/post")
async def user_post(article_text: UserPost):
    model = app.state.model
    assert model is not None
    # print('preprocessing data...')
    # processed_new_text = preprocessing_pipeline_sample(article_text.new_text)

    # print("vectorising data...")
    # new_text_tfidf = vectorize_data([processed_new_text])

    # print("predicting...")
    # predicted_label = model.predict(new_text_tfidf)

    print("predicting..")
    predicted_label = predict_baseline(article_text.new_text)

    if predicted_label==0:
        scientific_classifier = "scientific"
    else:
        scientific_classifier = "pseudoscientific"
    print("end")
    return {
    'greeting': scientific_classifier
}
    # return {'Your text is' : scientific_classifier,}
