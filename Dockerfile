FROM python:3.8.12-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY scientific_paper_classifier scientific_paper_classifier
COPY setup.py setup.py
RUN pip install .

CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT
