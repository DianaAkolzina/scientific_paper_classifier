import pandas as pd
import spacy
from langdetect import detect, LangDetectException
import string
import re
from collections import Counter
from unidecode import unidecode
import io
from google.cloud import storage
import os
from sklearn.feature_extraction.text import CountVectorizer
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.gcp.internal.clients import bigquery
from apache_beam.io.gcp.bigquery_tools import TableReference
import logging
from apache_beam.options.pipeline_options import SetupOptions, WorkerOptions
from apache_beam.io.gcp.bigquery import ReadFromBigQuery, WriteToBigQuery

logging.basicConfig(level=logging.INFO)

nlp = spacy.load("en_core_web_sm")

BUCKET_NAME = os.getenv("BUCKET_NAME")


class PreprocessTextFn(beam.DoFn):
    def process(self, element):
        df = pd.DataFrame([element])

        if df.empty:
            logging.warning("Empty DataFrame encountered. Skipping processing.")
            return

        text_column = 'Processed_Text'
        author_column = 'Author'
        label_column = 'Label'

        processed_df = preprocessing_pipeline(df, text_column, author_column, label_column)

        if not processed_df.empty:
            processed_text = processed_df.iloc[0]['Processed_Text']
            author = processed_df.iloc[0]['Author']
            label = int(processed_df.iloc[0]['Label'])

            if pd.isna(author) or author.strip() == '':
                logging.warning(f"Empty author value encountered for processed text: {processed_text}. Skipping yield.")
                return

            logging.info(f"Processed text: {processed_text}")
            yield {
                'Processed_Text': processed_text,
                'Author': author,
                'Label': label
            }
        else:
            logging.warning("Empty DataFrame after preprocessing. Skipping yield.")

def filter_and_update_categories(df):
    main_cat_counts = df['Main Category'].value_counts()
    freq_nouns = Counter()

    for index, row in df.iterrows():
        doc = nlp(row['Processed Text'])
        nouns = [token.text for token in doc if token.pos_ == 'NOUN']
        freq_nouns.update(nouns)

    df['Main Category'] = df['Main Category'].apply(
        lambda x: freq_nouns.most_common(1)[0][0] if pd.isna(x) or main_cat_counts[x] < 10 else x
    )

    all_cats = Counter([cat for sublist in df['All Categories'].astype(str).str.split(',').dropna() for cat in sublist])
    df['All Categories'] = df['All Categories'].astype(str).str.split(',').apply(
        lambda cats: [cat.strip() for cat in cats if cat.strip() in all_cats and all_cats[cat.strip()] >= 10] if isinstance(cats, list) else []
    )

    df['All Categories'] = df['All Categories'].apply(
        lambda x: [word for word, count in freq_nouns.most_common(3)] if not x else x
    )

    return df

def get_top_words(text, n=10):
    """Return the most common words in the text."""
    words = re.findall(r'\w+', text)
    most_common_words = Counter(words).most_common(n)
    return most_common_words

def standardize_dates(df, date_column):
    """Standardizes the dates in a DataFrame column to the format 'YYYY-MM-DD'."""
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce').dt.strftime('%Y-%m-%d')
    return df


def preprocess_text(text):
    """Processes the given text by cleaning and normalizing it."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()  # Convert text to lowercase and strip whitespaces
    text = unidecode(text)  # Normalize text
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces into one
    text = re.sub(r'\d', ' ', text)  # Remove digits
    return text

def lemmatize_text(text):
    """Lemmatizes the given text, removing stopwords, very short words, certain parts of speech, and named entities."""
    doc = nlp(text)
    ents = set([ent.text for ent in doc.ents])
    filtered_tokens = []
    for token in doc:
        if not token.is_stop and len(token.lemma_) > 2 and token.lemma_.isalpha() \
           and token.pos_ not in ['PRON', 'DET'] and token.text not in ents:
            filtered_tokens.append(token.lemma_)
    return ' '.join(filtered_tokens)

def extract_middle_text(text, word_count=400):
    """Extracts `word_count` words from the middle of the text."""
    words = text.split()
    total_words = len(words)
    if total_words <= word_count:
        return text
    start_index = (total_words - word_count) // 2
    return ' '.join(words[start_index:start_index + word_count])


def detect_english(text):
    """Detects if the given text is English."""
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def filter_english_text(df, text_column):
    """Filters the DataFrame to only include rows where the text is English."""
    df['is_english'] = df[text_column].apply(detect_english)
    filtered_df = df[df['is_english']].drop(columns=['is_english'])
    return filtered_df

def get_most_frequent_word(text):
    """Extracts the most frequent word from the text."""
    words = text.split()
    if words:
        return Counter(words).most_common(1)[0][0]
    return None

def get_most_frequent_bigram(text):
    """Extracts the most frequent two-word combination from the text."""
    words = text.split()
    if len(words) > 1:
        bigrams = zip(words[:-1], words[1:])
        return ' '.join(Counter(bigrams).most_common(1)[0][0])
    return None

def clean_text(text):
    """Further clean the text by removing numbers and non-alphanumeric characters, and converting to lowercase."""
    if pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    return text


def preprocessing_pipeline(df, text_column, author_column, label_column):
    """Applies text cleaning, English filtering, and removes specific common words from the text column of a DataFrame based on both label and author-specific frequency conditions, excluding authors with less than 20% representation."""
    print(df.columns)
    df = df.dropna(subset=[text_column])

    df = filter_english_text(df, text_column)

    df['Processed_Text'] = df[text_column].apply(preprocess_text)
    df['Processed_Text'] = df['Processed_Text'].apply(lemmatize_text)
    df['Processed_Text'] = df['Processed_Text'].apply(extract_middle_text)
    vectorizer = CountVectorizer()

    author_counts = df[author_column].value_counts(normalize=True)

    def filter_words(group_df, comparison_df, vectorizer):
        counts = vectorizer.fit_transform(group_df['Processed_Text']).toarray()
        feature_names = vectorizer.get_feature_names_out()
        word_freq = (counts > 0).sum(axis=0) / len(group_df)
        common_words = feature_names[word_freq > 0.5]
        words_to_remove = []

        for word in common_words:
            if word in vectorizer.vocabulary_:
                other_counts = vectorizer.transform(comparison_df['Processed_Text']).toarray()
                other_word_freq = (other_counts[:, vectorizer.vocabulary_[word]] > 0).sum() / len(comparison_df)
                if other_word_freq < 0.3:
                    words_to_remove.append(word)
        return words_to_remove

    for label in df[label_column].unique():
        label_df = df[df[label_column] == label]
        other_label_df = df[df[label_column] != label]
        label_words_to_remove = filter_words(label_df, other_label_df, vectorizer)
        label_df['Processed_Text'] = label_df['Processed_Text'].apply(lambda text: ' '.join([word for word in text.split() if word not in label_words_to_remove]))
        df.loc[label_df.index, 'Processed_Text'] = label_df['Processed_Text']

    for author, count in author_counts.items():
        if count >= 0.2:
            author_df = df[df[author_column] == author]
            other_author_df = df[df[author_column] != author]
            author_words_to_remove = filter_words(author_df, other_author_df, vectorizer)
            author_df['Processed_Text'] = author_df['Processed_Text'].apply(lambda text: ' '.join([word for word in text.split() if word not in author_words_to_remove]))
            df.loc[author_df.index, 'Processed_Text'] = author_df['Processed_Text']

    return df

def preprocessing_pipeline_sample(text):
    text = preprocess_text(text)
    text = lemmatize_text(text)
    text = extract_middle_text(text)
    return text

class WriteToGCSFn(beam.DoFn):
    def process(self, element):
        bucket_name = 'scientific_paper_classifier-bucket'
        blob_name = f"processed_text/{element['Author']}/{element['Label']}.txt"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.upload_from_string(element['Processed_Text'])

        logging.info(f"Uploaded processed text to GCS: {blob_name}")

class ChunkingFn(beam.DoFn):
    def process(self, element):
        yield element
logging.basicConfig(level=logging.INFO)

from apache_beam.io.gcp.internal.clients.bigquery import TableSchema, TableFieldSchema

def run(argv=None):
    PROJECT_ID = 'scientific-paper-classifier'
    GCS_TEMP_LOCATION = 'gs://scientific_paper_classifier-bucket/data_cloud'

    pipeline_options = PipelineOptions(
        project=PROJECT_ID,
        temp_location=GCS_TEMP_LOCATION,
        job_name='text-preprocessing',
        save_main_session=True,
        staging_location=GCS_TEMP_LOCATION
    )

    schema = TableSchema(fields=[
    TableFieldSchema(name='Processed_Text', type='STRING', mode='REQUIRED'),
    TableFieldSchema(name='Author', type='STRING', mode='NULLABLE'),
    TableFieldSchema(name='Label', type='INTEGER', mode='REQUIRED')
    ])

    with beam.Pipeline(options=pipeline_options) as p:
        data = (
        p
        | 'Read from BigQuery' >> ReadFromBigQuery(
            query=f'SELECT Processed_Text, Author, Label FROM `{PROJECT_ID}.data_train.all` LIMIT 13000',
            gcs_location=GCS_TEMP_LOCATION,
            use_standard_sql=True)
        | 'Apply Preprocessing' >> beam.ParDo(PreprocessTextFn())
        | 'Write Results to BigQuery' >> WriteToBigQuery(
            table='processed_text_13000',
            dataset='data_train',
            project=PROJECT_ID,
            schema=schema,
            method="STREAMING_INSERTS",
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
    )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
