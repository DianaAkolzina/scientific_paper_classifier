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

# Load the English language model for Spacy
nlp = spacy.load("en_core_web_sm")

def filter_and_update_categories(df):

    main_cat_counts = df['Main Category'].value_counts()
    freq_nouns = Counter()

    for index, row in df.iterrows():

        doc = nlp(row['Processed Text'])
        nouns = [token.text for token in doc if token.pos_ == 'NOUN']
        freq_nouns.update(nouns)

    df['Main Category'] = df['Main Category'].apply(
        lambda x: freq_nouns.most_common(1)[0][0] if main_cat_counts[x] < 10 else x
    )

    all_cats = Counter([cat for sublist in df['All Categories'].str.split(',').dropna() for cat in sublist])
    df['All Categories'] = df['All Categories'].str.split(',').apply(
        lambda cats: [cat.strip() for cat in cats if all_cats[cat.strip()] >= 10] if cats else []
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
    df[date_column] = df[date_column].apply(lambda date: parser.parse(date).strftime('%Y-%m-%d'))
    return df


def preprocess_text(text):
    """Processes the given text by cleaning and normalizing it."""
    text = text.lower().strip()  # Convert text to lowercase and strip whitespaces
    text = unidecode(text)  # Normalize text
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces into one
    text = re.sub(r'\d', ' ', text)  # Remove digits
    return text

def lemmatize_text(text):
    """Lemmatizes the given text, removing stopwords, very short words, and certain parts of speech."""
    doc = nlp(text)
    filtered_tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and len(token.lemma_) > 2 and token.lemma_.isalpha()
        and token.pos_ not in ['PRON', 'DET']
    ]
    return ' '.join(filtered_tokens)


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

def preprocessing_pipeline(df, text_column):
    """Applies text cleaning, English filtering, and updates most frequent words in a DataFrame."""
    df = df.dropna(subset=[text_column])
    df = filter_english_text(df, text_column)
    # df = standardize_dates(df, date_column)

    if not df.empty:
        df['Processed Text'] = df[text_column].apply(preprocess_text)
        df = filter_and_update_categories(df)
        df = lemmatize_text(df)
        df['Most Frequent Word'] = df['Processed Text'].apply(get_most_frequent_word)
        df['Word Count'] = df['Processed Text'].apply(lambda n: len(n.split()))
       # df['Most Frequent Word Combination'] = df['Processed Text'].apply(get_most_frequent_bigram)

    return df[['Title', 'Author', 'Published Date', 'Word Count', 'Most Frequent Word', 'Label', 'Processed Text']]
