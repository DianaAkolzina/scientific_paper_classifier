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


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def preprocessing_pipeline(df, text_column, author_column, label_column):
    """Applies text cleaning, English filtering, and removes specific common words from the text column of a DataFrame based on both label and author-specific frequency conditions, excluding authors with less than 20% representation."""

    df = df.dropna(subset=[text_column])

    df = filter_english_text(df, text_column)

    df['Processed Text'] = df[text_column].apply(preprocess_text)
    df['Processed Text'] = df['Processed Text'].apply(lemmatize_text)

    vectorizer = CountVectorizer()

    author_counts = df[author_column].value_counts(normalize=True)

    def filter_words(group_df, comparison_df, vectorizer):
        counts = vectorizer.fit_transform(group_df['Processed Text']).toarray()
        feature_names = vectorizer.get_feature_names_out()
        word_freq = (counts > 0).sum(axis=0) / len(group_df)
        common_words = feature_names[word_freq > 0.5]
        words_to_remove = []

        for word in common_words:
            if word in vectorizer.vocabulary_:
                other_counts = vectorizer.transform(comparison_df['Processed Text']).toarray()
                other_word_freq = (other_counts[:, vectorizer.vocabulary_[word]] > 0).sum() / len(comparison_df)
                if other_word_freq < 0.3:
                    words_to_remove.append(word)
        return words_to_remove

    for label in df[label_column].unique():
        label_df = df[df[label_column] == label]
        other_label_df = df[df[label_column] != label]
        label_words_to_remove = filter_words(label_df, other_label_df, vectorizer)
        label_df['Processed Text'] = label_df['Processed Text'].apply(lambda text: ' '.join([word for word in text.split() if word not in label_words_to_remove]))
        df.loc[label_df.index, 'Processed Text'] = label_df['Processed Text']

    for author, count in author_counts.items():
        if count >= 0.2:  
            author_df = df[df[author_column] == author]
            other_author_df = df[df[author_column] != author]
            author_words_to_remove = filter_words(author_df, other_author_df, vectorizer)
            author_df['Processed Text'] = author_df['Processed Text'].apply(lambda text: ' '.join([word for word in text.split() if word not in author_words_to_remove]))
            df.loc[author_df.index, 'Processed Text'] = author_df['Processed Text']

    return df


