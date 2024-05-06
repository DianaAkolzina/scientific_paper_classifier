import pandas as pd
import feedparser
import requests
from time import sleep
import logging
from urllib.request import urlopen
from io import BytesIO
import pdfplumber
import os
import re

def fetch_papers(search_query, max_results, base_url, file_path):
    chunk_size = 1
    start_index = 0
    while start_index < max_results:
        query = f'search_query={search_query}&start={start_index}&max_results={chunk_size}'
        with urlopen(base_url + query) as url:
            response = url.read()
        feed = feedparser.parse(response)

        for entry in feed.entries:
            arxiv_id = entry.id.split('/abs/')[-1]
            published = entry.published
            title = entry.title.replace(',', '').replace('.', '')
            authors = '|'.join(author.name.replace(',', '').replace('.', '') for author in entry.authors)
            journal_ref = entry.get('arxiv_journal_ref', 'No journal ref found').replace(',', '').replace('.', '')
            comment = entry.get('arxiv_comment', 'No comment found').replace(',', '').replace('.', '')
            primary_category = entry.tags[0]['term']
            all_categories = '|'.join(t['term'] for t in entry.tags)
            abstract = entry.summary.replace(',', '').replace('.', '')
            pdf_link = ''
            for link in entry.links:
                if hasattr(link, 'title') and link.title == 'pdf':
                    pdf_link = link.href
                    print(pdf_link)
                    break

            main_body, word_count = fetch_and_parse_pdf(pdf_link)

            row_data = {
                'arxiv-id': arxiv_id,
                'Published': published,
                'Title': title,
                'Authors': authors,
                'Journal reference': journal_ref,
                'Comments': comment,
                'Primary Category': primary_category,
                'All Categories': all_categories,
                'Abstract': abstract,
                'Main Body': main_body,
                'Word Count': word_count,
                'PDF Link': pdf_link,
                'Label': 0
            }

            save_row_to_csv(row_data, file_path)

        start_index += chunk_size
        if start_index >= int(feed.feed.opensearch_totalresults):
            break

def fetch_and_parse_pdf(pdf_url):
    if not pdf_url:
        return "No PDF link", 0

    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())

        cleaned_text = re.sub(r'[^\w\s]', '', full_text)

        word_count = len(cleaned_text.split())
        return cleaned_text, word_count

    except Exception as e:
        logging.error(f"Failed to process PDF {pdf_url}: {str(e)}")
        return "Failed to extract", 0

def save_row_to_csv(row_data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df = pd.DataFrame([row_data])

    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define base settings
base_url = 'http://export.arxiv.org/api/query?'
queries = health_words = [

    "Supplements",
    "Meditation",
    "Yoga",
    "Therapy",
    "Rehabilitation",
    "Cardiovascular",
    "Strength",
    "Immunity",
    "Detox",
    "Organic",
    "Vegan",
    "Gluten-free",
    "Probiotics",
    "Mental Health",
    "Stress",
    "Sleep",
    "Obesity",
    "Diabetes",
    "Hypertension",
    "Allergies",
    "Vaccination",
    "Epidemic",
    "Endurance",
    "Holistic",
    "Antibiotics",
    "Inflammation",
    "Geriatrics",
    "Pediatrics",
    "Neurology",
    "Oncology",
    "Gastroenterology",
    "Dermatology"
]

max_results = 90

os.chdir(os.path.dirname(__file__) + '/..')

file_path = 'data/raw/arxiv.csv'

for query in queries:
    logging.info(f"Fetching papers for query: {query}")
    fetch_papers(query, max_results, base_url, file_path)

logging.info(f"Data saved to {file_path}")
