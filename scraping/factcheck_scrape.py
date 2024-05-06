import requests
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import feedparser
import requests
from bs4 import BeautifulSoup
import re
import nltk
import feedparser
import json
import urllib.request
from io import BytesIO
import pdfplumber
from dotenv import load_dotenv
import os


nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def get_article_links(topic_url):
    response = requests.get(topic_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('article')
    links = [art.find('a')['href'] for art in articles if art.find('a')]
    return links

def get_topics(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    topic_links = soup.find_all('a', href=True, class_='tag-cloud-link')
    topics = {link.get_text(strip=True): link['href'] for link in topic_links}
    return topics

def clean_and_count(text):
    text = re.sub(r'[^\w\s]', '', text)
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return words

def scrape_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('h1').get_text(strip=True) if soup.find('h1') else 'No title found'
    author = soup.find('a', rel='author').get_text(strip=True) if soup.find('a', rel='author') else 'Unknown author'
    content = soup.find('div', {'class': 'entry-content'})
    main_body = content.get_text(strip=True) if content else 'No content found'
    word_list = clean_and_count(main_body)
    word_count = len(word_list)
    picture_count = len(soup.find_all('img'))
    most_common_word, most_common_count = Counter(word_list).most_common(1)[0] if word_list else ('None', 0)
    abstract = ' '.join(word_list[:150])

    return {
        'Title': title,
        'Author': author,
        'Label': 0,
        'Abstract': abstract,
        'Main Body': main_body,
        'Word Count': word_count,
        'Picture Count': picture_count,
        'Subject': url.split('/')[-2],
        'Most Recurrent Word': most_common_word
    }

def scrape_factcheck(base_url):
    topics = get_topics(base_url)
    all_articles = []
    for topic_name, topic_url in topics.items():
        print(f"Scraping articles for topic: {topic_name}")
        article_links = get_article_links(topic_url)
        for link in article_links:
            article_data = scrape_article(link)
            article_data['Topic'] = topic_name
            all_articles.append(article_data)
    return pd.DataFrame(all_articles)


if __name__ == '__main__':
    factcheck_base_url = 'https://www.factcheck.org/archives/'
    df_factcheck = scrape_factcheck(factcheck_base_url)
    df_factcheck.to_csv('data/raw/factcheck.csv')
