import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os

def clean_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text

def fetch_article_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_links = []

    for link in soup.find_all('a', class_='navPage-childList-action navPages-action'):
        article_url = link.get('href')
        if article_url:
            article_links.append(article_url)

    return article_links

def parse_article(url):
    response = requests.get(url)
    print(response)
    soup = BeautifulSoup(response.text, 'html.parser')

    title_element = soup.find('title')
    print(title_element)
    title = title_element.text.strip() if title_element else ''

    title = title.replace('- Holistic Shop', '').strip()

    author_element = soup.find('meta', attrs={'name': 'author'})
    author = author_element['content'] if author_element else ''

    category_elements = soup.find_all('meta', attrs={'property': 'article:tag'})
    categories = [element['content'] for element in category_elements]

    word_count = 0
    main_body = ''
    content_div = soup.find('div', class_='category-description')
    if content_div:
        main_body = content_div.get_text(separator=' ').strip()
        main_body = clean_text(main_body)
        print(main_body)
        word_count = len(main_body.split())

    published_date = ''
    date_pattern = r'©\s*holisticshop\s*(\d{4})\.\s*All\s*rights\s*reserved'
    date_match = re.search(date_pattern, main_body)
    if date_match:
        published_date = date_match.group(1)

    row_data = {
        'Title': clean_text(title),
        'Published Date': published_date,
        'Author': clean_text(author),
        'Main Category': clean_text(', '.join(categories)),
        'All Categories': clean_text(', '.join(categories)),
        'Main Body': main_body,
        'Word Count': word_count,
        'Link': url,
        'Label': 1
    }

    return row_data

def save_row_to_csv(row_data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df = pd.DataFrame([row_data])

    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

base_url = 'https://www.holisticshop.co.uk/articles/?page='
file_path = '..data/raw/cryst.csv'

for page in range(1, 6):
    url = base_url + str(page)
    article_links = fetch_article_links(url)

    for link in article_links:
        row_data = parse_article(link)
        print(row_data)
        save_row_to_csv(row_data, file_path)

print(f"Data saved to {file_path}")
