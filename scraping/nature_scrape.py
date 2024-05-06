import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os

def clean_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text

def fetch_main_content(html_content):
    """ Extracts the main body text from the provided HTML content. """
    soup = BeautifulSoup(html_content, 'html.parser')
    content = ''
    start_collecting = False

    for element in soup.find_all():
        if element.name == 'hr' and element.get('class', []) == ['Marker']:
            start_collecting = True
        if start_collecting:
            if element.name == 'p':
                content += element.get_text(separator=' ') + ' '
    return content.strip()

def fetch_article_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_links = []

    for div in soup.find_all('div', class_='Post'):
        headline = div.find('div', class_='Headline')
        if headline:
            link = headline.find('a')
            if link:
                article_url = link['href']
                if not article_url.startswith('http'):
                    article_url = 'https://www.naturalnews.com/' + article_url
                article_links.append(article_url)

    return article_links

def parse_article(url):
    response = requests.get(url)
    print(response)
    soup = BeautifulSoup(response.text, 'html.parser')

    title_element = soup.find('meta', attrs={'property': 'og:title'})
    title = title_element['content'] if title_element else ''
    print(title)
    title = title.replace('– NaturalNews.com', '').strip()

    author_div = soup.find('div', {"id": "AuthorInfo"})
    author = author_div.find('a').get_text() if author_div and author_div.find('a') else 'Unknown Author'

    category_elements = soup.find_all('meta', attrs={'property': 'article:tag'})
    categories = [element['content'] for element in category_elements]

    published_date_element = soup.find('meta', attrs={'property': 'article:published_time'})
    published_date = published_date_element['content'] if published_date_element else ''

    main_body = fetch_main_content(response.text)
    word_count = len(main_body.split())

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

base_url = 'https://www.naturalnews.com/category/culture-society/page/'
file_path = 'data/raw/naturalnews.csv'

for page in range(1, 1000):
    url = base_url + str(page) + '/'
    article_links = fetch_article_links(url)

    for link in article_links:
        row_data = parse_article(link)
        save_row_to_csv(row_data, file_path)

print(f"Data saved to {file_path}")
