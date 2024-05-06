import requests
from bs4 import BeautifulSoup
import os
import csv
from dotenv import load_dotenv
import json
from collections import Counter

load_dotenv()

def get_article_links(topic_url):
    response = requests.get(topic_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('https://')]
    print(links)
    return links

def scrape_article_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        # Try to find the script containing JSON data
        script = soup.find('script', type='application/ld+json')
        if script and script.string:  # Check if the script tag and its content exist
            data = json.loads(script.string)
        else:
            data = {}  # Use an empty dictionary if no JSON data is found

        main_body_element = soup.find('div', class_='editorial-article__feed')
        if main_body_element:
            main_body = main_body_element.text.strip()
            words = main_body.split()
            word_count = len(words)
            most_common_word, most_common_count = Counter(words).most_common(1)[0]
        else:
            main_body = ""
            words = []
            word_count = 0
            most_common_word = "N/A"
            most_common_count = 0

        article_data = {
            'Published': data.get('datePublished', ''),
            'Title': data.get('headline', '').strip(),
            'Authors': data.get('author', {}).get('name', '') if data.get('author') else '',
            'Comments': 'No comments',
            'Primary Category': 'Astrology',
            'All Categories': 'Astrology',
            'Abstract': data.get('description', ''),
            'Main Body': main_body,
            'Word Count': word_count,
            'Most Frequent Word': most_common_word,
            'Most Frequent Word Count': most_common_count,
            'Number of Images': len(soup.find_all('img')),
            'Link': url,
            'Label': 1
        }
        return article_data
    except Exception as e:
        print(f"Error scraping data from {url}: {e}")
        return None


def scrape_astrology(base_url):
    article_links = get_article_links(base_url)
    with open('data/raw/astrology.csv', mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['Published', 'Title', 'Authors', 'Comments', 'Primary Category',
                      'All Categories', 'Abstract', 'Word Count', 'Most Frequent Word',
                      'Most Frequent Word Count', 'Main Body',
                      'Number of Images', 'Link', 'Label']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for link in article_links:
            article_data = scrape_article_data(link)
            if article_data:
                writer.writerow(article_data)

if __name__ == '__main__':
    astrology_start_url = os.getenv('ASTROLOGY_START_URL', 'https://www.astrology.com/articles/astrology-news')
    scrape_astrology(astrology_start_url)
    print("Scraping completed and data saved.")
