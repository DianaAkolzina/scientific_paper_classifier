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
    links = [a['href'] for a in soup.find_all('a', class_='chakra-link css-10qsrqw')]
    base_url = 'https://isha.sadhguru.org'
    full_links = [base_url + link if not link.startswith('http') else link for link in links]
    return full_links

def scrape_article_data(url):
    response = requests.get(url)
    print(response)
    soup = BeautifulSoup(response.text, 'html.parser')

    try:
        title_element = soup.find('h1', class_='css-n0u6yx')
        title = title_element.text.strip() if title_element else ''
        print(title)

        published_date_element = soup.find('div', class_='css-1uf1z8m')
        published_date = published_date_element.text.strip() if published_date_element else ''

        author = 'Sadhguru'

        main_body_element = soup.find('div', class_='css-exoff3')
        if main_body_element:
            paragraphs = main_body_element.find_all('p')
            main_body = ' '.join([p.text.strip() for p in paragraphs])
            words = main_body.split()
            word_count = len(words)
            most_common_word, most_common_count = Counter(words).most_common(1)[0]
        else:
            main_body = ""
            words = []
            word_count = 0
            most_common_word = "N/A"
            most_common_count = 0

        related_tags = soup.find_all('a', class_='chakra-tag css-1xx9vtx')
        all_categories = [tag.text.strip() for tag in related_tags]
        primary_category = all_categories[0] if all_categories else ''

        article_data = {
            'Published': published_date,
            'Title': title,
            'Authors': author,
            'Comments': 'No comments',
            'Primary Category': primary_category,
            'All Categories': ', '.join(all_categories),
            'Abstract': main_body[:200] + '...' if len(main_body) > 200 else main_body,
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

def scrape_isha_sadhguru(base_url, num_pages):
    with open('data/raw/isha_sadhguru.csv', mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['Published', 'Title', 'Authors', 'Comments', 'Primary Category', 'All Categories', 'Abstract',
                      'Word Count', 'Most Frequent Word', 'Most Frequent Word Count', 'Main Body', 'Number of Images',
                      'Link', 'Label']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for page in range(1, num_pages + 1):
            topic_url = f"{base_url}?contentType=article&page={page}"
            article_links = get_article_links(topic_url)

            for link in article_links:
                article_data = scrape_article_data(link)
                if article_data:
                    writer.writerow(article_data)

if __name__ == '__main__':
    isha_sadhguru_base_url = os.getenv('ISHA_SADHGURU_BASE_URL', 'https://isha.sadhguru.org/en/wisdom/type/article')
    num_pages = 40
    scrape_isha_sadhguru(isha_sadhguru_base_url, num_pages)
    print("Scraping completed and data saved.")
