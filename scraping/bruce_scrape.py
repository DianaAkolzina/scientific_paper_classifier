import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter

def get_article_links(topic_url):
    response = requests.get(topic_url)

    soup = BeautifulSoup(response.text, 'html.parser')
    links = [a['href'] for a in soup.select('h3.entry-title a.entry-title-link')]

    return links

def scrape_article_data(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.find('h1', class_='entry-title').text.strip()

        author = 'Bruce Lipton'
        published_date = soup.find('meta', property='article:published_time')['content'][:10]

        main_body_element = soup.find('div', class_='post-body')
        if main_body_element:
            main_body = main_body_element.get_text(separator=' ').strip()

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
            'Title': title,
            'Author': author,
            'Published Date': published_date,
            'Main Body': main_body,
            'Word Count': word_count,
            'Most Frequent Word': most_common_word,
            'Most Frequent Word Count': most_common_count,
            'Link': url,
            'Label' : 1,
            'Primary Category': ' ',
            'All Categories': ' ',
        }
        return article_data

    except requests.exceptions.RequestException as e:
        print(f"Error occurred while making a request to {url}: {e}")
        return None

    except Exception as e:
        print(f"Error occurred while scraping data from {url}: {e}")
        return None

def scrape_bruce_lipton(base_url, num_pages):
    articles = []

    for page in range(1, num_pages + 1):
        topic_url = f"{base_url}&wpv_paged={page}"
        article_links = get_article_links(topic_url)

        for link in article_links:
            article_data = scrape_article_data(link)
            if article_data:
                articles.append(article_data)

    df = pd.DataFrame(articles)
    return df

if __name__ == '__main__':
    base_url = "https://www.brucelipton.com/resources/?wpv_view_count=6544&category%5B%5D=article"
    num_pages = 500
    df_bruce_lipton = scrape_bruce_lipton(base_url, num_pages)

    df_bruce_lipton.to_csv('data/raw/bruce_lipton.csv', index=False)
    print("Scraping completed and data saved.")
