import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
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

def clean_and_count(text):
    text = re.sub(r'[^\w\s]', '', text)
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return words

def get_topics(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    topic_links = soup.find_all('a', href=True, class_='tag-cloud-link')
    topics = {link.get_text(strip=True): link['href'] for link in topic_links}
    return topics

def get_article_links(topic_url):
    response = requests.get(topic_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('article')
    links = [art.find('a')['href'] for art in articles if art.find('a')]
    return links

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

def scrape_davidwolfe(base_url):
    url = base_url + "/category/health-longevity/"
    data = []

    with requests.Session() as session:
        session.max_redirects = 10000

        for page in range(1, 10):
            page_url = f"{url}page/{page}/" if page > 1 else url
            try:
                response = session.get(page_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                article_links = soup.find_all("h2", class_="cb-post-title")

                for link in article_links:
                    article_url = link.find("a")["href"]
                    print(article_url)
                    try:
                        article_response = session.get(article_url)
                        article_response.raise_for_status()
                        print(article_response)

                        article_soup = BeautifulSoup(article_response.content, "html.parser")

                        title = article_soup.find("h1", class_="entry-title").text.strip()
                        author_element = article_soup.find("span", class_="fn")
                        author = author_element.find("a").text.strip() if author_element else ""

                        main_body_element = article_soup.find("section", class_="cb-entry-content clearfix")
                        if main_body_element:
                            main_body = main_body_element.text.strip()
                            sentences = nltk.sent_tokenize(main_body)
                            abstract = ' '.join(sentences[:3])
                            word_count = len(main_body.split())
                            words = re.findall(r'\b\w+\b', main_body.lower())
                            filtered_words = [word for word in words if word not in stop_words]
                            most_recurrent_word = Counter(filtered_words).most_common(1)[0][0] if filtered_words else ""
                        else:
                            main_body = ""
                            abstract = ""
                            word_count = 0
                            most_recurrent_word = ""

                        picture_count = len(article_soup.find_all("img"))

                        subject_elements = article_soup.find_all("span", class_="cb-category cb-element")
                        subjects = [element.find("a").text.strip() for element in subject_elements]
                        subject = ', '.join(subjects)

                        data.append([title, author, 1, abstract, main_body, word_count, picture_count, subject, most_recurrent_word])

                    except requests.RequestException as e:
                        print(f"Failed to fetch article: {e}")
            except requests.RequestException as e:
                print(f"Failed to fetch page: {e}")

    df = pd.DataFrame(data, columns=["Title", "Author", "Label", "Abstract", "Main Body", "Word Count", "Picture Count", "Subject", "Most Recurrent Word"])
    return df

def scrape_collective_evolution(base_url):
    urls = [
        base_url + "/blog?tag=ce+insight",
        base_url + "/blog?page=2&tag=ce%20insight"
    ]

    data = []

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        article_links = soup.find_all("a", class_="blog-listing__title")

        for link in article_links:
            article_url = base_url + link["href"]
            print(article_url)

            article_response = requests.get(article_url)
            print(article_response)

            article_soup = BeautifulSoup(article_response.content, "html.parser")
            print(article_soup)

            title = article_soup.find("h1", class_="blog-post-body__title").text.strip()
            author = article_soup.find("p", class_="sidebar-instructor__name").text.strip()

            main_body = article_soup.find("div", class_="blog-post-body__content").text.strip()
            sentences = nltk.sent_tokenize(main_body)
            abstract = ' '.join(sentences[:3])

            word_count = len(main_body.split())
            picture_count = len(article_soup.find_all("img"))

            subject_element = article_soup.find("span", class_="subject")
            subject = subject_element.text.strip() if subject_element else ""

            words = re.findall(r'\b\w+\b', main_body.lower())
            filtered_words = [word for word in words if word not in stop_words]
            most_recurrent_word = Counter(filtered_words).most_common(1)[0][0]

            data.append([title, author, 1, abstract, main_body, word_count, picture_count, subject, most_recurrent_word])

    df = pd.DataFrame(data, columns=["Title", "Author", "Label", "Abstract", "Main Body", "Word Count", "Picture Count", "Subject", "Most Recurrent Word"])
    return df

def scrape_astrology(start_url):
    def scrape_article_data(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        try:
            script = soup.find('script', type='application/ld+json')
            data = json.loads(script.string)
            main_body = soup.find('div', class_='editorial-article__feed').text.strip()
            words = main_body.split()
            word_count = len(words)
            most_common_word, most_common_count = Counter(words).most_common(1)[0]

            article_data = {
                'Published': data.get('datePublished', ''),
                'Title': data.get('headline', '').strip(),
                'Authors': data.get('author', {}).get('name', ''),
                'Comments': 'No comments',
                'Primary Category': 'Astronomy',
                'All Categories': 'Astronomy',
                'Abstract': data.get('description', ''),
                'Word Count': word_count,
                'Most Frequent Word': most_common_word,
                'Number of Images': len(soup.find_all('img')),
                'Link': url,
                'Label': 1
            }
        except Exception as e:
            print(f"Error scraping data from {url}: {e}")
            return None

        return article_data

    data = []
    article_links = get_article_links(base_url)
    for link in article_links:
        article_data = scrape_article_data(link)
        if article_data:
            data.append(article_data)

    df = pd.DataFrame(data)
    return df

def fetch_articles(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = []
    article_divs = soup.find_all('div', class_='article-item')

    for article in article_divs:
        try:
            link = article.find('a')['href']
            title = article.find('header').text.strip()
            author = article.find('div', class_='author').text.strip()
            published_date = article.find('span', class_='article-date').text.strip()
            image_count = len(article.find_all('img'))

            article_response = requests.get(link)
            article_soup = BeautifulSoup(article_response.text, 'html.parser')
            main_content = article_soup.find('div', class_='entry-content')
            text = main_content.get_text() if main_content else "No content"
            word_count = len(text.split())
            most_common_word = Counter(text.split()).most_common(1)[0][0] if text else None

            articles.append({
                'Published': published_date,
                'Title': title,
                'Authors': author,
                'Comments': "No comments",
                'Primary Category': "Archeology",
                'All Categories': "Archeology",
                'Abstract': text,
                'Word Count': word_count,
                'Most Frequent Word': most_common_word,
                'Number of Images': image_count,
                'Link': link,
                'Label': 1
            })
        except Exception as e:
            print(f"Error processing article: {e}")

    return pd.DataFrame(articles)

import pandas as pd

import pandas as pd

pseudoscientific_queries = [
    'quantum',
    'astrology',
]

list_addons = [ 'crystals',
    'paranormal',
    'telepathy',
    'homeopathy',
    'chakras',
    'detox',
    'aura',
    'vortex',
    'morphic',
    'water-memory',
    'anti-vaccine',
    'flat-earth',
    'chemtrails',
    'bioharmonic',
    'numerology',
    'alchemy',
    'ley-lines',
    'fourth-dimension',
    'photonics',
    'pyramid',
    'telekinesis',
    'cryptozoology',
    'ufology',
    'noetics',
    'sound-healing',
    'energy-fields',
    'spirit-science',
    'quantum-consciousness']

if __name__ == '__main__':
    # Existing scraping code
    # factcheck_base_url = 'https://www.factcheck.org/archives/'
    # df_factcheck = scrape_factcheck(factcheck_base_url)

    # davidwolfe_base_url = "https://www.davidwolfe.com"
    # df_davidwolfe = scrape_davidwolfe(davidwolfe_base_url)

    # Fetch arXiv papers for each pseudoscientific query
    for query in pseudoscientific_queries:
        df_arxiv = fetch_papers(query, max_results, base_url)
        print(df_arxiv)
        df_arxiv_all = pd.concat([df_arxiv_all, df_arxiv], ignore_index=True)

    # Optional: Parse PDFs for detailed analysis (can be skipped for efficiency)
    for index, row in df_arxiv_all.iterrows():
        if row['PDF Link']:
            abstract, main_body, most_frequent_word, main_body_word_count = fetch_and_parse_pdf(row['PDF Link'])
            df_arxiv_all.at[index, 'Abstract'] = abstract
            df_arxiv_all.at[index, 'Main Body'] = main_body
            df_arxiv_all.at[index, 'Most Frequent Word'] = most_frequent_word
            df_arxiv_all.at[index, 'Word Count'] = main_body_word_count

    # # Concatenate dataframes
    # df_all = pd.concat([df_factcheck, df_davidwolfe, df_collective_evolution, df_astrology, df_arxiv_all], ignore_index=True)
    # df_all = df_all.dropna(how='all').drop_duplicates()

    # print(df_all.head())
    # print(df_all.shape)

    print(df_arxiv_all)
    print(df_arxiv_all.describe())
