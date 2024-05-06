import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from collections import Counter
import nltk
import os

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(nltk.corpus.stopwords.words('english'))

def scrape_davidwolfe(base_url, file_path):
    url = base_url + "/category/health-longevity/"

    with requests.Session() as session:
        session.max_redirects = 10000

        for page in range(1, 133):
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

                        row_data = {
                            "Title": title,
                            "Author": author,
                            "Label": 1,
                            "Abstract": abstract,
                            "Main Body": main_body,
                            "Word Count": word_count,
                            "Picture Count": picture_count,
                            "Subject": subject,
                            "Most Recurrent Word": most_recurrent_word,
                            'Link': page_url
                        }
                        save_row_to_csv(row_data, file_path)

                    except requests.RequestException as e:
                        print(f"Failed to fetch article: {e}")
            except requests.RequestException as e:
                print(f"Failed to fetch page: {e}")

def save_row_to_csv(row_data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df = pd.DataFrame([row_data])

    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

if __name__ == '__main__':
    base_url = "https://www.davidwolfe.com"
    file_path = 'data/raw/david.csv'
    scrape_davidwolfe(base_url, file_path)
    print(f"Data saved to {file_path}")
