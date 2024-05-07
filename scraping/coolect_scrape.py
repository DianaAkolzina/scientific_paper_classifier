
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from collections import Counter
import nltk
import requests
import re
from collections import Counter
from nltk.corpus import stopwords
from dotenv import load_dotenv



nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def scrape_collective_evolution(base_url):
    urls = [
        base_url + "/blog?tag=ce+insight",
        base_url + "/blog?page=2&tag=ce%20insight"
    ]

    data = []

    for url in urls:
        response = requests.get(url)
        print(response)
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



if __name__ == '__main__':


    collective_evolution_base_url = "https://www.collective-evolution.com"

    df_collective_evolution = scrape_collective_evolution(collective_evolution_base_url)
    df_collective_evolution.to_csv('data/raw/collective_evolution_2.csv', index=False)
