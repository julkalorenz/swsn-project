import requests
from lxml import html
from urllib.parse import urlparse, parse_qs, unquote
from html import unescape
import trafilatura
import pandas as pd

def get_articles_ddg(query, n_articles):
    def unwrap_ddg(href):
        href = unescape(href)                 # &amp; -> &
        if href.startswith("//"):
            href = "https:" + href            # add scheme
        qs = parse_qs(urlparse(href).query)
        return unquote(qs["uddg"][0])

    def page_text(url):
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        except Exception:
            return ""
        text = trafilatura.extract(r.text)

        return text

    url = "https://lite.duckduckgo.com/lite/"
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, params=params, headers=headers, timeout=10)
    tree = html.fromstring(r.text)
    links = tree.cssselect("a.result-link")[:n_articles]

    top3 = [page_text(unwrap_ddg(a.get("href"))) for a in links]
    return top3


if __name__ == "__main__":
    data = pd.read_csv('./data/processed/selected_posts_with_responses.csv')
    all_articles = []
    for i, row in data.iterrows():
        current_topic_articles = []
        for i in range(1, 4):
            query = row[f'response_{i}']
            print(f"Quering... {query}")
            current_topic_articles += get_articles_ddg(query, 1)
        all_articles.append(current_topic_articles)

    all_articles_augmented = []
    for i, row in data.iterrows():
        current_topic_articles = []
        for i in range(1, 4):
            query = row[f'response_{i}_augmented']
            print(f"Quering... {query}")
            current_topic_articles += get_articles_ddg(query, 1)
        all_articles_augmented.append(current_topic_articles)
    print(all_articles)
    print(all_articles_augmented)
