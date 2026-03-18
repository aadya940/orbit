from duckduckgo_search import DDGS
import requests
from readability import Document
from bs4 import BeautifulSoup
from typing import Dict, Any


def search(query, k=3):
    with DDGS() as ddgs:
        return [r["href"] for r in ddgs.text(query, max_results=k)]


def extract_content(url):
    try:
        html = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
    except:
        return ""

    try:
        doc = Document(html)
        return doc.summary()
    except:
        pass

    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)


def pipeline(query):
    urls = search(query)
    contents = [extract_content(u)[:3000] for u in urls]
    return contents


def duckduckgo_search(query: str) -> Dict[str, Any]:
    return pipeline(query)
