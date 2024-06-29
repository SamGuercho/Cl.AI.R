import requests
from bs4 import BeautifulSoup

def get_article_text(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch the page: {response.status_code}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # The Guardian uses <div> with class 'article-body-commercial-selector' for the main content
    article_body = soup.find('div', {'class': 'article-body-commercial-selector'})
    if not article_body:
        print("Could not find the article body.")
        return None
    
    # Extract text content from paragraphs
    paragraphs = article_body.find_all('p')
    article_text = "\n\n".join([p.get_text() for p in paragraphs])
    
    return article_text

if __name__ == '__main__':
    url = 'https://www.theguardian.com/world/article/2024/jun/24/israeli-far-right-minister-bezalel-smotrich-annex-west-bank'
    article_text = get_article_text(url)
    if article_text:
        print(article_text)
