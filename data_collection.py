import os, kagglehub, requests, pandas as pd
from bs4 import BeautifulSoup
from newspaper import Article
import time

import logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s %(levelname)s: %(message)s',
    handlers = [logging.StreamHandler()]
)

def install_data(source_path, target_path):
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    path = kagglehub.dataset_download(source_path)

    logging.info(f"Path to dataset files: {path}")
    logging.info('install completed!\n')




def news_api(news_articles_path):

    API_KEY = '75972c1193984f8d9dd7250d55ce4bcb'
    url = 'https://newsapi.org/v2/top-headlines'

    params = {
        'sources': 'bbc-news,cnn,bloomberg,the-washington-post,associated-press, breitbart-news,fox-news,rt,infowars.com,thegatewaypundit.com,worldnewsdailyreport.com',  # Source: BBC News
        'apiKey': API_KEY,
        'pageSize': 100
    }

    articles_list = []

    for page in range(1, 100):
        params['page'] = page
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])

            for article in articles:
               articles_list.append({
                    'title': article.get('title'),
                    'description': article.get('description'),
                    'content': article.get('content'),
                    # 'author': article.get('author'),
                    'source': article['source']['name'],
                    # 'url': article.get('url'),
                    # 'publishedAt': article.get('publishedAt')
                })
        else:
            logging.info(f"Failed to fetch news: {response.status_code}")
            break

    df_articles = pd.DataFrame(articles_list)

    # Define a mapping of sources to labels
    source_labels = {
        "BBC News": 1,
        "CNN": 1,
        "Bloomberg": 1,
        "The Washington Post": 1,
        "Associated Press": 1,
        "Breitbart News": 0,
        "Fox News": 0,
        "infowars.com": 0,
        "RT":0,
        "thegatewaypundit.com": 0,
        "worldnewsdailyreport.com": 0
    }

    # Add a 'label' column to df_articles based on the source
    df_articles['label'] = df_articles['source'].map(source_labels)

    # Display the updated DataFrame
    # display(df_articles.head())

    # Save the labeled DataFrame to a CSV file
    df_articles.to_csv(news_articles_path, index=False)

    # del articles_list
    logging.info('api fetch completed!\n')



def web_scraping_news(base_url, max_articles=100, output_csv="scraped_articles.csv"):
    """
    Scrape up to max_articles news articles from the given base_url and save to a CSV file.
    Args:
        base_url (str): The news website's main page URL.
        max_articles (int): Maximum number of articles to scrape.
        output_csv (str): Output CSV file path.
    """
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    article_links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Adjust the filter as per the site's URL structure
        if '/news/' in href and href.startswith('/news/'):
            article_links.add('https://www.bbc.com' + href)

    logging.info(f"Found {len(article_links)} article links on the main page.")

    articles_data = []
    count = 0
    for url in list(article_links):
        if count >= max_articles:
            break
        try:
            article = Article(url)
            article.download()
            article.parse()
            articles_data.append({
                'url': url,
                'title': article.title,
                'text': article.text,
                'publish_date': article.publish_date,
                'authors': ', '.join(article.authors)
            })
            count += 1
            logging.info(f"Scraped: {article.title}")
            time.sleep(1)  # Be polite!
        except Exception as e:
            logging.warning(f"Failed to scrape {url}: {e}")

    df = pd.DataFrame(articles_data)
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved {len(df)} articles to {output_csv}")
