import os
from datetime import datetime

from datetime import datetime, timedelta
import pandas as pd

from supabase import create_client, Client
import yfinance as yf
from dotenv import load_dotenv

from app.model.main import get_sentiment
from app.data.scrapers import CNBCScraper, FinvizScraper, YFinanceScraper

def process_articles(articles: dict[int, dict]) -> tuple[list[int], list[dict]]:
    """
    Gets sentiment for articles and prepares them for database insertion

    param
        articles: {article_id: {title: str, body: str, source: str, author: str}}

    Returns tuple of
        - list of article IDs that do not have sentiment (to be deleted)
        - list of sentiment data in form of
            {article_id: int, affected: str, score: float, confidence: float}
    """
    to_delete =[]
    sentiments_list =[]

    for article_id, content in articles.items():
        text_to_analyze = content.get("body") or content.get("title")

        if not text_to_analyze:
            to_delete.append()
            continue
    try:
        sentiment_score, confidence = get_sentiment(text_to_analyze)

        # Return tuple (Should be ready to insert into database)
        sentiments_list.append({
                "article_id": article_id,
                "affected": content.get("affected", "UNKNOWN"), # e.g., 'AAPL'
                "score": sentiment_score,
                "confidence": confidence
            })
    except Exception as e:
        print(f"{article_id}:{e} has an error")
        to_delete.append(article_id)

    return to_delete, sentiments_list

def daily_pipeline():
    """
    Runs the daily data pipeline

    - Get tracked tickers
    - Fetch news
    - Get sentiment from model
    - Insert into database
    """
    load_dotenv(".env")

    sb = create_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"]
    )

    # get all tracked tickers
    tickers_data = sb.table("tickers")\
        .select("*")\
        .execute().data

    # get keywords to fetch news for
    keywords = set()
    tickers = []
    for ticker in tickers_data:
        tickers.append(ticker["symbol"])
        keywords.add(ticker["region"])
        keywords.add(ticker["sector"])
        keywords.add(ticker["country"])

    # fetch news data from news source
    from_ = datetime.now() - timedelta(days=2) # last 2 days to be safe
    cnbc_scraper = CNBCScraper()
    all_articles = []

    for kw in keywords:
        # sector and region news are only available on cnbc
        all_articles.extend(
            cnbc_scraper.scrape(kw, from_, is_ticker=False)
        )

    scrapers = [
        cnbc_scraper, FinvizScraper(), YFinanceScraper()
    ]

    # fetch ticker news from all sources
    for ticker in tickers:
        for scraper in scrapers:
            all_articles.extend(
                scraper.scrape(ticker, from_, is_ticker=True)
            )

    insert_articles(all_articles, sb)

def insert_ticker(ticker: str, sb_client: Client) -> list[dict]:
    """
    Inserts a new ticker to be tracked

    - Get ticker metadata
    - Fetch news data from 2025
    - Get sentiment and insert into database

    Returns the row that is inserted into database
    """
    # fetch metadata from yf
    ticker_metadata = yf.Ticker(ticker).info
    if ticker_metadata == {"trailingPegRatio": None}:
        raise ValueError("Invalid ticker")

    # insert metadata into database
    sb_client.table("tickers")\
        .insert({
        "symbol": ticker,
        "region": ticker_metadata["region"],
        "sector": ticker_metadata["sector"],
        "country": ticker_metadata["country"]
    }).execute()

    # fetch news data from a year ago
    one_year_ago = datetime.now() - timedelta(days=365)
    keywords = {
        ticker_metadata["region"],
        ticker_metadata["sector"],
        ticker_metadata["country"]
    }

    articles = []

    cnbc_scraper = CNBCScraper()
    for kw in keywords:
        # sector and region news are only available on cnbc
        articles.extend(
            cnbc_scraper.scrape(kw, one_year_ago, is_ticker=False)
        )

    # fetch ticker news from all sources
    scrapers = [
        cnbc_scraper, FinvizScraper(), YFinanceScraper()
    ]

    for scraper in scrapers:
        articles.extend(
            scraper.scrape(ticker, one_year_ago, is_ticker=True)
        )

    insert_articles(articles, sb_client)

    return [{
        "symbol": ticker,
        "region": ticker_metadata["region"],
        "sector": ticker_metadata["sector"],
        "country": ticker_metadata["country"]
    }]

def insert_articles(articles: list, sb_client: Client):
    """
    Insert articles into database, get sentiments and save them
    """
    # insert into database
    inserted_articles = sb_client.table("articles")\
        .upsert(
            [a.to_json() for a in articles], on_conflict="url", ignore_duplicates=True
        ).execute()

    # get sentiment for each article
    articles_to_process = {
        a["id"]: {
            "title": a["title"],
            "body": a["body"],
            "source": a["source"],
            "author": a.get("author")
        } for a in inserted_articles.data
    }

    to_delete, sentiments = process_articles(articles_to_process)

    # delete articles that could not be processed or has no sentiment
    if to_delete:
        sb_client.table("articles")\
            .delete()\
            .in_("id", to_delete)\
            .execute()

    # insert sentiments into database
    if sentiments:
        sb_client.table("sentiments")\
            .insert(sentiments)\
            .execute()

if __name__ == "__main__":
    daily_pipeline()
