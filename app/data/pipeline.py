import os
from datetime import datetime

from datetime import datetime, timedelta
import pandas as pd

from supabase import create_client, Client
import yfinance as yf
from dotenv import load_dotenv

from app.model.main import get_sentiment
from app.data.scrapers import CNBCScraper, FinvizScraper, YFinanceScraper

# Adding helper fx: 
def process_and_save_news(ticker: str, articles: list, sb_client: Client):
    """
    Take a list of raw news articles, calculates sentiment, then adds to the database
    """
    # Duplicate URL checking
    existing_url_responce = sb_client.table("news_articles")\
        .select()\
        .eq("ticker",ticker)\
        .execute()

    # Set of URLS for lookup purposes
    existing_urls = {row["url"] for row in existing_url_responce}
    
    new_rows_to_add = []

    # Loop to skip existing URLs
    for article in articles:
        if article.get("url") in existing_urls:
            continue

    # Fallback: if paywalls etc block web scrapping for some URLs
    fallback_to_analyse = article.get("body") or article.get("title")

    sentiment_score, confidence = get_sentiment(fallback_to_analyse)

    # Add to database 
    # Note: if there are column mismatch, revisit this part
    new_rows_to_add.append({
            "ticker": ticker,
            "title": article.get('title'),
            "body": article.get('body'),
            "url": article.get('url'),
            "timestamp": article.get('timestamp'),
            "source": article.get('source'),
            "sentiment": sentiment_score,
            "confidence": confidence
        })
    
    # Bulk add into database
    if new_rows_to_add:
        try:
            sb_client.table("news_articles").insert(new_rows_to_add).execute()
            print(f"[{ticker}] Success: Saved {len(new_rows_to_add)} new articles.")
        except Exception as e:
            print(f"[{ticker}] Database Error: {e}")





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
    for ticker in tickers_data:
        keywords.add(ticker["symbol"])
        keywords.add(ticker["region"])
        keywords.add(ticker["sector"])
        keywords.add(ticker["country"])

    # TODO: fetch news data from news source
    # TODO: get sentiments from model
    # TODO: insert into database

    raise NotImplementedError()

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

    # fetch news data from 2025
    keywords = {
        ticker_metadata["region"],
        ticker_metadata["sector"],
        ticker_metadata["country"]
    }

    articles = []

    for kw in keywords:
        # sector and region news are only available on cnbc
        scraper = CNBCScraper()
        articles.extend(
            scraper.scrape(kw, datetime.fromisoformat("2025-01-01"), is_ticker=False)
        )

    # fetch ticker news from all sources
    scrapers = [
        CNBCScraper(), FinvizScraper(), YFinanceScraper()
    ]

    for scraper in scrapers:
        articles.extend(
            scraper.scrape(ticker, datetime.fromisoformat("2025-01-01"), is_ticker=True)
        )

    # insert into database
    sb_client.table("articles")\
        .upsert(
            [a.to_json() for a in articles], on_conflict="url", ignore_duplicates=True
        ).execute()


    # TODO: get sentiment for each article

    return [{
        "symbol": ticker,
        "region": ticker_metadata["region"],
        "sector": ticker_metadata["sector"],
        "country": ticker_metadata["country"]
    }]

if __name__ == "__main__":
    daily_pipeline()
