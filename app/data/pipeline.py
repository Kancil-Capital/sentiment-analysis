import os

from supabase import create_client, Client
import yfinance as yf
from dotenv import load_dotenv

from app.model.main import get_sentiment

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
    - Fetch news data from 2024
    - Get sentiment and insert into database

    Returns the row that is inserted into database
    """
    # fetch metadata from yf
    ticker_metadata = yf.Ticker(ticker).info
    if ticker_metadata.get("displayName") is None:
        raise ValueError("Invalid ticker")

    # insert metadata into database
    sb_client.table("tickers")\
        .insert({
        "symbol": ticker,
        "region": ticker_metadata["region"],
        "sector": ticker_metadata["sector"],
        "country": ticker_metadata["country"]
    }).execute()

    # TODO: fetch news data from 2024
    # TODO: get sentiment for news

    return [{
        "symbol": ticker,
        "region": ticker_metadata["region"],
        "sector": ticker_metadata["sector"],
        "country": ticker_metadata["country"]
    }]

if __name__ == "__main__":
    daily_pipeline()
