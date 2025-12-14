import os
from datetime import datetime

from supabase import create_client
import pandas as pd
import yfinance as yf

from app.data.pipeline import insert_ticker

if os.getenv("SUPABASE_URL") is None:
    from dotenv import load_dotenv
    load_dotenv(".env")


sb = create_client(
    supabase_url=os.environ["SUPABASE_URL"],
    supabase_key=os.environ["SUPABASE_SECRET_KEY"]
)

def get_articles(
    ticker: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Get news articles for a :ticker from :start_date to :end_date
    Returns a dataframe with the following columns

    title: str
    body: str
    url: str
    timestamp: pd.Timestamp
    source: str
    author: str | None
    sentiment: float
    confidence: float
    """
    ticker_data = sb.table("tickers")\
        .select("*")\
        .eq("symbol", ticker)\
        .execute().data

    if not ticker_data:
        # Ticker not yet in database, trigger new add and try again
        ticker_data = insert_ticker(ticker, sb)

    keywords = [
        ticker_data[0]["symbol"], ticker_data[0]["region"], ticker_data[0]["sector"], ticker_data[0]["country"]
    ]

    # get all articles with these keywords in the affected column
    articles = sb.rpc("get_sentiments", {
        "keywords": keywords,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }).execute().data

    df = pd.DataFrame(articles)
    df = df[["title", "body", "url", "timestamp", "source", "author", "sentiment", "confidence"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df

def get_price_data(
    ticker: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Get daily price data for a :ticker from :start_date to :end_date
    Returns a dataframe with the following columns

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int
    """
    df = yf.download(ticker, start=start_date, end=end_date,
                     progress=False, auto_adjust=True)

    df.reset_index(inplace=True)
 
    # column renaming
    df.rename(columns={
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)
 
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

if __name__ == "__main__":
    # Testing code

    print(get_articles("AAPL", datetime.fromisoformat("2025-01-01"), datetime.now()))
    print(get_price_data("AAPL", datetime.fromisoformat("2025-01-01"), datetime.now()))
