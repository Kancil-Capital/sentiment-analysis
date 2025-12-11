from datetime import datetime
import pandas as pd

from app.model.main import get_sentiment

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
    raise NotImplementedError()

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
    raise NotImplementedError()
