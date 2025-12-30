import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sentiment_data(ticker, days=60):
    """Generate mock sentiment and price data"""
    # Seed ensures the random data looks the same every time you refresh
    np.random.seed(hash(ticker) % 2**32)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate price data
    base_price = np.random.uniform(100, 200)
    price_change = np.random.randn(days).cumsum() * 2
    prices = base_price + price_change
    
    # Generate sentiment data
    sentiment = np.random.randn(days).cumsum() * 0.1
    sentiment = np.clip(sentiment, -1, 1)
    
    # Generate events
    event_indices = np.random.choice(days, size=min(15, days//4), replace=False)
    events = []
    event_types = ['Positive Spike', 'Negative Crash', 'Volume Surge']
    
    for idx in event_indices:
        events.append({
            'date': dates[idx],
            'type': np.random.choice(event_types),
            'price': prices[idx],
            'sentiment': sentiment[idx]
        })
    
    return dates, prices, sentiment, events

def generate_keywords(ticker):
    """Generate keyword data"""
    keywords = [
        'bullish', 'investigation', 'bearish', 'lawsuit', 'upgrade', 
        'growth', 'downgrade', 'earnings', 'layoffs', 'AI', 
        'innovation', 'miss', 'breakout', 'record', 'partnership',
        'overvalued', 'acquisition', 'shortage', 'dividend', 'delay',
        'momentum', 'catalyst', 'guidance', 'volatility'
    ]
    
    sizes = np.random.randint(10, 50, len(keywords))
    return list(zip(keywords, sizes))

def generate_news_articles(ticker):
    """Generate mock news articles"""
    sources = ['MarketWatch', 'Bloomberg', 'Reuters', 'CNBC']
    articles = []
    
    for i in range(5):
        articles.append({
            'source': np.random.choice(sources),
            'title': f'Strategic Partnership Revealed - {ticker}',
            'summary': 'This is a summary of the article about strategic partnership revealed. The analysis suggests significant market implications...',
            'sentiment': round(np.random.uniform(0.3, 0.9), 2),
            'date': datetime.now() - timedelta(days=i)
        })
    
    return articles
