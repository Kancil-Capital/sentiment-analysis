import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

@dataclass
class ModelConfig:
    SENTIMENT_MODEL: str ="ProsusAI/finbert"
    MAX_SEQUENCE_LENGTH: int =512
    MIN_CONFIDENCE: float =0.3
    SPACY_MODEL: str ="en_core_web_lg" #strong model, can be changed to medium or small for faster performance 
    CONTEXT_WINDOW: int =150 #number of words around an entity to consider for sentiment analysis

_spacy_nlp = None
_sentiment_analyzer = None
_ticker_lookup = None

def get_spacy_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load(ModelConfig.SPACY_MODEL)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", ModelConfig.SPACY_MODEL])
            _spacy_nlp = spacy.load(ModelConfig.SPACY_MODEL)
    return _spacy_nlp

def get_ticker_lookup() -> dict:
    """
    Build company name -> ticker lookup
    """
    global _ticker_lookup
    if _ticker_lookup is not None:
        return _ticker_lookup

    _ticker_lookup = {}

    #try pytickersymbols first
    try:
        from pytickersymbols import PyTickerSymbols
        stock_data = PyTickerSymbols()
        indices = ['S&P 500', 'NASDAQ 100', 'DOW JONES', 'FTSE 100', 'DAX']
        for index in indices:
            try:
                stocks = list(stock_data.get_stocks_by_index(index))
                for stock in stocks:
                    name = stock.get('name', '').lower()
                    symbols = stock.get('symbols', [])
                    for sym in symbols:
                        ticker = sym.get('yahoo', sym.get('google',''))
                        if name and ticker:
                            _ticker_lookup[name] = ticker
                            short_name = name.split()[0] if name else ''
                            if short_name and len(short_name) > 3:
                                _ticker_lookup[short_name] = ticker
            except Exception: #to be refactored, this some ass code
                continue

    except ImportError:
        pass

    manual_mappings = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
        'amazon': 'AMZN', 'meta': 'META', 'facebook': 'META', 'tesla': 'TSLA',
        'nvidia': 'NVDA', 'netflix': 'NFLX', 'adobe': 'ADBE', 'salesforce': 'CRM',
        'intel': 'INTC', 'amd': 'AMD', 'qualcomm': 'QCOM', 'cisco': 'CSCO',
        'oracle': 'ORCL', 'ibm': 'IBM', 'paypal': 'PYPL', 'visa': 'V',
        'mastercard': 'MA', 'jpmorgan': 'JPM', 'goldman sachs': 'GS',
        'bank of america': 'BAC', 'wells fargo': 'WFC', 'citigroup': 'C',
        'walmart': 'WMT', 'target': 'TGT', 'costco': 'COST', 'home depot': 'HD',
        'coca-cola': 'KO', 'pepsi': 'PEP', 'pepsico': 'PEP', 'nike': 'NKE',
        'disney': 'DIS', 'boeing': 'BA', 'lockheed martin': 'LMT',
        'exxon': 'XOM', 'chevron': 'CVX', 'shell': 'SHEL',
        'pfizer': 'PFE', 'johnson & johnson': 'JNJ', 'moderna': 'MRNA',
        'uber': 'UBER', 'airbnb': 'ABNB', 'spotify': 'SPOT', 'twitter': 'X',
    }
    _ticker_lookup.update(manual_mappings)
    
    return _ticker_lookup

def extract_tickers(text: str) -> list[tuple[str, float]]:
    """
    Extract tickers from text using cashtag matching, spaCy NER for organizations, and company name lookup
    Returns list of (ticker, confidence)
    """
    found_tickers = {}

    # Cashtag extraction
    cashtags = re.findall(r'\$([A-Z]{1,5})', text)
    for ticker in cashtags:
        found_tickers[ticker] = max(found_tickers.get(ticker, 0.0), 0.95)
    
    potential_tickers = re.findall(r'\b([A-Z]{2,5})\b', text)
    ticker_lookup = get_ticker_lookup()
    
    known_tickers = set(ticker_lookup.values())

    lower_text = text.lower()
    # Fallback: direct company-name match (catches NVIDIA even if spaCy misses it)
    for name, tick in ticker_lookup.items():
        if len(name) >= 4 and name in lower_text:
            in_title = lower_text.find(name) < 200
            score = 0.75 if in_title else 0.55
            found_tickers[tick] = max(found_tickers.get(tick, 0.0), score)
 

    for potential in potential_tickers:
        if potential in known_tickers:
            in_title = potential in text[:200]  # Check if in first 200 characters
            score = 0.85 if in_title else 0.65
            found_tickers[potential] = max(found_tickers.get(potential, 0.0), score)

    # NER extraction using spaCy
    nlp = get_spacy_nlp()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ('ORG', 'GPE', 'PRODUCT'):
            org_name = ent.text.lower()
            ticker = ticker_lookup.get(org_name)

            if not ticker:
                for name, tick in ticker_lookup.items():
                    if name in org_name or org_name in name:
                        ticker = tick
                        break

            if ticker:
                in_title = ent.start_char < 200  # Check if in first 200 characters
                score = 0.8 if in_title else 0.6
                found_tickers[ticker] = max(found_tickers.get(ticker, 0.0), score)
    
    return [(ticker, score) for ticker, score in found_tickers.items()]

def get_sentiment(
    title: str,
    body: str,
    source: str,
    author: str | None
) -> list[tuple[str, float, float]]:
    """
    Gets sentiment for an article
    Returns list of (affected_party, sentiment_score, confidence)
    """
    raise NotImplementedError()
