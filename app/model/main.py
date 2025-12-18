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
    SPACY_MODEL: str ="en_core_web_sm" #smaller model for faster performance for now, can upgrade to md or lg later
    CONTEXT_WINDOW: int =150 #number of words around an entity to consider for sentiment analysis

_spacy_nlp = None
_sentiment_analyzer = None
_ticker_lookup_ = None

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

def get_ticker_loadup() -> dict:
    """
    Build company name -> ticker lookup
    """
    global _ticker_lookup
    if _ticker_lookup is not None:
        return _ticker_lookup

    _ticker_lookup_ = {}

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
                            _ticker_lookup_[name] = ticker
                            short_name = name.split()[0] if name else ''
                            if short_name and len(short_name) > 3:
                                _ticker_lookup_[short_name] = ticker
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

