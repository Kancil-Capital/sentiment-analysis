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
            subprocess.run(["python", "-m", "spacy", "download", ModelConfig.SPACY_MODEL], check=True)
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

#sector classifiction

SECTOR_KEYWORDS = { #gpted this, will cross check when have time
    'Technology': [
        'software', 'hardware', 'semiconductor', 'chip', 'AI', 'artificial intelligence',
        'cloud', 'SaaS', 'cybersecurity', 'tech', 'digital', 'computing', 'data center',
        'machine learning', 'algorithm', 'app', 'platform', 'internet', 'startup'
    ],
    'Healthcare': [
        'pharma', 'pharmaceutical', 'biotech', 'drug', 'FDA', 'clinical trial',
        'healthcare', 'medical', 'hospital', 'vaccine', 'therapeutic', 'patient',
        'diagnosis', 'treatment', 'medicine', 'health'
    ],
    'Financials': [
        'bank', 'banking', 'financial', 'investment', 'hedge fund', 'interest rate',
        'Fed', 'Federal Reserve', 'mortgage', 'loan', 'credit', 'fintech', 'insurance',
        'asset management', 'wealth', 'trading', 'stock', 'bond'
    ],
    'Energy': [
        'oil', 'gas', 'petroleum', 'crude', 'OPEC', 'renewable', 'solar', 'wind',
        'energy', 'utility', 'power', 'drilling', 'refinery', 'pipeline', 'fossil fuel'
    ],
    'Consumer Discretionary': [
        'retail', 'e-commerce', 'shopping', 'brand', 'luxury', 'apparel', 'fashion',
        'restaurant', 'hotel', 'travel', 'leisure', 'entertainment', 'gaming'
    ],
    'Consumer Staples': [
        'food', 'beverage', 'grocery', 'supermarket', 'household', 'tobacco',
        'personal care', 'consumer goods'
    ],
    'Industrials': [
        'manufacturing', 'industrial', 'factory', 'supply chain', 'logistics',
        'aerospace', 'defense', 'machinery', 'construction', 'transportation'
    ],
    'Real Estate': [
        'real estate', 'property', 'REIT', 'housing', 'commercial property',
        'residential', 'rent', 'lease', 'mortgage', 'development'
    ],
    'Materials': [
        'mining', 'metals', 'steel', 'aluminum', 'copper', 'gold', 'silver',
        'chemical', 'commodity', 'raw material'
    ],
    'Utilities': [
        'electric', 'water', 'gas utility', 'power grid', 'regulated', 'utility'
    ],
    'Communication Services': [
        'telecom', 'media', 'streaming', 'social media', 'advertising',
        'broadcast', 'cable', 'wireless', '5G'
    ]
}

def extract_sectors(text: str) -> list[tuple[str, float]]:
    """
    Extracts relevant sectors from text/
    Returns list of (sector, confidence)
    """
    text_lower = text.lower()
    sector_scores = {}

    for sector, keywords in SECTOR_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        if matches > 0: #normalise score based on keyword density
            score = min(0.9, 0.3 + (matches * 0.12))
            sector_scores[sector] = score
    
    return [(sector, score) for sector, score in sector_scores.items()]

#geographic extraction
REGION_MAPPINGS = {
    # Countries to regions
    'united states': 'US', 'usa': 'US', 'america': 'US',
    'china': 'China', 'chinese': 'China',
    'japan': 'Japan', 'japanese': 'Japan',
    'united kingdom': 'UK', 'britain': 'UK', 'british': 'UK', 'england': 'UK',
    'germany': 'EU', 'france': 'EU', 'italy': 'EU', 'spain': 'EU',
    'european union': 'EU', 'eurozone': 'EU', 'europe': 'EU',
    'india': 'India',
    'brazil': 'LATAM', 'mexico': 'LATAM', 'argentina': 'LATAM',
    'korea': 'APAC', 'south korea': 'APAC', 'singapore': 'APAC',
    'hong kong': 'APAC', 'taiwan': 'APAC', 'australia': 'APAC',
    
    # Financial centers
    'wall street': 'US', 'nasdaq': 'US', 'nyse': 'US', 'silicon valley': 'US',
    'city of london': 'UK', 'ftse': 'UK',
    'shanghai': 'China', 'beijing': 'China', 'shenzhen': 'China',
    'tokyo': 'Japan', 'nikkei': 'Japan',
    'frankfurt': 'EU', 'ecb': 'EU',
    
    # Central banks (important for financial news)
    'federal reserve': 'US', 'fed': 'US',
    'bank of england': 'UK', 'boe': 'UK',
    'european central bank': 'EU',
    'pboc': 'China', "people's bank of china": 'China',
    'bank of japan': 'Japan', 'boj': 'Japan',
}

def extract_geographies(text: str) -> list[tuple[str, float]]:
    """
    Extract geographic regions using SpaCy NER and pycountry
    Returns list of (region, confidence)
    """
    region_scores = {}
    text_lower = text.lower()

    #keyword mappings first (catches more specific financial terms)
    for keyword, region in REGION_MAPPINGS.items():
        if keyword in text_lower:
            current_score = region_scores.get(region, 0.0)
            region_scores[region] = max(current_score, 0.7)
                                        
    #spaCy NER
    nlp = get_spacy_nlp()
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ in ('GPE', 'LOC'):
            location = ent.text.lower()

            region = REGION_MAPPINGS.get(location)

            if not region:
                try:
                    import pycountry
                    matches = pycountry.countries.search_fuzzy(ent.text)
                    if matches:
                        country_name = matches[0].name.lower()
                        region = REGION_MAPPINGS.get(country_name, country_name.title())
                except (ImportError, LookupError):
                    region = ent.text

            if region:
                in_title = ent.start_char < 200
                score = 0.8 if in_title else 0.6
                region_scores[region] = max(region_scores.get(region, 0.0), score)

    return [(region, score) for region, score in region_scores.items()]

# SENTIMENT ANALYSIS
class SentimentAnalyzer:
    def __init__(self, model_name: str = ModelConfig.SENTIMENT_MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

        if 'finbert' in model_name.lower():
            self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        else:
            self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    def analyze(self, text: str) -> tuple[float, float]:
        """
        Returns (sentiment_score, confidence)
        - sentiment_score = P(postive) - P(negative)
        - confidence = prediction strength
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=ModelConfig.MAX_SEQUENCE_LENGTH,
            padding= True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            probs = self.torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        p_pos = probs[self.label_map['positive']].item()
        p_neg = probs[self.label_map['negative']].item()
        p_neutral = probs[self.label_map['neutral']].item()
        sentiment_score = p_pos - p_neg
        confidence = max(p_pos, p_neg) * (1 - p_neutral * 0.5)

        return sentiment_score, confidence
    
def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer

SOURCE_TIERS = {
    1.0: ['reuters', 'bloomberg', 'wsj', 'wall street journal', 'financial times', 'ft', 'sec.gov'],
    0.95: ['cnbc', 'barrons', 'economist', 'marketwatch'],
    0.9: ['cnn', 'bbc', 'nytimes', 'new york times', 'associated press', 'ap news'],
    0.85: ['yahoo finance', 'google finance', 'investing.com'],
    0.8: ['techcrunch', 'seeking alpha', 'motley fool', 'benzinga'],
    0.7: ['reddit', 'twitter', 'stocktwits'],  # Social media - lower weight
}


def get_source_weight(source: str) -> float:
    """Get credibility weight for a news source"""
    source_lower = source.lower()
    for weight, sources in SOURCE_TIERS.items():
        if any(s in source_lower for s in sources):
            return weight
    return 0.75  # Default for unknown sources

def get_sentiment(
    title: str,
    body: str,
    source: str,
    author: str | None
) -> list[tuple[str, float, float]]:
    """
    Gets sentiment for an article

    Args:
        title: Article title
        body: Article body
        source: News source
        author: Article author (optional)

    Returns list of (affected_party, sentiment_score, confidence)
    """
    full_text = f"{title}\n\n{body}"
    
    #Extract all affected parties (tickers)
    tickers = extract_tickers(full_text)
    sectors = extract_sectors(full_text)
    geographies = extract_geographies(full_text)

    #get sentiment
    analyzer = get_sentiment_analyzer()
    title_sentiment, title_conf = analyzer.analyze(title)
    full_sentiment, full_conf = analyzer.analyze(full_text[:2000]) #limit to first 2000 chars for performance

    #weighted combination (title higher weightage)
    base_sentiment = 0.4 * title_sentiment + 0.6 * full_sentiment
    base_confidence = 0.4 * title_conf + 0.6 * full_conf

    #source weight
    source_weight = get_source_weight(source)

    #build results
    results: list[tuple[str, float, float]] = []

    #tickers (highest weightage)
    for ticker, relevance in tickers:
        conf = base_confidence * source_weight * relevance
        if conf >= ModelConfig.MIN_CONFIDENCE:
            results.append((ticker, round(base_sentiment, 4), round(conf, 4)))
    
    #sectors
    for sector, relevance in sectors:
        conf = base_confidence * source_weight * relevance * 0.85
        if conf >= ModelConfig.MIN_CONFIDENCE:
            results.append((sector, round(base_sentiment, 4), round(conf, 4)))

    #geographies
    for region, relevance in geographies:
        conf = base_confidence * source_weight * relevance * 0.75
        if conf >= ModelConfig.MIN_CONFIDENCE:
            results.append((region, round(base_sentiment, 4), round(conf, 4)))

    results.sort(key=lambda x: x[2], reverse=True)  # Sort by confidence descending
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("SENTIMENT ANALYSIS TEST")
    print("=" * 70)
    
    test_cases = [
        {
            "title": "Apple Reports Record Q4 Earnings, Beats Wall Street Expectations",
            "body": """Apple Inc. announced record-breaking fourth quarter results today, 
            with revenue exceeding analyst expectations by 5%. The tech giant reported 
            strong iPhone sales across all regions, particularly in China and Europe.
            CEO Tim Cook expressed optimism about AI initiatives. Goldman Sachs upgraded 
            the stock to "Buy".""",
            "source": "Reuters",
        },
        {
            "title": "Federal Reserve Signals Potential Rate Cuts Amid Slowing Inflation",
            "body": """The Federal Reserve indicated it may begin cutting interest rates
            in the coming months as inflation shows signs of cooling. This news boosted
            bank stocks and the broader financial sector. European markets also rallied
            on expectations of similar moves from the ECB.""",
            "source": "Bloomberg",
        },
        {
            "title": "Tesla Shares Plunge After Missing Delivery Targets",
            "body": """Tesla stock fell sharply after the company reported Q3 deliveries
            below analyst expectations. The electric vehicle maker blamed supply chain
            issues and increased competition in China. Analysts at Morgan Stanley 
            downgraded their price target.""",
            "source": "CNBC",
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {test['title'][:50]}...")
        print(f"Source: {test['source']}")
        print("-" * 70)
        
        results = get_sentiment(
            title=test['title'],
            body=test['body'],
            source=test['source'],
            author=None
        )
        
        if results:
            print(f"{'Affected Party':<30} {'Sentiment':>12} {'Confidence':>12}")
            print("-" * 70)
            for party, sentiment, confidence in results:
                label = " Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
                print(f"{party:<30} {sentiment:>+12.4f} {confidence:>12.4f}  {label}")
        else:
            print("No entities extracted with sufficient confidence.")
    
    print("\n" + "=" * 70)
    print("DONE")