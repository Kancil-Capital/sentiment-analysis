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

_spacy_nlp = None
_sentiment_analyzer = None
_ticker_lookup = None
_ticker_lookup_lower = None
_known_tickers_set = None

def get_spacy_nlp():
    """Load spaCy model once"""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load(ModelConfig.SPACY_MODEL)
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{ModelConfig.SPACY_MODEL}' not installed. "
                f"Install it via dependencies (pyproject.toml) during build."
            ) from e
    return _spacy_nlp

def get_ticker_lookup() -> tuple[dict, dict, set]:
    """
    Build company name -> ticker lookup
    """
    global _ticker_lookup, _ticker_lookup_lower, _known_tickers_set
    
    if _ticker_lookup is not None:
        return _ticker_lookup, _ticker_lookup_lower, _known_tickers_set

    _ticker_lookup = {}

    #try pytickersymbols first
    try:
        from pytickersymbols import PyTickerSymbols
        stock_data = PyTickerSymbols()
        for index in ['S&P 500', 'NASDAQ 100', 'DOW JONES']:  # Reduced for speed
            try:
                for stock in stock_data.get_stocks_by_index(index):
                    name = stock.get('name', '').lower()
                    for sym in stock.get('symbols', []):
                        ticker = sym.get('yahoo', sym.get('google', ''))
                        if name and ticker:
                            _ticker_lookup[name] = ticker
                            # Add first word if long enough
                            first = name.split()[0]
                            if len(first) > 4:
                                _ticker_lookup[first] = ticker
            except Exception:
                continue
    except ImportError:
        pass

    # Manual mappings (guaranteed clean US tickers)
    manual_mappings = {
        # Tech
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
        'amazon': 'AMZN', 'meta': 'META', 'facebook': 'META', 'tesla': 'TSLA',
        'nvidia': 'NVDA', 'netflix': 'NFLX', 'adobe': 'ADBE', 'salesforce': 'CRM',
        'intel': 'INTC', 'amd': 'AMD', 'qualcomm': 'QCOM', 'cisco': 'CSCO',
        'oracle': 'ORCL', 'ibm': 'IBM', 'palantir': 'PLTR', 'snowflake': 'SNOW',
        
        # Finance
        'paypal': 'PYPL', 'visa': 'V', 'mastercard': 'MA',
        'jpmorgan': 'JPM', 'jp morgan': 'JPM',
        'goldman sachs': 'GS', 'goldman': 'GS',
        'morgan stanley': 'MS',
        'bank of america': 'BAC', 'wells fargo': 'WFC', 'citigroup': 'C',
        
        # Consumer
        'walmart': 'WMT', 'target': 'TGT', 'costco': 'COST', 'home depot': 'HD',
        'coca-cola': 'KO', 'pepsi': 'PEP', 'pepsico': 'PEP', 'nike': 'NKE',
        'disney': 'DIS', 'starbucks': 'SBUX', 'mcdonalds': 'MCD',
        
        # Industrial/Energy/Auto
        'boeing': 'BA', 'lockheed martin': 'LMT', 'caterpillar': 'CAT',
        'exxon': 'XOM', 'chevron': 'CVX',
        'ford': 'F', 'gm': 'GM', 'general motors': 'GM',
        
        # Healthcare
        'pfizer': 'PFE', 'johnson & johnson': 'JNJ', 'moderna': 'MRNA',
        'unitedhealth': 'UNH', 'eli lilly': 'LLY',
        
        # Other
        'uber': 'UBER', 'airbnb': 'ABNB', 'spotify': 'SPOT',
    }

    _ticker_lookup.update(manual_mappings)
    _ticker_lookup_lower = {k.lower(): v for k, v in _ticker_lookup.items()}
    _known_tickers_set = set(_ticker_lookup.values())
    return _ticker_lookup, _ticker_lookup_lower, _known_tickers_set

#sector classifiction

SECTOR_KEYWORDS = {
    'Technology': ['software', 'semiconductor', 'chip', 'AI', 'cloud', 'tech', 'SaaS'],
    'Healthcare': ['pharma', 'biotech', 'drug', 'FDA', 'healthcare', 'medical'],
    'Financials': ['bank', 'financial', 'investment', 'interest rate', 'Fed', 'mortgage'],
    'Energy': ['oil', 'gas', 'petroleum', 'OPEC', 'renewable', 'solar', 'energy'],
    'Consumer': ['retail', 'e-commerce', 'shopping', 'brand', 'consumer'],
}

REGION_MAPPINGS = {
    'united states': 'US', 'usa': 'US', 'america': 'US', 'u.s.': 'US',
    'china': 'China', 'chinese': 'China',
    'japan': 'Japan', 'japanese': 'Japan',
    'united kingdom': 'UK', 'britain': 'UK', 'uk': 'UK',
    'germany': 'EU', 'france': 'EU', 'europe': 'EU', 'european': 'EU',
    'federal reserve': 'US', 'fed': 'US', 'wall street': 'US',
    'ecb': 'EU', 'european central bank': 'EU',
}

_sector_patterns = None

def get_sector_patterns():
    global _sector_patterns
    if _sector_patterns is None:
        _sector_patterns = {}
        for sector, keywords in SECTOR_KEYWORDS.items():
            pattern = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b', re.IGNORECASE)
            _sector_patterns[sector] = pattern
    return _sector_patterns

@dataclass
class EntityMention:
    """An entity mention with position for context extraction"""
    text: str           # Original text ("Apple", "$TSLA")
    ticker: str         # Normalized ("AAPL", "TSLA")
    entity_type: str    # "TICKER", "SECTOR", "REGION"
    start: int          # Start position in text
    end: int            # End position
    relevance: float    # Confidence score



def extract_entities_with_positions(text: str) -> list[EntityMention]:
    """
    Extracts relevant sectors and geographies from text.
    Returns list of EntityMention objects
    """
    ticker_lookup, ticker_lookup_lower, known_tickers = get_ticker_lookup()
    text_lower = text.lower()

    entities = []
    seen_positions = set()

    #Cashtags ($AAPL)
    for match in re.finditer(r'\$([A-Z]{1,5})\b', text):
        ticker = match.group(1)
        if ticker in known_tickers:
            pos_key = (match.start(), ticker)
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                entities.append(EntityMention(
                    text=match.group(0),
                    ticker=ticker,
                    entity_type="TICKER",
                    start=match.start(),
                    end=match.end(),
                    relevance=0.95
                ))
    
    #Direct tickers
    for match in re.finditer(r'\b([A-Z]{2,5})\b', text):
        potential = match.group(1)
        if potential in known_tickers:
            pos_key = (match.start(), potential)
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                in_title = match.start() < 200
                entities.append(EntityMention(
                    text=potential,
                    ticker=potential,
                    entity_type="TICKER",
                    start=match.start(),
                    end=match.end(),
                    relevance=0.85 if in_title else 0.65
                ))
    
    #Company name matching
    for name, ticker in ticker_lookup_lower.items():
        if len(name) >= 4 and ticker in known_tickers:
            start = 0
            while True:
                pos = text_lower.find(name, start)
                if pos == -1:
                    break
                pos_key = (pos, ticker)
                if pos_key not in seen_positions:
                    seen_positions.add(pos_key)
                    in_title = pos < 200
                    entities.append(EntityMention(
                        text=text[pos:pos+len(name)],
                        ticker=ticker,
                        entity_type="TICKER",
                        start=pos,
                        end=pos + len(name),
                        relevance=0.75 if in_title else 0.55
                    ))
                start = pos + 1
    
    # 4. spaCy NER
    nlp = get_spacy_nlp()
    doc = nlp(text)
    
    for ent in doc.ents:
        in_title = ent.start_char < 200
        
        if ent.label_ == 'ORG':
            org_lower = ent.text.lower()
            ticker = ticker_lookup_lower.get(org_lower)
            
            if not ticker:
                first_word = org_lower.split()[0]
                ticker = ticker_lookup_lower.get(first_word)
            
            if ticker and ticker in known_tickers:
                pos_key = (ent.start_char, ticker)
                if pos_key not in seen_positions:
                    seen_positions.add(pos_key)
                    entities.append(EntityMention(
                        text=ent.text,
                        ticker=ticker,
                        entity_type="TICKER",
                        start=ent.start_char,
                        end=ent.end_char,
                        relevance=0.8 if in_title else 0.6
                    ))
        
        elif ent.label_ in ('GPE', 'LOC'):
            loc_lower = ent.text.lower()
            region = REGION_MAPPINGS.get(loc_lower)
            if region:
                entities.append(EntityMention(
                    text=ent.text,
                    ticker=region,
                    entity_type="REGION",
                    start=ent.start_char,
                    end=ent.end_char,
                    relevance=0.7 if in_title else 0.5
                ))
    
    # 5. Sector keywords
    for sector, pattern in get_sector_patterns().items():
        for match in pattern.finditer(text):
            entities.append(EntityMention(
                text=match.group(0),
                ticker=sector,
                entity_type="SECTOR",
                start=match.start(),
                end=match.end(),
                relevance=0.6
            ))
    
    # 6. Region keywords
    for keyword, region in REGION_MAPPINGS.items():
        pos = text_lower.find(keyword)
        if pos != -1:
            entities.append(EntityMention(
                text=keyword,
                ticker=region,
                entity_type="REGION",
                start=pos,
                end=pos + len(keyword),
                relevance=0.6
            ))
    
    return entities

def get_entity_context(text:str, entity: EntityMention, window: int = 200) -> str:
    """
    Extract context window around entity mention
    """
    start = max(0, entity.start - window)
    end = min(len(text), entity.end + window)
    
    sentence_start = text.rfind('.', start, entity.start)
    if sentence_start != -1 and sentence_start > start:
        start = sentence_start + 1
    
    sentence_end = text.find('.', entity.end, end)
    if sentence_end != -1:
        end = sentence_end + 1
    
    return text[start:end].strip()

    
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
        sentiment = p_pos - p_neg
        confidence = max(p_pos, p_neg) * (1 - p_neutral * 0.5)

        return sentiment, confidence
    
    def analyze_batch(self, texts: list[str]) -> list[tuple[float, float]]:
        """
        Analyze a batch of texts at once
        """
        if not texts:
            return []
        
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=ModelConfig.MAX_SEQUENCE_LENGTH,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            probs = self.torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        results = []
        for i in range(len(texts)):
            p_pos = probs[i][self.label_map['positive']].item()
            p_neg = probs[i][self.label_map['negative']].item()
            p_neutral = probs[i][self.label_map['neutral']].item()
            sentiment = p_pos - p_neg
            confidence = max(p_pos, p_neg) * (1 - p_neutral * 0.5)
            results.append((sentiment, confidence))
        
        return results
    
_sentiment_analyzer = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer

SOURCE_WEIGHTS = {
    'reuters': 1.0, 'bloomberg': 1.0, 'wsj': 1.0, 'financial times': 1.0,
    'cnbc': 0.95, 'barrons': 0.95, 'marketwatch': 0.9,
    'yahoo finance': 0.85, 'seeking alpha': 0.8, 'motley fool': 0.8,
}

def get_source_weight(source: str) -> float:
    source_lower = source.lower()
    for name, weight in SOURCE_WEIGHTS.items():
        if name in source_lower:
            return weight
    return 0.75


def get_sentiment(
    title: str,
    body: str,
    source: str,
    author: str | None
) -> list[tuple[str, float, float]]:
    """
    Gets sentiment for an article.
    
    OPTIMIZED VERSION:
    - Single spaCy pass
    - Single sentiment analysis (not duplicate)
    - Pre-computed lookups
    """
    full_text = f"{title}\n\n{body}"
    
    # Extract entities with positions
    entities = extract_entities_with_positions(full_text)
    
    if not entities:
        return []
    
    # Group entities by (type, ticker)
    entity_groups: dict[str, list[EntityMention]] = {}
    for ent in entities:
        if ent.entity_type == "TICKER":
            key = ent.ticker
        else:
            key = f"{ent.entity_type}:{ent.ticker}"
        
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(ent)
    
    # Get analyzer and source weight
    analyzer = get_sentiment_analyzer()
    source_weight = get_source_weight(source)
    
    # Calculate sentiment for EACH entity based on ITS context
    results: list[tuple[str, float, float]] = []
    
    for key, mentions in entity_groups.items():
        # Collect unique contexts for this entity
        contexts = []
        relevances = []
        seen_contexts = set()
        
        for mention in mentions:
            context = get_entity_context(full_text, mention)
            # Avoid duplicate contexts
            context_key = context[:100]  # Use first 100 chars as key
            if context_key not in seen_contexts:
                seen_contexts.add(context_key)
                contexts.append(context)
                relevances.append(mention.relevance)
        
        if not contexts:
            continue
        
        # Batch analyze all contexts for this entity
        sentiment_results = analyzer.analyze_batch(contexts)
        
        # Weighted average by relevance
        total_weight = sum(relevances)
        weighted_sentiment = sum(s * r for (s, _), r in zip(sentiment_results, relevances))
        weighted_confidence = sum(c * r for (_, c), r in zip(sentiment_results, relevances))
        
        avg_sentiment = weighted_sentiment / total_weight
        avg_confidence = weighted_confidence / total_weight
        
        # Apply source weight and max relevance
        max_relevance = max(relevances)
        final_confidence = avg_confidence * source_weight * max_relevance
        
        # Type-specific multiplier
        if mentions[0].entity_type == "SECTOR":
            final_confidence *= 0.85
        elif mentions[0].entity_type == "REGION":
            final_confidence *= 0.75
        
        if final_confidence >= ModelConfig.MIN_CONFIDENCE:
            results.append((
                key,
                round(avg_sentiment, 4),
                round(final_confidence, 4)
            ))
    
    # Sort by confidence and remove duplicates
    results.sort(key=lambda x: x[2], reverse=True)
    
    seen = set()
    unique_results = []
    for r in results:
        if r[0] not in seen:
            seen.add(r[0])
            unique_results.append(r)
    
    return unique_results

def get_sentiment_batch(
    articles: list[dict]
) -> list[list[tuple[str, float, float]]]:
    """
    Process multiple articles efficiently.
    
    Args:
        articles: List of {"title": str, "body": str, "source": str}
    
    Returns:
        List of results for each article
    """
    if not articles:
        return []
    
    _ = get_spacy_nlp()
    analyzer = get_sentiment_analyzer()

    all_results = []

    for article in articles:
        results = get_sentiment(
            article['title'],
            article['body'],
            article['source'],
            None
        )
        all_results.append(results)

    return all_results

#TEST!!

if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("SENTIMENT ANALYSIS")
    print("=" * 70)
    
    test_cases = [
        # 1) Winner vs loser (same sector, opposite outcome)
        {
            "title": "NVIDIA Surges on Blowout Earnings as Intel Warns of Weak Demand",
            "body": """
            NVIDIA reported blockbuster quarterly earnings, beating expectations on revenue and guidance.
            Management said demand for AI chips and data center products remains exceptionally strong.
            Shares jumped after the results.

            In contrast, Intel issued a profit warning, citing weak PC demand and margin pressure.
            The company guided revenue lower and said it expects profitability to deteriorate.
            Analysts downgraded Intel after the announcement.
            """,
            "source": "Reuters",
        },

        # 2) Two companies: one upgraded, one under investigation
        {
            "title": "Apple Upgraded on Services Strength While Meta Faces EU Privacy Probe",
            "body": """
            Apple was upgraded by analysts who cited accelerating services growth and resilient demand.
            The firm raised its price target and said margins could expand.

            Meanwhile, Meta Platforms came under scrutiny after EU regulators opened a privacy investigation.
            Officials said potential violations could lead to significant penalties, pressuring the stock.
            """,
            "source": "Bloomberg",
        },

        # 3) Acquisition: acquirer mixed/negative, target positive
        {
            "title": "Microsoft to Acquire Cybersecurity Firm in Deal Viewed as Expensive",
            "body": """
            Microsoft agreed to acquire a cybersecurity company for a large premium.
            Shares of the target surged as investors welcomed the buyout price.

            Some analysts said the valuation looks expensive and could dilute Microsoft’s near-term earnings,
            even as the deal expands Microsoft’s security offerings over the long term.
            """,
            "source": "WSJ",
        },

        # 4) Product recall: strong negative for one, neutral/positive for competitor
        {
            "title": "Tesla Recalls Vehicles Over Safety Issue as Ford Sees Strong Demand",
            "body": """
            Tesla announced a recall due to a safety issue and said it would address the problem via updates.
            The news raised concerns about quality and potential regulatory scrutiny.

            Separately, Ford reported strong demand for its latest models and said orders remain healthy,
            supporting a more optimistic outlook for the quarter.
            """,
            "source": "CNBC",
        },

        # 5) Bank stress: bank negative, “Fed”/macro mixed, sector negative
        {
            "title": "Bank of America Shares Slide After Margin Warning; Fed Comments Add Uncertainty",
            "body": """
            Bank of America warned that net interest margins may decline due to deposit competition.
            Shares fell as investors worried about profitability.

            Federal Reserve officials said the path of rates will depend on data, adding uncertainty for banks.
            The broader financial sector weakened on the comments.
            """,
            "source": "MarketWatch",
        },

        # 6) Energy: oil surge positive for Exxon/Chevron, negative for airlines/consumer
        {
            "title": "Oil Prices Jump on Supply Shock; Exxon Gains While Airlines Warn of Higher Costs",
            "body": """
            Oil prices jumped after a supply disruption, boosting energy producers.
            Exxon and Chevron shares rose as traders priced in stronger cash flows.

            Airlines warned that higher fuel costs could pressure earnings and force ticket price increases.
            Consumer spending could also slow if energy prices remain elevated.
            """,
            "source": "Financial Times",
        },

        # 7) Litigation: big negative for one company, neutral for sector
        {
            "title": "Johnson & Johnson Hit With Lawsuit Ruling; Analysts Cut Estimates",
            "body": """
            Johnson & Johnson faced a major adverse court ruling in ongoing litigation.
            Analysts said the decision increases financial risk and could lead to higher settlement costs.

            Healthcare peers were largely unchanged as investors focused on company-specific exposure.
            """,
            "source": "Reuters",
        },

        # 8) Neutral/uncertain guidance: should trend closer to neutral
        {
            "title": "Amazon Signals Cautious Outlook; Results In Line With Expectations",
            "body": """
            Amazon reported results largely in line with expectations and reiterated cautious guidance.
            Management said demand is steady but visibility remains limited due to macro uncertainty.

            Analysts described the update as balanced, with neither a clear upside surprise nor a major miss.
            """,
            "source": "Bloomberg",
        },

        # 9) China exposure: Apple negative, US region mixed, China region negative
        {
            "title": "China Tightens Tech Rules; Apple Suppliers Face Disruption Risk",
            "body": """
            China announced tighter rules affecting technology supply chains, raising disruption risk.
            Analysts warned Apple suppliers could see delays and higher compliance costs.

            Markets reacted cautiously, with investors watching for potential spillover into US tech earnings.
            """,
            "source": "Reuters",
        },

        # 10) Strong positive for consumer brand, negative for competitor
        {
            "title": "Nike Beats Expectations With Strong Demand as Adidas Struggles With Inventory",
            "body": """
            Nike reported better-than-expected sales and said demand remained strong across key regions.
            The company raised its outlook and highlighted improving profitability.

            Adidas warned that elevated inventory levels are forcing discounts, pressuring margins.
            Analysts said near-term results could remain challenged.
            """,
            "source": "FT",
        },

        # 11) Multiple entities in one paragraph (tests context window separation)
        {
            "title": "Alphabet Announces Major AI Expansion; Microsoft Faces Cloud Outage Backlash",
            "body": """
            Alphabet announced a major AI expansion and said new products could boost revenue growth.
            Investors welcomed the strategy and analysts called the plan a catalyst.

            In separate news, Microsoft faced backlash after a cloud outage disrupted customers.
            Some clients reported operational losses and questioned reliability guarantees.
            """,
            "source": "Reuters",
        },

        # 12) Very negative scandal for one, positive recovery for another
        {
            "title": "Wells Fargo Fined Again for Compliance Failures as JPMorgan Sees Record Trading",
            "body": """
            Wells Fargo was fined again after regulators cited repeated compliance failures.
            The penalty raised concerns about governance and ongoing oversight.

            JPMorgan, meanwhile, reported record trading revenue and said market conditions were favorable.
            Analysts praised execution and reaffirmed bullish views on the stock.
            """,
            "source": "Bloomberg",
        },
    ]

    
    # Call models
    _ = get_sentiment(test_cases[0]['title'], test_cases[0]['body'], test_cases[0]['source'], None)
    
    # Test individually
    print("\n" + "=" * 70)
    print("INDIVIDUAL PROCESSING")
    print("=" * 70)
    
    start = time.time()
    for i, test in enumerate(test_cases, 1):
        t0 = time.time()
        results = get_sentiment(test['title'], test['body'], test['source'], None)
        elapsed = time.time() - t0
        
        print(f"\nTest {i}: {test['title'][:45]}...")
        print(f"Time: {elapsed:.3f}s")
        print(f"{'Entity':<25} {'Sentiment':>10} {'Confidence':>10} {'Label':<8}")
        print("-" * 60)
        
        for entity, sentiment, confidence in results[:6]:
            if sentiment > 0.15:
                label = "POS"
            elif sentiment < -0.15:
                label = "NEG"
            else:
                label = "NEU"
            print(f"{entity:<25} {sentiment:>+10.4f} {confidence:>10.4f} {label}")
    
    print(f"\nTotal individual time: {time.time() - start:.2f}s")
    
    # Batch processing
    print("\n" + "=" * 70)
    print("BATCH PROCESSING")
    print("=" * 70)
    
    start = time.time()
    batch_results = get_sentiment_batch(test_cases)
    batch_time = time.time() - start
    
    print(f"\nBatch time for {len(test_cases)} articles: {batch_time:.2f}s")
    print(f"Average per article: {batch_time/len(test_cases):.3f}s")
    
    for i, (test, results) in enumerate(zip(test_cases, batch_results), 1):
        print(f"\nArticle {i}: {test['title'][:45]}...")
        print(f"{'Entity':<25} {'Sentiment':>10} {'Confidence':>10} {'Label':<8}")
        print("-" * 60)
        
        for entity, sentiment, confidence in results[:5]:
            if sentiment > 0.15:
                label = "POS"
            elif sentiment < -0.15:
                label = "NEG"
            else:
                label = "NEU"
            print(f"{entity:<25} {sentiment:>+10.4f} {confidence:>10.4f} {label}")
    
    print("\n" + "=" * 70)
    print("DONE")
