import re
from dataclasses import dataclass
from functools import lru_cache
import warnings

warnings.filterwarnings("ignore")

@dataclass
class ModelConfig:
    SENTIMENT_MODEL: str ="ProsusAI/finbert"
    MAX_SEQUENCE_LENGTH: int =512
    MIN_CONFIDENCE: float = 0.25
    SPACY_MODEL: str ="en_core_web_lg"

_spacy_nlp = None
_sentiment_analyzer = None
_ticker_lookup = None
_ticker_lookup_lower = None
_known_tickers_set = None
_phrase_matcher = None

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
    global _ticker_lookup, _ticker_lookup_lower, _known_tickers_set
    
    if _ticker_lookup is not None:
        return _ticker_lookup, _ticker_lookup_lower, _known_tickers_set

    _ticker_lookup = {}

    # 1. Load from pytickersymbols FIRST (lower priority)
    try:
        from pytickersymbols import PyTickerSymbols
        stock_data = PyTickerSymbols()
        for index in ['S&P 500', 'NASDAQ 100', 'DOW JONES']:
            try:
                for stock in stock_data.get_stocks_by_index(index):
                    name = stock.get('name', '').lower()
                    for sym in stock.get('symbols', []):
                        ticker = sym.get('yahoo', sym.get('google', ''))
                        if name and ticker and '.' not in ticker:
                            _ticker_lookup[name] = ticker
            except Exception:
                continue
    except ImportError:
            warnings.warn(
            "Optional dependency 'pytickersymbols' not installed; using manual ticker mappings only.",
            RuntimeWarning,
        )

    # Manual mappings
    manual_mappings = {
        # Tech
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
        'amazon': 'AMZN', 'meta': 'META', 'facebook': 'META', 'tesla': 'TSLA',
        'nvidia': 'NVDA', 'netflix': 'NFLX', 'adobe': 'ADBE', 'salesforce': 'CRM',
        'intel': 'INTC', 'amd': 'AMD', 'qualcomm': 'QCOM', 'cisco': 'CSCO',
        'oracle': 'ORCL', 'ibm': 'IBM', 'palantir': 'PLTR', 'snowflake': 'SNOW',
        'autodesk': 'ADSK', 'adsk': 'ADSK',
        'grab': 'GRAB', 'grab holdings': 'GRAB',
        
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
        'alibaba': 'BABA', 'baba': 'BABA',
        
        # Industrial/Energy/Auto
        'boeing': 'BA', 'lockheed martin': 'LMT', 'caterpillar': 'CAT',
        'exxon': 'XOM', 'chevron': 'CVX',
        'ford': 'F', 'gm': 'GM', 'general motors': 'GM',
        'general electric': 'GE', 'ge': 'GE',
        
        # Healthcare
        'pfizer': 'PFE', 'johnson & johnson': 'JNJ', 'moderna': 'MRNA',
        'unitedhealth': 'UNH', 'eli lilly': 'LLY',
        
        # Other
        'uber': 'UBER', 'airbnb': 'ABNB', 'spotify': 'SPOT',
        'cbre': 'CBRE', 'cbre group': 'CBRE',
    }

    _ticker_lookup.update(manual_mappings)  # This overwrites any conflicts
    
    # 3. Build lowercase lookup and known set
    _ticker_lookup_lower = {k.lower(): v for k, v in _ticker_lookup.items()}
    
    # 4. FILTER: Only keep US tickers in known set
    _known_tickers_set = {v for v in _ticker_lookup.values() if '.' not in v}
    
    return _ticker_lookup, _ticker_lookup_lower, _known_tickers_set

#sector classifiction

SECTOR_KEYWORDS = {
    'Technology': ['software', 'semiconductor', 'chip', 'AI', 'cloud', 'tech', 'SaaS'],
    'Healthcare': ['pharma', 'biotech', 'drug', 'FDA', 'healthcare', 'medical'],
    'Financials': ['bank', 'financial', 'investment', 'interest rate', 'Fed', 'mortgage'],
    'Energy': ['oil', 'gas', 'petroleum', 'OPEC', 'renewable', 'solar', 'energy'],
    'Consumer': ['retail', 'e-commerce', 'shopping', 'brand', 'consumer'],
    'Real Estate': ['real estate', 'property', 'REIT', 'commercial real estate', 'housing', 'mortgage'],
    'Industrials': ['industrial', 'manufacturing', 'aerospace', 'defense', 'machinery', 'construction'],
    'Communication Services': ['social media', 'advertising', 'streaming', 'media', 'telecom', 'communication'],
}

REGION_MAPPINGS = {
    'united states': 'United States', 'usa': 'United States', 'america': 'United States', 'u.s.': 'United States',
    'china': 'China', 'chinese': 'China',
    'japan': 'Japan', 'japanese': 'Japan',
    'united kingdom': 'UK', 'britain': 'UK', 'uk': 'UK',
    'germany': 'EU', 'france': 'EU', 'europe': 'EU', 'european': 'EU',
    'federal reserve': 'United States', 'fed': 'United States', 'wall street': 'United States',
    'ecb': 'EU', 'european central bank': 'EU',
    'hong kong': 'Hong Kong', 'hk': 'Hong Kong',
    'singapore': 'Singapore', 'singaporean': 'Singapore',
    'southeast asia': 'SEA', 'asean': 'SEA',
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
    
    
    def get_phrase_matcher():
        global _phrase_matcher
        if _phrase_matcher is not None:
            return _phrase_matcher

        from spacy.matcher import PhraseMatcher
        nlp = get_spacy_nlp()
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

        _, ticker_lookup_lower, known_tickers = get_ticker_lookup()

        # group phrases by ticker so we can map match -> ticker quickly
        phrases_by_ticker: dict[str, list[str]] = {}
        for name, ticker in ticker_lookup_lower.items():
            if len(name) >= 4 and ticker in known_tickers:
                phrases_by_ticker.setdefault(ticker, []).append(name)

        # add patterns; use ticker as match_id
        for ticker, names in phrases_by_ticker.items():
            matcher.add(ticker, [nlp.make_doc(n) for n in names])

        _phrase_matcher = matcher
        return _phrase_matcher
    
    matcher = get_phrase_matcher()
    nlp = get_spacy_nlp()
    doc = nlp(text)

    for match_id, start_i, end_i in matcher(doc):
        ticker = nlp.vocab.strings[match_id]  # we stored ticker as the match_id string
        span = doc[start_i:end_i]
        pos_key = (span.start_char, ticker)
        if pos_key in seen_positions:
            continue
        seen_positions.add(pos_key)

        in_title = span.start_char < 200
        entities.append(EntityMention(
            text=span.text,
            ticker=ticker,
            entity_type="TICKER",
            start=span.start_char,
            end=span.end_char,
            relevance=0.75 if in_title else 0.55
        ))
    
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
        start = 0
        while True:
            pos = text_lower.find(keyword, start)
            if pos == -1:
                break

            entities.append(EntityMention(
                text=text[pos:pos + len(keyword)],
                ticker=region,
                entity_type="REGION",
                start=pos,
                end=pos + len(keyword),
                relevance=0.6
            ))

            start = pos + 1
    
    return entities

def get_entity_context(text: str, entity: EntityMention, all_entities: list[EntityMention] = None, window: int = 200, mask_token: str = "[COMPANY]"
) -> str:
    """
    Extract context window around entity mention
    """
    # Get initial window
    start = max(0, entity.start - window)
    end = min(len(text), entity.end + window)
    
    # Expand to sentence boundaries
    sentence_start = text.rfind('.', start, entity.start)
    if sentence_start != -1 and sentence_start > start:
        start = sentence_start + 1
    
    sentence_end = text.find('.', entity.end, end)
    if sentence_end != -1:
        end = sentence_end + 1
    
    context = text[start:end].strip()
    
    if all_entities:
        # Sort other entities by position (descending) to replace from end first
        # This prevents position shifts from affecting earlier replacements
        other_entities = [
            e for e in all_entities 
            if e.start != entity.start and e.ticker != entity.ticker
        ]
        other_entities.sort(key=lambda e: e.start, reverse=True)
        
        for other in other_entities:
            # Calculate position relative to our context window
            rel_start = other.start - start
            rel_end = other.end - start
            
            # Only mask if this entity falls within our context
            if 0 <= rel_start < len(context) and rel_end > 0:
                rel_start = max(0, rel_start)
                rel_end = min(len(context), rel_end)
                
                # Replace other entity with mask token
                context = context[:rel_start] + mask_token + context[rel_end:]
    
    return context

    
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
    
    UPDATED: Now passes all entities to context extraction for masking.
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
            key = ent.ticker  # SECTOR or REGION uses its name as key
        
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
            # === KEY CHANGE: Pass all entities for masking ===
            context = get_entity_context(
                full_text, 
                mention, 
                all_entities=entities,  # NEW: pass all entities
                window=75               # NEW: reduced window
            )
            
            # Avoid duplicate contexts
            context_key = context[:100]
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
            final_confidence *= 0.65
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
    _ = get_sentiment_analyzer()

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