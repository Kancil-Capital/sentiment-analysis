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

