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

