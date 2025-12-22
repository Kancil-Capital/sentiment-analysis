import numpy as np
import pandas as pd

def load_demo_data(n_days: int = 180, seed: int = 42):
    np.random.seed(seed)

    # -----------------------------
    # Dates
    # -----------------------------
    dates = pd.date_range(
        end=pd.Timestamp.today(),
        periods=n_days,
        freq="B"
    )

    # -----------------------------
    # Price process
    # -----------------------------
    returns = np.random.normal(0.0005, 0.015, n_days)
    price = 150 * (1 + returns).cumprod()

    # -----------------------------
    # Sentiment
    # -----------------------------
    sentiment = np.clip(
        np.random.normal(0, 0.35, n_days),
        -1,
        1
    )

    # -----------------------------
    # Volume
    # -----------------------------
    volume = (
        1e7
        + np.abs(returns) * 5e7
        + np.random.normal(0, 1e6, n_days)
    ).astype(int)

    # -----------------------------
    # Events
    # -----------------------------
    event = [None] * n_days
    event_impact = [np.nan] * n_days

    event_days = np.random.choice(
        range(n_days),
        size=6,
        replace=False
    )

    event_labels = [
        "Earnings",
        "Product Launch",
        "Guidance Update",
        "Regulatory News",
        "Analyst Upgrade",
        "Macro Shock"
    ]

    for i, label in zip(event_days, event_labels):
        event[i] = label

        # Impact score: negative to positive
        event_impact[i] = np.random.uniform(-1.0, 1.0)

    # -----------------------------
    # Assemble DataFrame
    # -----------------------------
    df = pd.DataFrame({
        "date": dates,
        "price": price,
        "return": returns,
        "sentiment": sentiment,
        "volume": volume,
        "event": event,
        "event_impact": event_impact
    })

    # -----------------------------
    # Keyword data
    # -----------------------------
    keyword_data = {
        "earnings": {"count": 120, "sentiment": -0.15},
        "iphone": {"count": 95, "sentiment": 0.25},
        "ai": {"count": 140, "sentiment": 0.45},
        "services": {"count": 80, "sentiment": 0.30},
        "china": {"count": 60, "sentiment": -0.20},
        "guidance": {"count": 70, "sentiment": -0.05},
        "revenue": {"count": 90, "sentiment": 0.10},
        "regulation": {"count": 40, "sentiment": -0.35},
        "buyback": {"count": 55, "sentiment": 0.40},
        "innovation": {"count": 65, "sentiment": 0.35},
        "vision pro": {"count": 45, "sentiment": 0.20},
        "supply chain": {"count": 50, "sentiment": -0.10},
    }


    return df, keyword_data
