import plotly.express as px
import pandas as pd

def create_example_figure():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=7),
        "Positive": [5, 6, 8, 7, 9, 10, 12],
        "Neutral": [4, 5, 4, 6, 5, 4, 3],
        "Negative": [3, 2, 2, 3, 1, 2, 1]
    })

    df_melted = df.melt(
        id_vars="Date",
        value_vars=["Positive", "Neutral", "Negative"],
        var_name="Sentiment",
        value_name="Count"
    )

    fig = px.line(
        df_melted,
        x="Date",
        y="Count",
        color="Sentiment",
        title="Financial News Sentiment Over Time"
    )

    return fig
