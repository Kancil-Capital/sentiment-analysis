"""
Figure generation functions for SentimentPulse dashboard.
Each function takes DataFrames and returns a plotly Figure.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image

COLORS = {
    'background': '#0d1117',
    'card': '#161b22',
    'border': '#30363d',
    'text': '#ffffff',
    'text_muted': '#888888',
    'accent_cyan': '#00d4ff',
    'accent_green': '#00ff88',
    'accent_red': '#ff4444',
}


def create_price_sentiment_chart(price_df: pd.DataFrame, daily_sentiment_df: pd.DataFrame) -> go.Figure:
    """
    Dual-axis time series showing price and aggregated daily sentiment.
    """
    if price_df.empty or daily_sentiment_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available for selected range", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text']),
            height=350
        )
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: Closing price
    fig.add_trace(
        go.Scatter(
            x=price_df['timestamp'],
            y=price_df['close'],
            name='Price',
            line=dict(color=COLORS['accent_cyan'], width=2),
            mode='lines'
        ),
        secondary_y=False
    )

    # Trace 2: Daily mean sentiment
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment_df['date'],
            y=daily_sentiment_df['sentiment'],
            name='Sentiment',
            line=dict(color=COLORS['accent_green'], width=2, dash='dot'),
            mode='lines'
        ),
        secondary_y=True
    )

    # Add markers for significant sentiment events
    significant = daily_sentiment_df[daily_sentiment_df['sentiment'].abs() > 0.7]
    if not significant.empty:
        fig.add_trace(
            go.Scatter(
                x=significant['date'],
                y=significant['sentiment'],
                name='Significant Events',
                mode='markers',
                marker=dict(size=8, color=COLORS['accent_green']),
                hovertemplate='<b>Date:</b> %{x}<br><b>Sentiment:</b> %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            secondary_y=True
        )

    # Layout
    fig.update_xaxes(gridcolor=COLORS['border'], showgrid=True)
    fig.update_yaxes(title_text="Price ($)", gridcolor=COLORS['border'], showgrid=True, secondary_y=False)
    fig.update_yaxes(title_text="Sentiment", gridcolor=COLORS['border'], showgrid=True, secondary_y=True)

    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=30, b=40),
        height=350,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


def create_candlestick_chart(price_df: pd.DataFrame) -> go.Figure:
    """
    Standard OHLC view with volume bars.
    """
    if price_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available for selected range", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text']),
            height=400
        )
        return fig

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.02
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=price_df['timestamp'],
            open=price_df['open'],
            high=price_df['high'],
            low=price_df['low'],
            close=price_df['close'],
            name='OHLC',
            increasing=dict(line=dict(color=COLORS['accent_green'])),
            decreasing=dict(line=dict(color=COLORS['accent_red']))
        ),
        row=1, col=1
    )

    # Volume bars (colored by price direction)
    colors = [COLORS['accent_green'] if close > open_ else COLORS['accent_red']
              for close, open_ in zip(price_df['close'], price_df['open'])]

    fig.add_trace(
        go.Bar(
            x=price_df['timestamp'],
            y=price_df['volume'],
            name='Volume',
            marker=dict(color=colors),
            showlegend=False
        ),
        row=2, col=1
    )

    # Layout
    fig.update_xaxes(gridcolor=COLORS['border'], showgrid=True, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(gridcolor=COLORS['border'], showgrid=True, row=1, col=1)
    fig.update_yaxes(title_text="Volume", gridcolor=COLORS['border'], showgrid=True, row=2, col=1)

    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=30, b=40),
        height=400,
        xaxis_rangeslider_visible=False
    )

    return fig


def create_sentiment_histogram(articles_df: pd.DataFrame) -> go.Figure:
    """
    Show distribution of sentiment scores.
    """
    if articles_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available for selected range", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text']),
            height=250
        )
        return fig

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=articles_df['sentiment'],
            nbinsx=20,
            marker=dict(
                color=COLORS['accent_cyan'],
                line=dict(color=COLORS['border'], width=1)
            ),
            name='Sentiment Distribution'
        )
    )

    # Add vertical lines for mean and median
    mean_val = articles_df['sentiment'].mean()
    median_val = articles_df['sentiment'].median()

    fig.add_vline(x=mean_val, line_dash="dash", line_color=COLORS['accent_green'],
                  annotation_text=f"Mean: {mean_val:.3f}", annotation_position="top left")
    fig.add_vline(x=median_val, line_dash="dot", line_color=COLORS['text_muted'],
                  annotation_text=f"Median: {median_val:.3f}", annotation_position="top right")

    # Layout
    fig.update_xaxes(title_text="Sentiment Score", gridcolor=COLORS['border'], showgrid=True, range=[-1, 1])
    fig.update_yaxes(title_text="Article Count", gridcolor=COLORS['border'], showgrid=True)

    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=30, b=40),
        height=250,
        showlegend=False
    )

    return fig


def create_sentiment_breakdown_chart(articles_df: pd.DataFrame) -> go.Figure:
    """
    Show daily proportion of positive/neutral/negative articles.
    """
    if articles_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available for selected range", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text']),
            height=250
        )
        return fig

    # Categorize sentiment
    articles_df = articles_df.copy()
    articles_df['sentiment_label'] = pd.cut(
        articles_df['sentiment'],
        bins=[-1, -0.2, 0.2, 1],
        labels=['Negative', 'Neutral', 'Positive']
    )

    # Group by date and sentiment label
    daily_breakdown = articles_df.groupby([
        articles_df['timestamp'].dt.date,
        'sentiment_label'
    ]).size().unstack(fill_value=0)

    fig = go.Figure()

    # Add traces for each sentiment category
    if 'Negative' in daily_breakdown.columns:
        fig.add_trace(go.Bar(
            x=daily_breakdown.index,
            y=daily_breakdown['Negative'],
            name='Negative',
            marker=dict(color=COLORS['accent_red'])
        ))

    if 'Neutral' in daily_breakdown.columns:
        fig.add_trace(go.Bar(
            x=daily_breakdown.index,
            y=daily_breakdown['Neutral'],
            name='Neutral',
            marker=dict(color=COLORS['text_muted'])
        ))

    if 'Positive' in daily_breakdown.columns:
        fig.add_trace(go.Bar(
            x=daily_breakdown.index,
            y=daily_breakdown['Positive'],
            name='Positive',
            marker=dict(color=COLORS['accent_green'])
        ))

    # Layout
    fig.update_xaxes(title_text="Date", gridcolor=COLORS['border'], showgrid=True)
    fig.update_yaxes(title_text="Article Count", gridcolor=COLORS['border'], showgrid=True)

    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=30, b=40),
        height=250,
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


def create_lag_correlation_chart(combined_df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of sentiment vs next-day returns.
    """
    if combined_df.empty or 'sentiment' not in combined_df.columns or 'close' not in combined_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available for selected range", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text']),
            height=300
        )
        return fig

    # Calculate next-day return
    combined_df = combined_df.copy()
    combined_df['next_day_return'] = combined_df['close'].pct_change().shift(-1) * 100

    # Drop NaN values
    scatter_df = combined_df[['sentiment', 'next_day_return']].dropna()

    if scatter_df.empty or len(scatter_df) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for correlation analysis", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text']),
            height=300
        )
        return fig

    fig = go.Figure()

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=scatter_df['sentiment'],
        y=scatter_df['next_day_return'],
        mode='markers',
        marker=dict(color=COLORS['accent_cyan'], size=10, opacity=0.6),
        name='Data Points',
        showlegend=False
    ))

    # Add trendline
    if len(scatter_df) >= 2:
        z = np.polyfit(scatter_df['sentiment'], scatter_df['next_day_return'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(scatter_df['sentiment'].min(), scatter_df['sentiment'].max(), 100)
        y_line = p(x_line)

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color=COLORS['accent_green'], width=2, dash='dash'),
            name='Trendline',
            showlegend=False
        ))

        # Calculate R² and correlation
        correlation_matrix = np.corrcoef(scatter_df['sentiment'], scatter_df['next_day_return'])
        corr = correlation_matrix[0, 1]
        r2 = corr ** 2

        # Add annotation
        fig.add_annotation(
            text=f"R² = {r2:.3f}, ρ = {corr:.3f}",
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor=COLORS['card'],
            bordercolor=COLORS['border'],
            borderwidth=1,
            font=dict(size=12, color=COLORS['text'])
        )

    # Layout
    fig.update_xaxes(title_text="Daily Sentiment", gridcolor=COLORS['border'], showgrid=True)
    fig.update_yaxes(title_text="Next-Day Return (%)", gridcolor=COLORS['border'], showgrid=True)

    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=30, b=40),
        height=300
    )

    return fig


def create_lag_heatmap(combined_df: pd.DataFrame, max_lag: int = 7) -> go.Figure:
    """
    Show correlation strength across multiple lag periods as a bar chart.
    """
    if combined_df.empty or 'sentiment' not in combined_df.columns or 'close' not in combined_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available for selected range", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text']),
            height=250
        )
        return fig

    # Calculate correlations for each lag
    correlations = []
    for lag in range(0, max_lag + 1):
        shifted_return = combined_df['close'].pct_change().shift(-lag)
        valid_data = combined_df[['sentiment']].join(shifted_return.rename('return')).dropna()

        if len(valid_data) >= 2:
            corr = valid_data['sentiment'].corr(valid_data['return'])
        else:
            corr = 0

        correlations.append({'lag': lag, 'correlation': corr})

    lag_df = pd.DataFrame(correlations)

    # Color bars by positive/negative correlation
    colors = [COLORS['accent_green'] if corr > 0 else COLORS['accent_red'] if corr < 0 else COLORS['text_muted']
              for corr in lag_df['correlation']]

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=lag_df['lag'],
        y=lag_df['correlation'],
        marker=dict(
            color=colors,
            line=dict(color=COLORS['border'], width=1)
        ),
        text=[f"{val:.3f}" for val in lag_df['correlation']],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='<b>Lag:</b> %{x} days<br><b>Correlation:</b> %{y:.3f}<extra></extra>',
        showlegend=False
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['text_muted'], line_width=1)

    # Layout
    fig.update_xaxes(
        title_text="Lag (Days)",
        gridcolor=COLORS['border'],
        showgrid=True,
        dtick=1
    )
    fig.update_yaxes(
        title_text="Correlation",
        gridcolor=COLORS['border'],
        showgrid=True,
        range=[min(lag_df['correlation'].min() - 0.1, -0.2), max(lag_df['correlation'].max() + 0.1, 0.2)]
    )

    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=40, b=40),
        height=250
    )

    return fig


def preprocess_text(articles_df: pd.DataFrame) -> dict:
    """Preprocess article text and return word frequencies."""
    if articles_df.empty:
        return {}

    # Combine all text
    all_text = ' '.join(articles_df['title'].fillna('').tolist() + articles_df['body'].fillna('').tolist())

    # Lowercase and remove non-alpha characters
    text = all_text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()

    # Setup nltk tools
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        # If stopwords not downloaded, return empty
        return {}

    lemmatizer = WordNetLemmatizer()

    # Additional domain-specific stopwords
    financial_stopwords = {
        'said', 'says', 'also', 'would', 'could', 'one', 'two', 'new',
        'year', 'years', 'company', 'companies', 'stock', 'market',
        'shares', 'percent', 'quarter', 'reuters', 'bloomberg'
    }
    stop_words.update(financial_stopwords)

    # Filter and lemmatize
    processed_words = []
    for word in words:
        if word not in stop_words and len(word) > 2:
            lemma = lemmatizer.lemmatize(word)
            if lemma not in stop_words:
                processed_words.append(lemma)

    # Count frequencies
    return dict(Counter(processed_words))


def create_keyword_cloud(articles_df: pd.DataFrame) -> go.Figure:
    """Generate word cloud as Plotly figure."""
    word_freq = preprocess_text(articles_df)

    if not word_freq:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No keywords available",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS['text_muted'])
        )
        fig.update_layout(
            plot_bgcolor=COLORS['card'],
            paper_bgcolor=COLORS['card'],
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    # Generate word cloud image
    wc = WordCloud(
        width=1200,
        height=500,
        background_color=COLORS['card'],
        colormap='cool',
        max_words=50,
        min_font_size=10,
        max_font_size=80
    ).generate_from_frequencies(word_freq)

    # Convert to image
    img = wc.to_image()

    # Convert PIL image to plotly format
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing="stretch",
            layer="above"
        )
    )

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        plot_bgcolor=COLORS['card'],
        paper_bgcolor=COLORS['card'],
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )

    return fig
