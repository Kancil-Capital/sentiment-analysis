# app/viz/figures.py - PROFESSIONAL VERSION
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Professional dark theme color palette
THEME = {
    'primary': '#6366f1',
    'secondary': '#10b981', 
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'text': '#f0f6fc',
    'grid': 'rgba(71, 85, 105, 0.3)',
    'bg': '#161b22',
    'paper_bg': '#0a0e1a'
}

def create_price_sentiment_figure(df):
    """Create the main dual-axis chart with price and sentiment"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Price line (primary axis)
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['price'],
            mode='lines',
            name='Price',
            line=dict(color='#3b82f6', width=2.5),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Price: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1, secondary_y=False
    )
    
    # Sentiment line (secondary axis)
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['sentiment'],
            mode='lines',
            name='Sentiment',
            line=dict(color='#8b5cf6', width=2),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Sentiment: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Event markers
    event_df = df[df['event'].notna()]
    if len(event_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=event_df['date'],
                y=event_df['price'],
                mode='markers',
                name='Events',
                marker=dict(
                    size=12,
                    color=event_df['event_impact'],
                    colorscale=[[0, '#ef4444'], [0.5, '#6b7280'], [1, '#10b981']],
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                text=event_df['event'],
                hovertemplate='<b>%{text}</b><br>Date: %{x|%b %d}<br>Price: $%{y:.2f}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1, secondary_y=False
        )
    
    # Volume bars with color coding
    colors = [THEME['danger'] if ret < 0 else THEME['secondary'] for ret in df['return']]
    
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False,
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Volume: %{y:,.0f}<extra></extra>',
            opacity=0.6
        ),
        row=2, col=1
    )
    
    # Layout configuration
    fig.update_layout(
        height=500,
        autosize=False,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=THEME['text']),
            bgcolor='rgba(30, 41, 59, 0.8)'
        ),
        template='plotly_dark',
        plot_bgcolor=THEME['bg'],
        paper_bgcolor=THEME['paper_bg'],
        font=dict(color=THEME['text'], family='Inter, sans-serif'),
        margin=dict(l=60, r=60, t=30, b=60)
    )
    
    # Primary Y-axis (Price)
    fig.update_yaxes(
        title_text="Price ($)",
        row=1, col=1,
        secondary_y=False,
        gridcolor=THEME['grid'],
        showgrid=True,
        title_font=dict(size=12, color=THEME['text'])
    )
    
    # Secondary Y-axis (Sentiment)
    fig.update_yaxes(
        title_text="Sentiment",
        row=1, col=1,
        secondary_y=True,
        range=[-1, 1],
        gridcolor='rgba(139, 92, 246, 0.1)',
        showgrid=False,
        title_font=dict(size=12, color=THEME['text'])
    )
    
    # Volume Y-axis
    fig.update_yaxes(
        title_text="Volume",
        row=2, col=1,
        gridcolor=THEME['grid'],
        title_font=dict(size=12, color=THEME['text'])
    )
    
    # X-axes
    fig.update_xaxes(
        gridcolor=THEME['grid'],
        showgrid=True
    )
    
    return fig

def create_lag_scatter(df, lag=1):
    """Create sentiment vs future return scatter plot"""
    lagged_df = df.copy()
    lagged_df["future_return"] = lagged_df["return"].shift(-lag)
    lagged_df = lagged_df.dropna()
    
    if len(lagged_df) == 0:
        lagged_df = df.copy()
        lagged_df["future_return"] = lagged_df["return"]
    
    # Calculate regression line
    z = np.polyfit(lagged_df["sentiment"], lagged_df["future_return"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(lagged_df["sentiment"].min(), lagged_df["sentiment"].max(), 100)
    
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=lagged_df["sentiment"],
            y=lagged_df["future_return"],
            mode="markers",
            marker=dict(
                size=10,
                color=lagged_df["future_return"],
                colorscale=[[0, THEME['danger']], [0.5, '#6b7280'], [1, THEME['secondary']]],
                showscale=True,
                colorbar=dict(
                    title=dict(text="Return", side="right"),
                    tickformat=".1%",
                    x=1.12
                ),
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
                opacity=0.8
            ),
            name=f"Lag {lag}d",
            hovertemplate='<b>Sentiment:</b> %{x:.3f}<br><b>Return:</b> %{y:.2%}<extra></extra>'
        )
    )
    
    # Regression line
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=p(x_range),
            mode="lines",
            line=dict(color=THEME['primary'], width=3, dash="dash"),
            name="Trend Line",
            hoverinfo='skip'
        )
    )
    
    # Reference lines
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    
    # Statistics
    correlation = lagged_df["sentiment"].corr(lagged_df["future_return"])
    n = len(lagged_df)
    
    fig.update_layout(
        title=dict(
            text=f"Sentiment → {lag}-Day Return Correlation",
            font=dict(size=16, color=THEME['text'])
        ),
        xaxis_title="Sentiment Score",
        yaxis_title=f"{lag}-Day Forward Return",
        template="plotly_dark",
        height=350,
        autosize=False,
        plot_bgcolor=THEME['bg'],
        paper_bgcolor=THEME['paper_bg'],
        font=dict(color=THEME['text'], family='Inter, sans-serif'),
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"<b>ρ = {correlation:.3f}</b><br>n = {n}",
                showarrow=False,
                bgcolor="rgba(30, 41, 59, 0.9)",
                bordercolor=THEME['primary'],
                borderwidth=2,
                borderpad=8,
                font=dict(size=12, color=THEME['text'])
            )
        ]
    )
    
    fig.update_xaxes(gridcolor=THEME['grid'], showgrid=True)
    fig.update_yaxes(gridcolor=THEME['grid'], showgrid=True, tickformat=".1%")
    
    return fig

def create_word_cloud(keywords):
    """Create a word cloud visualization using bubble chart"""
    if not keywords:
        keywords = {'placeholder': {'count': 1, 'sentiment': 0}}
    
    words = list(keywords.keys())
    counts = [keywords[w]['count'] for w in words]
    sentiments = [keywords[w]['sentiment'] for w in words]
    
    df_words = pd.DataFrame({
        'word': words,
        'count': counts,
        'sentiment': sentiments
    })
    
    df_words = df_words.sort_values('count', ascending=False).head(30)
    
    fig = go.Figure()
    
    # Create bubble chart with colors based on sentiment
    fig.add_trace(go.Scatter(
        x=np.random.uniform(-1, 1, len(df_words)),
        y=np.random.uniform(-1, 1, len(df_words)),
        mode='text+markers',
        text=df_words['word'],
        textfont=dict(
            size=df_words['count'] * 0.8 + 10,
            color=[THEME['secondary'] if s > 0 else THEME['danger'] if s < 0 else '#6b7280' 
                   for s in df_words['sentiment']]
        ),
        marker=dict(
            size=df_words['count'] * 3,
            color=df_words['sentiment'],
            colorscale=[[0, THEME['danger']], [0.5, '#6b7280'], [1, THEME['secondary']]],
            opacity=0.3,
            line=dict(width=0)
        ),
        hovertext=[f"<b>{row['word']}</b><br>Mentions: {row['count']}<br>Sentiment: {row['sentiment']:.2f}" 
                  for _, row in df_words.iterrows()],
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(
            text="Top Keywords by Frequency",
            font=dict(size=16, color=THEME['text'])
        ),
        template="plotly_dark",
        height=400,
        autosize=False,
        plot_bgcolor=THEME['bg'],
        paper_bgcolor=THEME['paper_bg'],
        font=dict(color=THEME['text'], family='Inter, sans-serif'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_sentiment_distribution(df):
    """Create sentiment distribution histogram"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['sentiment'],
        nbinsx=40,
        marker=dict(
            color=THEME['primary'],
            line=dict(color='white', width=0.5)
        ),
        opacity=0.7,
        name='Sentiment Distribution',
        hovertemplate='<b>Sentiment:</b> %{x:.2f}<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    # Statistics
    mean_val = df['sentiment'].mean()
    median_val = df['sentiment'].median()
    
    # Mean line
    fig.add_vline(
        x=mean_val,
        line_width=2,
        line_dash="dash",
        line_color=THEME['warning'],
        annotation_text=f"μ = {mean_val:.3f}",
        annotation_position="top"
    )
    
    # Median line
    fig.add_vline(
        x=median_val,
        line_width=2,
        line_dash="dot",
        line_color=THEME['secondary'],
        annotation_text=f"M = {median_val:.3f}",
        annotation_position="bottom"
    )
    
    fig.update_layout(
        title=dict(
            text="Sentiment Distribution",
            font=dict(size=14, color=THEME['text'])
        ),
        xaxis_title="Sentiment Score",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=180,
        autosize=False,
        plot_bgcolor=THEME['bg'],
        paper_bgcolor=THEME['paper_bg'],
        font=dict(color=THEME['text'], family='Inter, sans-serif'),
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor=THEME['grid'], showgrid=True, range=[-1, 1])
    fig.update_yaxes(gridcolor=THEME['grid'], showgrid=True)
    
    return fig
