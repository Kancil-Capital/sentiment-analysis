"""
SentimentPulse Dashboard - Main Application
"""
from dotenv import load_dotenv
load_dotenv(".env")

from datetime import datetime, timedelta
import pandas as pd
from dash import Dash, html, dcc, Input, Output

from app.data.main import get_articles, get_price_data
from app.figures import (
    create_price_sentiment_chart,
    create_candlestick_chart,
    create_sentiment_histogram,
    create_sentiment_breakdown_chart,
    create_lag_correlation_chart,
    create_lag_heatmap,
    create_keyword_cloud,
    COLORS
)

app = Dash(__name__, title="SentimentPulse")
server = app.server


def sort_articles(articles_df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    """Sort articles based on selected criteria."""
    if articles_df.empty:
        return articles_df

    if sort_by == 'impact_desc':
        return articles_df.assign(abs_sent=articles_df['sentiment'].abs()).sort_values('abs_sent', ascending=False).drop(columns='abs_sent')
    elif sort_by == 'impact_asc':
        return articles_df.assign(abs_sent=articles_df['sentiment'].abs()).sort_values('abs_sent', ascending=True).drop(columns='abs_sent')
    elif sort_by == 'recent':
        return articles_df.sort_values('timestamp', ascending=False)
    elif sort_by == 'oldest':
        return articles_df.sort_values('timestamp', ascending=True)
    elif sort_by == 'positive':
        return articles_df.sort_values('sentiment', ascending=False)
    elif sort_by == 'negative':
        return articles_df.sort_values('sentiment', ascending=True)
    return articles_df


def render_articles_list(articles_df: pd.DataFrame) -> list:
    """Render all articles as Dash HTML components."""
    if articles_df.empty:
        return [html.Div(
            "No articles available for selected filters",
            style={
                'textAlign': 'center',
                'color': COLORS['text_muted'],
                'padding': '20px'
            }
        )]

    articles = []
    for _, row in articles_df.iterrows():
        sentiment_color = COLORS['accent_green'] if row['sentiment'] > 0 else COLORS['accent_red'] if row['sentiment'] < 0 else COLORS['text_muted']
        sentiment_arrow = '↗' if row['sentiment'] > 0 else '↘' if row['sentiment'] < 0 else '→'

        # Format affected parties
        affected_text = ""
        if 'affected' in row and pd.notna(row['affected']):
            if isinstance(row['affected'], list):
                affected_text = f" • {', '.join(row['affected'])}"
            elif isinstance(row['affected'], str):
                affected_text = f" • {row['affected']}"

        articles.append(html.Div([
            html.Div([
                html.Span(row['source'], style={'color': COLORS['accent_cyan'], 'fontSize': '12px'}),
                html.Span(f" • {row['timestamp'].strftime('%Y-%m-%d %H:%M')}", style={'color': COLORS['text_muted'], 'fontSize': '12px'}),
                html.Span(affected_text, style={'color': COLORS['text_muted'], 'fontSize': '11px', 'fontStyle': 'italic'}),
                html.Span(
                    f" {sentiment_arrow} {row['sentiment']:.2f}",
                    style={'color': sentiment_color, 'fontSize': '12px', 'float': 'right'}
                )
            ]),
            html.A(
                row['title'],
                href=row['url'],
                target='_blank',
                style={'color': COLORS['text'], 'fontWeight': 'bold', 'textDecoration': 'none', 'display': 'block', 'marginTop': '5px'}
            ),
            html.P(
                row['body'][:200] + '...' if len(str(row['body'])) > 200 else str(row['body']),
                style={'color': COLORS['text_muted'], 'fontSize': '12px', 'marginTop': '5px', 'lineHeight': '1.4'}
            )
        ], style={
            'padding': '15px',
            'backgroundColor': COLORS['background'],
            'borderRadius': '6px',
            'marginBottom': '10px',
            'border': f"1px solid {COLORS['border']}"
        }))

    return articles


# Calculate default date range
end_date = datetime.now().date()
start_date = end_date - timedelta(days=30)
min_date = end_date - timedelta(days=365)

# Layout
app.layout = html.Div(style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '20px'}, children=[
    # Header
    html.Div(style={
        'backgroundColor': COLORS['card'],
        'padding': '20px',
        'borderRadius': '8px',
        'marginBottom': '20px',
        'border': f"1px solid {COLORS['border']}"
    }, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '15px'}, children=[
            html.H1("SentimentPulse", style={'color': COLORS['text'], 'margin': '0', 'fontSize': '28px'}),
            html.Div(style={'display': 'flex', 'gap': '15px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
                html.Div([
                    html.Label("Ticker", style={'color': COLORS['text_muted'], 'fontSize': '12px', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='ticker-dropdown',
                        options=[
                            {'label': ticker, 'value': ticker}
                            for ticker in ['ADSK', 'BABA', 'C', 'CBRE', 'GE', 'GRAB', 'META', 'NVDA']
                        ],
                        value='META',
                        style={
                            'width': '120px',
                            'backgroundColor': COLORS['background'],
                            'color': COLORS['text']
                        },
                        clearable=False
                    )
                ]),
                html.Div([
                    html.Label("Date Range", style={'color': COLORS['text_muted'], 'fontSize': '12px', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.DatePickerRange(
                        id='date-picker',
                        start_date=start_date,
                        end_date=end_date,
                        min_date_allowed=min_date,
                        max_date_allowed=end_date,
                        style={'backgroundColor': COLORS['background']}
                    )
                ])
            ])
        ])
    ]),

    # Filters Row
    html.Div(style={
        'backgroundColor': COLORS['card'],
        'padding': '20px',
        'borderRadius': '8px',
        'marginBottom': '20px',
        'border': f"1px solid {COLORS['border']}"
    }, children=[
        html.Div(style={'display': 'flex', 'gap': '30px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
            html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                html.Label("Confidence Threshold", style={'color': COLORS['text_muted'], 'fontSize': '12px', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Slider(
                    id='confidence-slider',
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    value=0.5,
                    marks={i/10: f"{i/10:.1f}" for i in range(0, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),
            html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
                html.Label("Affected Parties", style={'color': COLORS['text_muted'], 'fontSize': '12px', 'marginBottom': '5px', 'display': 'block'}),
                dcc.Checklist(
                    id='affected-parties-checklist',
                    options=[],
                    value=[],
                    inline=True,
                    style={'color': COLORS['text'], 'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'},
                    labelStyle={'marginRight': '15px', 'display': 'flex', 'alignItems': 'center'}
                )
            ])
        ])
    ]),

    # Main Content Grid
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '60% 40%', 'gap': '20px', 'marginBottom': '20px'}, children=[
        # Left Column
        html.Div(children=[
            # Price & Sentiment Overlay
            html.Div(style={
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'marginBottom': '20px',
                'border': f"1px solid {COLORS['border']}"
            }, children=[
                html.H3("Price & Sentiment", style={'color': COLORS['text'], 'fontSize': '16px', 'marginBottom': '15px'}),
                dcc.Graph(id='price-sentiment-chart', config={'displayModeBar': False})
            ]),

            # Candlestick Chart
            html.Div(style={
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'border': f"1px solid {COLORS['border']}"
            }, children=[
                html.H3("Price Action", style={'color': COLORS['text'], 'fontSize': '16px', 'marginBottom': '15px'}),
                dcc.Graph(id='candlestick-chart', config={'displayModeBar': False})
            ])
        ]),

        # Right Column
        html.Div(children=[
            # Statistics Panel
            html.Div(style={
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'marginBottom': '20px',
                'border': f"1px solid {COLORS['border']}"
            }, children=[
                html.H3("Statistics", style={'color': COLORS['text'], 'fontSize': '16px', 'marginBottom': '15px'}),
                html.Div(id='statistics-panel')
            ]),

            # News Drill-Down
            html.Div(style={
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'border': f"1px solid {COLORS['border']}"
            }, children=[
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '15px'}, children=[
                    html.H3("News Articles", style={'color': COLORS['text'], 'fontSize': '16px', 'margin': '0'}),
                    dcc.Dropdown(
                        id='sort-dropdown',
                        options=[
                            {'label': 'Most Recent', 'value': 'recent'},
                            {'label': 'Oldest', 'value': 'oldest'},
                            {'label': 'Most Impactful', 'value': 'impact_desc'},
                            {'label': 'Least Impactful', 'value': 'impact_asc'},
                            {'label': 'Most Positive', 'value': 'positive'},
                            {'label': 'Most Negative', 'value': 'negative'}
                        ],
                        value='recent',
                        style={'width': '180px', 'backgroundColor': COLORS['background']},
                        clearable=False
                    )
                ]),
                html.Div(
                    id='news-articles',
                    style={'maxHeight': '500px', 'overflowY': 'auto'}
                )
            ])
        ])
    ]),

    # Bottom Row - Sentiment Analysis
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '20px'}, children=[
        # Sentiment Breakdown
        html.Div(style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'border': f"1px solid {COLORS['border']}"
        }, children=[
            html.H3("Sentiment Breakdown", style={'color': COLORS['text'], 'fontSize': '16px', 'marginBottom': '15px'}),
         dcc.Graph(id='sentiment-breakdown-chart', config={'displayModeBar': False}, style={'height': '250px'})
        ]),

        # Sentiment Histogram
        html.Div(style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'border': f"1px solid {COLORS['border']}"
        }, children=[
            html.H3("Sentiment Distribution", style={'color': COLORS['text'], 'fontSize': '16px', 'marginBottom': '15px'}),
            dcc.Graph(id='sentiment-histogram', config={'displayModeBar': False}, style={'height': '250px'})
        ])
    ]),

    # Keyword Cloud
    html.Div(style={
        'backgroundColor': COLORS['card'],
        'padding': '20px',
        'borderRadius': '8px',
        'marginBottom': '20px',
        'border': f"1px solid {COLORS['border']}"
    }, children=[
        html.H3("Key Themes", style={'color': COLORS['text'], 'fontSize': '16px', 'marginBottom': '15px'}),
        dcc.Graph(id='keyword-cloud', config={'displayModeBar': False})
    ]),

    # Correlation Analysis
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
        # Lag Correlation Scatter
        html.Div(style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'border': f"1px solid {COLORS['border']}"
        }, children=[
            html.H3("Sentiment vs Returns", style={'color': COLORS['text'], 'fontSize': '16px', 'marginBottom': '15px'}),
            dcc.Graph(id='lag-correlation-scatter', config={'displayModeBar': False}, style={'height': '300px'})
        ]),

        # Lag Correlation Heatmap
        html.Div(style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'border': f"1px solid {COLORS['border']}"
        }, children=[
            html.H3("Lag Correlation", style={'color': COLORS['text'], 'fontSize': '16px', 'marginBottom': '15px'}),
            dcc.Graph(id='lag-correlation-heatmap', config={'displayModeBar': False}, style={'height': '150px'})
        ])
    ])
])


# Main Callback
@app.callback(
    [
        Output('price-sentiment-chart', 'figure'),
        Output('candlestick-chart', 'figure'),
        Output('sentiment-histogram', 'figure'),
        Output('sentiment-breakdown-chart', 'figure'),
        Output('lag-correlation-scatter', 'figure'),
        Output('lag-correlation-heatmap', 'figure'),
        Output('keyword-cloud', 'figure'),
        Output('statistics-panel', 'children'),
        Output('news-articles', 'children'),
        Output('affected-parties-checklist', 'options'),
        Output('affected-parties-checklist', 'value')
    ],
    [
        Input('ticker-dropdown', 'value'),
        Input('date-picker', 'start_date'),
        Input('date-picker', 'end_date'),
        Input('confidence-slider', 'value'),
        Input('affected-parties-checklist', 'value'),
        Input('sort-dropdown', 'value')
    ]
)
def update_dashboard(ticker, start_date, end_date, confidence_threshold, affected_parties, sort_by):
    """Main callback to update all dashboard components."""
    # Convert string dates to datetime objects
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date).date()
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date).date()

    # Fetch data
    price_df = get_price_data(ticker, start_date, end_date)
    articles_df = get_articles(ticker, start_date, end_date)

    # Filter by confidence
    if not articles_df.empty:
        articles_df = articles_df[articles_df['confidence'] >= confidence_threshold]

    # Handle affected parties filter
    affected_parties_options = []
    if not articles_df.empty and 'affected' in articles_df.columns:
        # Extract unique affected parties
        all_parties = set()
        for parties in articles_df['affected'].dropna():
            if isinstance(parties, list):
                all_parties.update(parties)
            elif isinstance(parties, str):
                all_parties.add(parties)
        affected_parties_options = [{'label': party, 'value': party} for party in sorted(all_parties)]

        # Filter by affected parties if selected
        if affected_parties and len(affected_parties) > 0:
            articles_df = articles_df[
                articles_df['affected'].apply(
                    lambda x: (isinstance(x, list) and any(party in x for party in affected_parties)) or
                              (isinstance(x, str) and x in affected_parties)
                )
            ]
    else:
        # If no affected column, set all as selected
        affected_parties = []

    # Compute daily sentiment aggregation
    daily_sentiment_df = pd.DataFrame()
    if not articles_df.empty:
        daily_sentiment_df = articles_df.groupby(articles_df['timestamp'].dt.date).agg({
            'sentiment': 'mean'
        }).reset_index()
        daily_sentiment_df.columns = ['date', 'sentiment']

    # Merge price and sentiment on date
    combined_df = pd.DataFrame()
    if not price_df.empty and not daily_sentiment_df.empty:
        price_df_daily = price_df.copy()
        price_df_daily['date'] = price_df_daily['timestamp'].dt.date
        combined_df = pd.merge(
            price_df_daily[['date', 'close']],
            daily_sentiment_df,
            on='date',
            how='inner'
        )

    # Generate figures
    price_sentiment_fig = create_price_sentiment_chart(price_df, daily_sentiment_df)
    candlestick_fig = create_candlestick_chart(price_df)
    sentiment_histogram_fig = create_sentiment_histogram(articles_df)
    sentiment_breakdown_fig = create_sentiment_breakdown_chart(articles_df)
    lag_correlation_scatter_fig = create_lag_correlation_chart(combined_df)
    lag_correlation_heatmap_fig = create_lag_heatmap(combined_df)
    keyword_cloud_fig = create_keyword_cloud(articles_df)

    # Generate statistics panel
    statistics_panel = generate_statistics_panel(daily_sentiment_df, articles_df, combined_df)

    # Sort and render articles
    sorted_articles = sort_articles(articles_df, sort_by)
    news_articles = render_articles_list(sorted_articles)

    # Set default affected parties value if not set
    if not affected_parties_options:
        affected_parties_value = []
    elif affected_parties is None:
        affected_parties_value = [opt['value'] for opt in affected_parties_options]
    else:
        affected_parties_value = affected_parties

    return (
        price_sentiment_fig,
        candlestick_fig,
        sentiment_histogram_fig,
        sentiment_breakdown_fig,
        lag_correlation_scatter_fig,
        lag_correlation_heatmap_fig,
        keyword_cloud_fig,
        statistics_panel,
        news_articles,
        affected_parties_options,
        affected_parties_value
    )


def generate_statistics_panel(daily_sentiment_df: pd.DataFrame, articles_df: pd.DataFrame, combined_df: pd.DataFrame) -> list:
    """Generate the statistics panel with key metrics."""
    if daily_sentiment_df.empty and articles_df.empty:
        return [html.Div("No data available", style={'color': COLORS['text_muted'], 'textAlign': 'center'})]

    metrics = []

    # Daily Mean Sentiment
    if not daily_sentiment_df.empty:
        mean_val = daily_sentiment_df['sentiment'].mean()
        mean_color = COLORS['accent_green'] if mean_val > 0 else COLORS['accent_red'] if mean_val < 0 else COLORS['text']
        mean_sign = '+' if mean_val > 0 else ''
        metrics.append(create_stat_card("Daily Mean", f"{mean_sign}{mean_val:.3f}", mean_color))
    else:
        metrics.append(create_stat_card("Daily Mean", "N/A", COLORS['text_muted']))

    # Daily Median Sentiment
    if not daily_sentiment_df.empty:
        median_val = daily_sentiment_df['sentiment'].median()
        median_color = COLORS['accent_green'] if median_val > 0 else COLORS['accent_red'] if median_val < 0 else COLORS['text']
        median_sign = '+' if median_val > 0 else ''
        metrics.append(create_stat_card("Daily Median", f"{median_sign}{median_val:.3f}", median_color))
    else:
        metrics.append(create_stat_card("Daily Median", "N/A", COLORS['text_muted']))

    # Net Sentiment
    if not daily_sentiment_df.empty:
        net_val = daily_sentiment_df['sentiment'].sum()
        net_color = COLORS['accent_green'] if net_val > 0 else COLORS['accent_red'] if net_val < 0 else COLORS['text']
        net_sign = '+' if net_val > 0 else ''
        metrics.append(create_stat_card("Net Sentiment", f"{net_sign}{net_val:.1f}", net_color))
    else:
        metrics.append(create_stat_card("Net Sentiment", "N/A", COLORS['text_muted']))

    # Best Lag Correlation
    if not combined_df.empty and len(combined_df) >= 2:
        max_corr = -999
        best_lag = 0
        for lag in range(0, 8):
            shifted_return = combined_df['close'].pct_change().shift(-lag)
            valid_data = combined_df[['sentiment']].join(shifted_return.rename('return')).dropna()
            if len(valid_data) >= 2:
                corr = abs(valid_data['sentiment'].corr(valid_data['return']))
                if corr > max_corr:
                    max_corr = corr
                    best_lag = lag
        if max_corr > -999:
            metrics.append(create_stat_card("Best Lag Corr", f"{max_corr:.2f} @ {best_lag}d", COLORS['accent_cyan']))
        else:
            metrics.append(create_stat_card("Best Lag Corr", "N/A", COLORS['text_muted']))
    else:
        metrics.append(create_stat_card("Best Lag Corr", "N/A", COLORS['text_muted']))

    # Total Articles
    total_articles = len(articles_df) if not articles_df.empty else 0
    metrics.append(create_stat_card("Total Articles", str(total_articles), COLORS['text']))

    # Positive/Neutral/Negative Ratio
    if not articles_df.empty:
        positive_count = len(articles_df[articles_df['sentiment'] > 0.2])
        neutral_count = len(articles_df[(articles_df['sentiment'] >= -0.2) & (articles_df['sentiment'] <= 0.2)])
        negative_count = len(articles_df[articles_df['sentiment'] < -0.2])
        total = positive_count + neutral_count + negative_count

        if total > 0:
            ratio_bar = html.Div(style={'marginTop': '10px', 'height': '8px', 'borderRadius': '4px', 'overflow': 'hidden', 'display': 'flex'}, children=[
                html.Div(style={'width': f"{(positive_count/total)*100}%", 'backgroundColor': COLORS['accent_green'], 'height': '100%'}),
                html.Div(style={'width': f"{(neutral_count/total)*100}%", 'backgroundColor': COLORS['text_muted'], 'height': '100%'}),
                html.Div(style={'width': f"{(negative_count/total)*100}%", 'backgroundColor': COLORS['accent_red'], 'height': '100%'})
            ])
            ratio_text = html.Div(
                f"{positive_count} / {neutral_count} / {negative_count}",
                style={'fontSize': '10px', 'color': COLORS['text_muted'], 'marginTop': '5px'}
            )
            metrics.append(html.Div([
                html.Div("Sentiment Ratio", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '5px'}),
                ratio_bar,
                ratio_text
            ], style={
                'padding': '12px',
                'backgroundColor': COLORS['background'],
                'borderRadius': '6px',
                'border': f"1px solid {COLORS['border']}"
            }))

    return html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '10px'}, children=metrics)


def create_stat_card(label: str, value: str, color: str) -> html.Div:
    """Create a styled statistics card."""
    return html.Div([
        html.Div(label, style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '5px'}),
        html.Div(value, style={'color': color, 'fontSize': '20px', 'fontWeight': 'bold'})
    ], style={
        'padding': '12px',
        'backgroundColor': COLORS['background'],
        'borderRadius': '6px',
        'border': f"1px solid {COLORS['border']}"
    })


if __name__ == '__main__':
    app.run(debug=True, port=8050)
