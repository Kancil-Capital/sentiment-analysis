import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd

# IMPORT MOCK DATA LOADER 
from app.viz.data_loader import generate_sentiment_data, generate_keywords, generate_news_articles

app = dash.Dash(__name__, title="Sentiment Pulse")
server = app.server

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Div([
                html.Div('📈', style={'fontSize': '24px', 'marginRight': '10px'}),
                html.Div([
                    html.Div([
                        html.Span('Sentiment', style={'color': '#fff', 'fontWeight': 'bold'}),
                        html.Span('Pulse', style={'color': '#00d4ff', 'fontWeight': 'bold'})
                    ]),
                    html.Div('Investment Sentiment Analysis', 
                            style={'fontSize': '12px', 'color': '#888'})
                ])
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ], style={'flex': '1'}),
        
        html.Div([
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[
                    {'label': ticker, 'value': ticker} 
                    for ticker in ['ADSK', 'BABA', 'C', 'CBRE', 'GE', 'GRAB', 'META', 'NVDA']
                ],
                value='META',
                style={
                    'width': '150px',
                    'marginRight': '20px',
                    'backgroundColor': "#dce0e7",
                    'color': '#333'
                }
            ),
            html.Div([
                html.Span('↗ +2.4%', style={'color': '#00ff88', 'marginRight': '10px'}),
                html.Span('24h', style={'color': '#888', 'marginRight': '20px'}),
                html.Span('📊 0.68', style={'color': '#888', 'marginRight': '10px'}),
                html.Span('Sentiment', style={'color': '#888', 'marginRight': '20px'}),
                html.Span('🟢 Live', style={'color': '#00ff88'})
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'padding': '20px',
        'backgroundColor': '#0d1117',
        'borderBottom': '1px solid #30363d'
    }),
    
    # Main content
    html.Div([
        # Left column - Charts
        html.Div([
            # Price & Sentiment Overlay
            html.Div([
                html.H3('Price & Sentiment Overlay', 
                        style={'color': '#fff', 'fontSize': '18px', 'marginBottom': '5px'}),
                html.P('Dual-axis visualization with event annotations',
                      style={'color': '#888', 'fontSize': '12px', 'marginBottom': '15px'}),
                dcc.Graph(id='price-sentiment-chart', config={'displayModeBar': False})
            ], style={
                'backgroundColor': '#161b22',
                'padding': '20px',
                'borderRadius': '8px',
                'marginBottom': '20px'
            }),
            
            # Keyword Explorer
            html.Div([
                html.Div([
                    html.H3('💬 Keyword Explorer', 
                            style={'color': '#fff', 'fontSize': '18px'}),
                    html.Div([
                        html.Button('All', id='filter-all', n_clicks=0,
                                   style={'marginRight': '10px', 'padding': '5px 15px',
                                          'backgroundColor': '#00d4ff', 'border': 'none',
                                          'borderRadius': '15px', 'color': '#000', 'cursor': 'pointer'}),
                        html.Button('Bullish', id='filter-bullish', n_clicks=0,
                                   style={'marginRight': '10px', 'padding': '5px 15px',
                                          'backgroundColor': '#30363d', 'border': 'none',
                                          'borderRadius': '15px', 'color': '#fff', 'cursor': 'pointer'}),
                        html.Button('Bearish', id='filter-bearish', n_clicks=0,
                                   style={'padding': '5px 15px', 'backgroundColor': '#30363d',
                                          'border': 'none', 'borderRadius': '15px',
                                          'color': '#fff', 'cursor': 'pointer'})
                    ])
                ], style={'display': 'flex', 'justifyContent': 'space-between', 
                          'alignItems': 'center', 'marginBottom': '20px'}),
                
                html.Div(id='keyword-cloud', style={'minHeight': '200px'})
            ], style={
                'backgroundColor': '#161b22',
                'padding': '20px',
                'borderRadius': '8px',
                'marginBottom': '20px'
            }),
            
            # Lag Correlation Analysis
            html.Div([
                html.H3('Lag Correlation Analysis',
                        style={'color': '#fff', 'fontSize': '18px', 'marginBottom': '5px'}),
                html.P('Sentiment ↔ Next-day return relationship',
                      style={'color': '#888', 'fontSize': '12px', 'marginBottom': '15px'}),
                dcc.Graph(id='correlation-chart', config={'displayModeBar': False})
            ], style={
                'backgroundColor': '#161b22',
                'padding': '20px',
                'borderRadius': '8px'
            })
        ], style={'flex': '2', 'marginRight': '20px'}),
        
        # Right column - Stats and News
        html.Div([
            # Aggregated Sentiment
            html.Div([
                html.H3('Aggregated Sentiment',
                        style={'color': '#fff', 'fontSize': '18px', 'marginBottom': '5px'}),
                html.P('Daily statistics & filters',
                      style={'color': '#888', 'fontSize': '12px', 'marginBottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.Div('📈 DAILY MEAN', style={'color': '#888', 'fontSize': '11px'}),
                        html.Div('+0.170', id='daily-mean',
                                style={'color': '#00ff88', 'fontSize': '24px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '15px'}),
                    
                    html.Div([
                        html.Div('📊 DAILY MEDIAN', style={'color': '#888', 'fontSize': '11px'}),
                        html.Div('+0.131', id='daily-median',
                                style={'color': '#00ff88', 'fontSize': '24px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '15px'}),
                    
                    html.Div([
                        html.Div('📈 NET SENTIMENT', style={'color': '#888', 'fontSize': '11px'}),
                        html.Div('+15.3', id='net-sentiment',
                                style={'color': '#00ff88', 'fontSize': '24px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '15px'}),
                    
                    html.Div([
                        html.Div('🔗 CORRELATION', style={'color': '#888', 'fontSize': '11px'}),
                        html.Div('0.42 1d lag', id='correlation',
                                style={'color': '#fff', 'fontSize': '20px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '20px'}),
                    
                    html.Hr(style={'borderColor': '#30363d', 'margin': '20px 0'}),
                    
                    html.Div([
                        html.Div('⚡ Confidence Filter', 
                                style={'color': '#fff', 'fontSize': '14px', 'marginBottom': '10px'}),
                        html.Div('0.00', style={'color': '#00d4ff', 'fontSize': '20px', 'marginBottom': '10px'}),
                        dcc.Slider(0, 1, 0.1, value=0, id='confidence-slider',
                                  marks=None, tooltip={"placement": "bottom", "always_visible": False}),
                        html.P('Filter data points with sentiment magnitude below threshold',
                              style={'color': '#666', 'fontSize': '11px', 'marginTop': '10px'})
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Div('Total Articles', style={'color': '#888', 'fontSize': '12px'}),
                        html.Div('150', id='total-articles',
                                style={'color': '#fff', 'fontSize': '24px', 'fontWeight': 'bold'}),
                        html.Div([
                            html.Div(style={'width': '30%', 'height': '8px', 
                                          'backgroundColor': '#ff4444', 'borderRadius': '4px 0 0 4px'}),
                            html.Div(style={'width': '20%', 'height': '8px', 
                                          'backgroundColor': '#888'}),
                            html.Div(style={'width': '50%', 'height': '8px', 
                                          'backgroundColor': '#00ff88', 'borderRadius': '0 4px 4px 0'})
                        ], style={'display': 'flex', 'marginTop': '10px'}),
                        html.Div([
                            html.Span('Negative', style={'color': '#ff4444', 'fontSize': '10px'}),
                            html.Span('Neutral', style={'color': '#888', 'fontSize': '10px', 'margin': '0 20px'}),
                            html.Span('Positive', style={'color': '#00ff88', 'fontSize': '10px'})
                        ], style={'marginTop': '5px'})
                    ])
                ])
            ], style={
                'backgroundColor': '#161b22',
                'padding': '20px',
                'borderRadius': '8px',
                'marginBottom': '20px'
            }),
            
            # News Drill-Down
            html.Div([
                html.H3('📰 News Drill-Down',
                        style={'color': '#fff', 'fontSize': '18px', 'marginBottom': '15px'}),
                html.Div(id='news-articles')
            ], style={
                'backgroundColor': '#161b22',
                'padding': '20px',
                'borderRadius': '8px'
            })
        ], style={'flex': '1'})
    ], style={'display': 'flex', 'padding': '20px'})
], style={'backgroundColor': '#0d1117', 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'})

# Callbacks
@app.callback(
    [Output('price-sentiment-chart', 'figure'),
     Output('correlation-chart', 'figure'),
     Output('keyword-cloud', 'children'),
     Output('news-articles', 'children'),
     Output('daily-mean', 'children'),
     Output('daily-median', 'children'),
     Output('net-sentiment', 'children'),
     Output('correlation', 'children')],
    [Input('ticker-dropdown', 'value'),
     Input('confidence-slider', 'value')]
)
def update_dashboard(ticker, confidence):
    # Fetch data from the mock loader
    dates, prices, sentiment, events = generate_sentiment_data(ticker)
    keywords = generate_keywords(ticker)
    articles = generate_news_articles(ticker)
    
    # Price & Sentiment Chart
    price_fig = go.Figure()
    
    price_fig.add_trace(go.Scatter(
        x=dates, y=prices,
        name='Price',
        line=dict(color='#00d4ff', width=2),
        yaxis='y1'
    ))
    
    price_fig.add_trace(go.Scatter(
        x=dates, y=sentiment,
        name='Sentiment',
        line=dict(color='#00ff88', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    for event in events:
        color = {'Positive Spike': '#00ff88', 'Negative Crash': '#ff4444', 'Volume Surge': '#ffaa00'}[event['type']]
        price_fig.add_trace(go.Scatter(
            x=[event['date']],
            y=[event['price']],
            mode='markers',
            marker=dict(size=10, color=color),
            name=event['type'],
            yaxis='y1',
            showlegend=False
        ))
    
    price_fig.update_layout(
        plot_bgcolor='#0d1117',
        paper_bgcolor='#161b22',
        font=dict(color='#fff'),
        xaxis=dict(gridcolor='#30363d', showgrid=True),
        yaxis=dict(title='Price ($)', side='left', gridcolor='#30363d', showgrid=True),
        yaxis2=dict(title='Sentiment', side='right', overlaying='y', gridcolor='#30363d', showgrid=False),
        margin=dict(l=40, r=40, t=20, b=40),
        height=350,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Correlation Chart
    corr_sentiment = np.random.randn(100)
    corr_returns = corr_sentiment * 0.02 + np.random.randn(100) * 0.02
    
    corr_fig = go.Figure()
    corr_fig.add_trace(go.Scatter(
        x=corr_sentiment,
        y=corr_returns,
        mode='markers',
        marker=dict(size=8, color='#00d4ff', opacity=0.6),
        showlegend=False
    ))
    
    corr_fig.update_layout(
        plot_bgcolor='#0d1117',
        paper_bgcolor='#161b22',
        font=dict(color='#fff'),
        xaxis=dict(title='Sentiment', gridcolor='#30363d', showgrid=True, zeroline=True, zerolinecolor='#30363d'),
        yaxis=dict(title='Next-day Return', gridcolor='#30363d', showgrid=True, zeroline=True, zerolinecolor='#30363d'),
        margin=dict(l=40, r=40, t=20, b=40),
        height=300,
        annotations=[
            dict(text=f'R² = 0.42', x=0.05, y=0.95, xref='paper', yref='paper',
                 showarrow=False, font=dict(color='#fff', size=12)),
            dict(text='p-value < 0.01', x=0.05, y=0.88, xref='paper', yref='paper',
                 showarrow=False, font=dict(color='#888', size=10)),
            dict(text='n = 100', x=0.95, y=0.05, xref='paper', yref='paper',
                 showarrow=False, font=dict(color='#888', size=10))
        ]
    )
    
    # Keyword Cloud
    keyword_elements = []
    for word, size in keywords:
        keyword_elements.append(
            html.Span(word, style={
                'fontSize': f'{size}px',
                'color': '#fff' if size > 30 else '#888',
                'margin': '5px 10px',
                'display': 'inline-block',
                'fontWeight': 'bold' if size > 35 else 'normal'
            })
        )
    
    # News Articles
    news_elements = []
    for article in articles[:2]:
        news_elements.append(
            html.Div([
                html.Div([
                    html.Span(article['source'], style={'color': '#00d4ff', 'fontSize': '12px'}),
                    html.Span(f" • {article['date'].strftime('%Y-%m-%d')}", 
                             style={'color': '#888', 'fontSize': '12px'}),
                    html.Span(f" ↗ {article['sentiment']}", 
                             style={'color': '#00ff88', 'fontSize': '12px', 'float': 'right'})
                ], style={'marginBottom': '5px'}),
                html.Div(article['title'], 
                        style={'color': '#fff', 'fontSize': '14px', 'fontWeight': 'bold', 'marginBottom': '5px'}),
                html.Div(article['summary'],
                        style={'color': '#888', 'fontSize': '12px', 'lineHeight': '1.4'})
            ], style={
                'padding': '15px',
                'backgroundColor': '#0d1117',
                'borderRadius': '6px',
                'marginBottom': '10px',
                'border': '1px solid #30363d'
            })
        )
    
    # Stats
    daily_mean = f"+{sentiment.mean():.3f}"
    daily_median = f"+{np.median(sentiment):.3f}"
    net_sentiment = f"+{sentiment.sum():.1f}"
    correlation_text = f"{np.corrcoef(sentiment[:-1], np.diff(prices))[0,1]:.2f} 1d lag"
    
    return price_fig, corr_fig, keyword_elements, news_elements, daily_mean, daily_median, net_sentiment, correlation_text
