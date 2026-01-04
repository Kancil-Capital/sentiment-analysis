from dotenv import load_dotenv

load_dotenv(".env")

from datetime import datetime, timedelta
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.data.main import get_articles, get_price_data
import re
from app.model.main import get_sentiment, get_sentiment_batch
from app.viz.data_loader import generate_sentiment_data, generate_keywords, generate_news_articles
import numpy as np
app = Dash(__name__, title="Sentiment Pulse")
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
    # --- Define date range ---
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # --- Fetch price data ---
    price_df = get_price_data(ticker, start_date, end_date)
    if price_df.empty:
        price_df = pd.DataFrame({'timestamp': [], 'close': []})
    
    # --- Fetch articles ---
    articles_df = get_articles(ticker, start_date, end_date)
    if articles_df.empty:
        articles_df = pd.DataFrame(columns=['title','body','sentiment','confidence','timestamp','source'])
    
    # --- Filter articles by confidence ---
    filtered_articles = articles_df[articles_df['confidence'] >= confidence]

    # --- Aggregate daily sentiment ---
    if not filtered_articles.empty:
        sentiment_df = pd.DataFrame({
            'date': pd.to_datetime(filtered_articles['timestamp']).dt.date,
            'sentiment': filtered_articles['sentiment']
        })
        daily_sentiment = sentiment_df.groupby('date').mean()
    else:
        daily_sentiment = pd.DataFrame(columns=['sentiment'])

    # --- Align sentiment with price data ---
    price_df['date'] = pd.to_datetime(price_df['timestamp']).dt.date
    combined_df = pd.merge(price_df, daily_sentiment, on='date', how='left')
    combined_df['sentiment'].fillna(0, inplace=True)  # fill missing sentiment with 0

    # --- Price & Sentiment Chart ---
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=combined_df['timestamp'], y=combined_df['close'],
        name='Price', line=dict(color='#00d4ff', width=2), yaxis='y1'
    ))
    price_fig.add_trace(go.Scatter(
        x=combined_df['timestamp'], y=combined_df['sentiment'],
        name='Sentiment', line=dict(color='#00ff88', width=2, dash='dot'), yaxis='y2'
    ))
    price_fig.update_layout(
        plot_bgcolor='#0d1117', paper_bgcolor='#161b22', font=dict(color='#fff'),
        xaxis=dict(gridcolor='#30363d', showgrid=True),
        yaxis=dict(title='Price ($)', side='left', gridcolor='#30363d', showgrid=True),
        yaxis2=dict(title='Sentiment', side='right', overlaying='y', showgrid=False),
        margin=dict(l=40, r=40, t=20, b=40), height=350, hovermode='x unified'
    )

    # --- Correlation Chart (sentiment vs next-day returns) ---
    if len(combined_df) > 1:
        returns = combined_df['close'].diff().shift(-1).fillna(0)  # next-day return
        corr_fig = go.Figure()
        corr_fig.add_trace(go.Scatter(
            x=combined_df['sentiment'][:-1], y=returns[:-1],
            mode='markers', marker=dict(size=8, color='#00d4ff', opacity=0.6),
            showlegend=False
        ))
        if combined_df['sentiment'][:-1].std() > 0 and returns[:-1].std() > 0:
            corr_value = np.corrcoef(combined_df['sentiment'][:-1], returns[:-1])[0,1]
        else:
            corr_value = 0.0
    else:
        corr_fig = go.Figure()
        corr_value = 0.0

    corr_fig.update_layout(
        plot_bgcolor='#0d1117', paper_bgcolor='#161b22', font=dict(color='#fff'),
        xaxis=dict(title='Sentiment', gridcolor='#30363d', showgrid=True),
        yaxis=dict(title='Next-day Return', gridcolor='#30363d', showgrid=True),
        margin=dict(l=40, r=40, t=20, b=40), height=300,
        annotations=[dict(
            text=f'R² = {corr_value**2:.2f}', x=0.05, y=0.95, xref='paper', yref='paper',
            showarrow=False, font=dict(color='#fff', size=12)
        )]
    )

    # --- Keyword Cloud (top 15 words) ---
    keyword_elements = []
    if not filtered_articles.empty:
        all_text = " ".join(filtered_articles['title'].tolist() + filtered_articles['body'].tolist())
        words = pd.Series(re.findall(r'\b\w+\b', all_text.lower()))
        top_words = words.value_counts().head(15)
        for word, count in top_words.items():
            keyword_elements.append(html.Span(
                word, style={'fontSize': f'{8 + count*0.005}px', 'color':'#fff', 'margin':'5px', 'display':'inline-block'}
            ))

    # --- News Articles (top 2) ---
    news_elements = []
    for _, row in filtered_articles.head(2).iterrows():
        news_elements.append(html.Div([
            html.Div([
                html.Span(row['source'], style={'color': '#00d4ff', 'fontSize': '12px'}),
                html.Span(f" • {row['timestamp'].strftime('%Y-%m-%d')}", style={'color':'#888','fontSize':'12px'}),
                html.Span(f" ↗ {row['sentiment']:.2f}", style={'color':'#00ff88','fontSize':'12px','float':'right'})
            ], style={'marginBottom':'5px'}),
            html.Div(row['title'], style={'color':'#fff','fontWeight':'bold','marginBottom':'5px'}),
            html.Div(row.get('body',''), style={'color':'#888','fontSize':'12px','lineHeight':'1.4'})
        ], style={'padding':'15px','backgroundColor':'#0d1117','borderRadius':'6px','marginBottom':'10px','border':'1px solid #30363d'}))

    # --- Stats ---
    sentiments_array = combined_df['sentiment'].to_numpy()
    daily_mean = f"+{sentiments_array.mean():.3f}" if len(sentiments_array) > 0 else "+0.000"
    daily_median = f"+{np.median(sentiments_array):.3f}" if len(sentiments_array) > 0 else "+0.000"
    net_sentiment = f"+{sentiments_array.sum():.1f}" if len(sentiments_array) > 0 else "+0.0"
    correlation_text = f"{corr_value:.2f} 1d lag"

    return price_fig, corr_fig, keyword_elements, news_elements, daily_mean, daily_median, net_sentiment, correlation_text

if __name__ == '__main__':
    app.run(debug=True, port=8050)
