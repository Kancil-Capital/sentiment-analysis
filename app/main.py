from dotenv import load_dotenv

load_dotenv(".env")

from datetime import datetime, timedelta
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.data.main import get_articles, get_price_data

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Price & Sentiment Dashboard"),
    
    html.Div([
        html.Label("Ticker:"),
        dcc.Input(id='ticker-input', type='text', value='META', style={'marginLeft': '10px'}),
        
        html.Label("Start Date:", style={'marginLeft': '20px'}),
        dcc.DatePickerSingle(
            id='start-date',
            date=(datetime.now() - timedelta(days=30)).date(),
            style={'marginLeft': '10px'}
        ),
        
        html.Label("End Date:", style={'marginLeft': '20px'}),
        dcc.DatePickerSingle(
            id='end-date',
            date=datetime.now().date(),
            style={'marginLeft': '10px'}
        ),
        
        html.Button('Load Data', id='submit-button', n_clicks=0, style={'marginLeft': '20px'}),
    ], style={'marginBottom': '20px'}),
    
    html.Div([
        html.Label("Filter by Affected Party:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Checklist(
            id='affected-filter',
            options=[],
            value=[],
            inline=True,
            style={'display': 'inline-block'}
        ),
    ], id='filter-container', style={'marginBottom': '20px', 'display': 'none'}),
    
    dcc.Graph(id='price-sentiment-graph'),
    dcc.Store(id='articles-store'),
])

@app.callback(
    [Output('articles-store', 'data'),
     Output('affected-filter', 'options'),
     Output('affected-filter', 'value'),
     Output('filter-container', 'style')],
    Input('submit-button', 'n_clicks'),
    State('ticker-input', 'value'),
    State('start-date', 'date'),
    State('end-date', 'date')
)
def load_data(n_clicks, ticker, start_date, end_date):
    if n_clicks == 0:
        return None, [], [], {'display': 'none'}
    
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    
    # Get articles
    articles_df = get_articles(ticker, start_dt, end_dt)
    
    # Get distinct affected parties
    affected_parties = sorted(articles_df['affected'].unique().tolist())
    
    # Create checkbox options
    options = [{'label': party, 'value': party} for party in affected_parties]
    
    # Store articles as dict
    articles_data = articles_df.to_dict('records')
    
    return articles_data, options, affected_parties, {'marginBottom': '20px', 'display': 'block'}

@app.callback(
    Output('price-sentiment-graph', 'figure'),
    [Input('submit-button', 'n_clicks'),
     Input('affected-filter', 'value')],
    [State('ticker-input', 'value'),
     State('start-date', 'date'),
     State('end-date', 'date'),
     State('articles-store', 'data')]
)
def update_graph(n_clicks, selected_affected, ticker, start_date, end_date, articles_data):
    if n_clicks == 0 or articles_data is None:
        return go.Figure()
    
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    
    # Get price data
    price_df = get_price_data(ticker, start_dt, end_dt)
    
    # Convert articles back to DataFrame and filter
    articles_df = pd.DataFrame(articles_data)
    
    if selected_affected:
        articles_df = articles_df[articles_df['affected'].isin(selected_affected)]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price candlestick
    fig.add_trace(
        go.Candlestick(
            x=price_df['timestamp'],
            open=price_df['open'],
            high=price_df['high'],
            low=price_df['low'],
            close=price_df['close'],
            name='Price'
        ),
        secondary_y=False
    )
    
    # Add sentiment scatter
    if len(articles_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=articles_df['timestamp'],
                y=articles_df['sentiment'],
                mode='markers',
                name='Sentiment',
                marker=dict(
                    size=articles_df['confidence'] * 10,  # Scale confidence for visibility
                    color=articles_df['sentiment'],
                    colorscale='RdYlGn',
                    cmin=-1,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(title="Sentiment", x=1.1)
                ),
                text=articles_df['title'],
                hovertemplate='<b>%{text}</b><br>Sentiment: %{y:.2f}<br>%{x}<extra></extra>'
            ),
            secondary_y=True
        )
    
    # Update axes
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, range=[-1, 1])
    
    fig.update_layout(
        title=f"{ticker} Price & Sentiment",
        hovermode='x unified',
        height=600
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)
