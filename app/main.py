from dotenv import load_dotenv

load_dotenv(".env")

from dash import Dash, html

from app.data.main import get_articles, get_price_data

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Hello World!")
])

if __name__ == '__main__':
    app.run(debug=True, port=8050)
