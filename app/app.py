from dash import Dash, html, dcc
from viz.example_plot import create_example_figure

app = Dash(__name__)
server = app.server 

app.layout = html.Div([
    html.H1("Sentiment Analysis Dashboard"),
    dcc.Graph(figure=create_example_figure())
])

if __name__ == "__main__":
    app.run(debug=True) 
