from dash import Dash, html, dcc
from viz.example_plot import create_example_figure

app = Dash(__name__)
server = app.server  # IMPORTANT for Plotly Cloud

app.layout = html.Div(
    style={"padding": "40px"},
    children=[
        html.H1("Plotly Cloud Demo App"),
        dcc.Graph(
            id="example-graph",
            figure=create_example_figure()
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
