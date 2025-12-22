from dash import Dash
from app.viz.layout import create_dashboard_layout
from app.viz.tempdata import load_demo_data

app = Dash(__name__)
server = app.server  # IMPORTANT for Plotly Cloud

df, keyword_data = load_demo_data()

app.layout = create_dashboard_layout(df, keyword_data)

if __name__ == "__main__":
    app.run(debug=True)
