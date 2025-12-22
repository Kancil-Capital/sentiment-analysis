from dash import html, dcc
from app.viz.figures import create_price_sentiment_figure, create_lag_scatter, create_word_cloud, create_sentiment_distribution


# ---------- small UI helpers ----------

def card(children, padding="16px"):
    return html.Div(
        children=children,
        style={
            "backgroundColor": "#0b1220",
            "border": "1px solid #1e293b",
            "borderRadius": "16px",
            "padding": padding,
        },
    )


def section(title, subtitle=None):
    return html.Div(
        style={"marginBottom": "12px"},
        children=[
            html.H3(
                title,
                style={
                    "margin": 0,
                    "color": "#e5e7eb",
                    "fontSize": "16px",
                    "fontWeight": "600",
                },
            ),
            html.P(
                subtitle or "",
                style={
                    "margin": "4px 0 0",
                    "color": "#94a3b8",
                    "fontSize": "13px",
                },
            ),
        ],
    )


def kpi(label, value):
    return html.Div(
        children=[
            html.P(label, style={"color": "#94a3b8", "fontSize": "12px"}),
            html.H3(value, style={"color": "#f8fafc", "margin": "4px 0"}),
        ]
    )


# ---------- main layout ----------

def create_dashboard_layout(df, keyword_data):
    return html.Div(
        style={
            "backgroundColor": "#020617",
            "minHeight": "100vh",
            "padding": "32px",
            "fontFamily": "Inter, system-ui, sans-serif",
        },
        children=[

            # ================= HEADER =================
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "28px",
                },
                children=[
                    html.Div([
                        html.H1(
                            "Equity Sentiment Intelligence",
                            style={
                                "margin": 0,
                                "color": "#f8fafc",
                                "fontWeight": "700",
                            },
                        ),
                        html.P(
                            "News-driven ML sentiment & market response",
                            style={"margin": "6px 0 0", "color": "#94a3b8"},
                        ),
                    ]),
                    html.Div(
                        "LIVE · DEMO",
                        style={
                            "border": "1px solid #1e293b",
                            "padding": "6px 14px",
                            "borderRadius": "999px",
                            "color": "#22c55e",
                            "fontSize": "12px",
                            "fontWeight": "600",
                        },
                    ),
                ],
            ),

            # ================= KPI ROW =================
            card(
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(5, 1fr)",
                        "gap": "20px",
                    },
                    children=[
                        kpi("Avg Sentiment", f"{df['sentiment'].mean():+.2f}"),
                        kpi(
                            "Positive / Neutral / Negative",
                            f"{(df['sentiment']>0).mean():.0%} / "
                            f"{(df['sentiment']==0).mean():.0%} / "
                            f"{(df['sentiment']<0).mean():.0%}",
                        ),
                        kpi("Total News Volume", f"{len(df):,}"),
                        kpi(
                            "Price Return",
                            f"{(df['price'].iloc[-1]/df['price'].iloc[0]-1):+.2%}",
                        ),
                        kpi(
                            "Sentiment ↔ Return Corr",
                            f"{df['sentiment'].corr(df['return']):.2f}",
                        ),
                    ],
                )
            ),

            html.Div(style={"height": "28px"}),

            # ================= HERO CHART =================
            section(
                "Price, Sentiment & Events",
                "Dual-axis view showing how market price responds to sentiment shifts and news events",
            ),
            card(
                dcc.Graph(
                    figure=create_price_sentiment_figure(df),
                    config={"displayModeBar": False},
                ),
                padding="20px",
            ),

            html.Div(style={"height": "28px"}),

            # ================= ANALYSIS GRID =================
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "2fr 1fr",
                    "gap": "24px",
                },
                children=[

                    # Predictive scatter
                    card([
                        section(
                            "Sentiment → Forward Returns",
                            "Tests whether sentiment has predictive power over future price moves",
                        ),
                        dcc.Graph(
                            figure=create_lag_scatter(df, lag=1),
                            config={"displayModeBar": False},
                        ),
                    ]),

                    # Distribution
                    card([
                        section(
                            "Sentiment Distribution",
                            "Overall polarity and bias of the news coverage",
                        ),
                        dcc.Graph(
                            figure=create_sentiment_distribution(df),
                            config={"displayModeBar": False},
                        ),
                    ]),
                ],
            ),

            html.Div(style={"height": "28px"}),

            # ================= NARRATIVE =================
            section(
                "Narrative Drivers",
                "Keywords most responsible for sentiment signals across the period",
            ),
            card(
                dcc.Graph(
                    figure=create_word_cloud(keyword_data),
                    config={"displayModeBar": False},
                )
            ),
        ],
    )