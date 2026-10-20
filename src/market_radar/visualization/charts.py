"""Plotly figures for historical market research."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from market_radar.visualization.data import grouped_frequency, monthly_frequency


NAVY = "#183153"
BLUE = "#2468a0"
TEAL = "#147d75"
RED = "#bc3945"
AMBER = "#d08a29"
GRAY = "#8796a6"


def _layout(fig: go.Figure, *, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=14, t=36, b=20),
        font=dict(family="Arial, sans-serif", color=NAVY),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e8edf2", zerolinecolor="#dce3ea")
    return fig


def frequency_chart(frame: pd.DataFrame, column: str) -> go.Figure:
    """Show readable counts and eligible-session rates for each category."""
    data = grouped_frequency(frame, column).sort_values("anomalies")
    labels = [f"{row.anomalies} · {row.rate_pct:.1f}%" for row in data.itertuples()]
    fig = go.Figure(go.Bar(
        x=data.anomalies, y=data[column], orientation="h", marker_color=RED,
        text=labels, textposition="outside", cliponaxis=False,
        customdata=data[["observations", "rate_pct"]].to_numpy(),
        hovertemplate="%{y}<br>Flags: %{x}<br>Eligible: %{customdata[0]}<br>Rate: %{customdata[1]:.2f}%<extra></extra>",
    ))
    fig.update_xaxes(title_text="Flagged sessions", range=[0, max(1, data.anomalies.max() * 1.42)])
    return _layout(fig, height=390)


def time_chart(frame: pd.DataFrame) -> go.Figure:
    data = monthly_frequency(frame)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=data.month, y=data.anomalies, name="Flagged sessions", marker_color=RED,
                         hovertemplate="%{x}<br>Flags: %{y}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=data.month, y=data.rate_pct, name="Monthly flag rate", mode="lines+markers",
                             line=dict(color=TEAL, width=2),
                             hovertemplate="%{x}<br>Rate: %{y:.2f}%<extra></extra>"), secondary_y=True)
    fig.update_yaxes(title_text="Flagged sessions", secondary_y=False)
    fig.update_yaxes(title_text="Rate of eligible sessions (%)", secondary_y=True)
    return _layout(fig)


def score_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for partition, label, color in (("train", "Training", AMBER), ("holdout", "Holdout", BLUE)):
        values = frame.loc[frame.partition == partition, "anomaly_score"]
        fig.add_trace(go.Histogram(x=values, name=label, marker_color=color, opacity=0.72,
                                   hovertemplate="Score: %{x:.3f}<br>Sessions: %{y}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color=RED,
                  annotation_text="Fitted flag boundary", annotation_position="top right")
    fig.update_layout(barmode="overlay", xaxis_title="Anomaly score (higher = more unusual)",
                      yaxis_title="Sessions")
    return _layout(fig)


def partition_chart(frame: pd.DataFrame) -> go.Figure:
    data = grouped_frequency(frame, "partition")
    order = [name for name in ("train", "holdout") if name in set(data.partition)]
    data = data.set_index("partition").loc[order].reset_index()
    fig = go.Figure(go.Bar(x=data.partition, y=data.rate_pct, marker_color=[AMBER, BLUE][:len(data)],
                           text=[f"{row.anomalies}/{row.observations}" for row in data.itertuples()],
                           textposition="outside",
                           hovertemplate="%{x}<br>Flag rate: %{y:.2f}%<extra></extra>"))
    fig.update_yaxes(title_text="Rate of eligible sessions (%)")
    return _layout(fig, height=310)


def price_volume_chart(stock: pd.DataFrame, *, event_date: str | None = None) -> go.Figure:
    """Plot real close/volume dates and markers at their corresponding close."""
    stock = stock.sort_values("date")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.72, 0.28])
    fig.add_trace(go.Scatter(x=stock.date, y=stock.close, name="Close", mode="lines",
                             line=dict(color=BLUE, width=2),
                             hovertemplate="%{x}<br>Close: %{y:,.2f} thousand VND<extra></extra>"), row=1, col=1)
    for partition, label, color, symbol in (("train", "Training anomaly", AMBER, "circle"),
                                            ("holdout", "Holdout anomaly", RED, "diamond")):
        points = stock.loc[(stock.partition == partition) & stock.anomaly_flag.astype("boolean").fillna(False)]
        fig.add_trace(go.Scatter(x=points.date, y=points.close, name=label, mode="markers",
                                 marker=dict(color=color, size=11, symbol=symbol,
                                             line=dict(color="white", width=1.5)),
                                 customdata=points.anomaly_score,
                                 hovertemplate="%{x}<br>Close: %{y:,.2f} thousand VND<br>Score: %{customdata:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=stock.date, y=stock.volume, name="Volume", marker_color="#a6b7c7",
                         hovertemplate="%{x}<br>Volume: %{y:,.0f} shares<extra></extra>"), row=2, col=1)
    if event_date is not None:
        fig.add_vline(x=event_date, line_dash="dash", line_color=RED, row=1, col=1)
    fig.update_yaxes(title_text="Close (thousand VND)", row=1, col=1)
    fig.update_yaxes(title_text="Shares", row=2, col=1)
    fig.update_xaxes(type="date", title_text="Trading date", row=2, col=1)
    return _layout(fig, height=510)


def baseline_chart(model: int, volume: int, returns: int) -> go.Figure:
    fig = go.Figure(go.Bar(x=["Isolation Forest", "Volume z > 3", "|Return z| > 3"],
                           y=[model, volume, returns], marker_color=[RED, BLUE, TEAL],
                           text=[model, volume, returns], textposition="outside",
                           hovertemplate="%{x}<br>Flagged holdout sessions: %{y}<extra></extra>"))
    fig.update_yaxes(title_text="Flagged holdout sessions")
    return _layout(fig, height=320)
