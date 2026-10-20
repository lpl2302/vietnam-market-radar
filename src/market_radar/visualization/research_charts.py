"""Sector and retrospective Plotly views using saved historical observations."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from market_radar.visualization.charts import AMBER, BLUE, NAVY, RED, TEAL, _layout


def sector_heatmap(monthly: pd.DataFrame) -> go.Figure:
    """Heatmap of flagged/eligible stock-sessions by covered sector and month."""
    grouped = monthly.groupby(["sector", "month"], sort=True)[["anomalies", "eligible_observations"]].sum().reset_index()
    grouped["rate_pct"] = 100 * grouped.anomalies / grouped.eligible_observations
    rates = grouped.pivot(index="sector", columns="month", values="rate_pct")
    flags = grouped.pivot(index="sector", columns="month", values="anomalies").reindex_like(rates)
    eligible = grouped.pivot(index="sector", columns="month", values="eligible_observations").reindex_like(rates)
    fig = go.Figure(go.Heatmap(x=list(rates.columns), y=list(rates.index), z=rates.to_numpy(),
                               customdata=np.stack([flags.to_numpy(), eligible.to_numpy()], axis=-1),
                               colorscale=[[0, "#eff5f6"], [0.45, "#8bbeb9"], [1, TEAL]],
                               colorbar=dict(title="Flag rate %"),
                               hovertemplate="%{y} · %{x}<br>Flags: %{customdata[0]:.0f}<br>Eligible: %{customdata[1]:.0f}<br>Flag rate: %{z:.2f}%<extra></extra>"))
    fig.update_xaxes(title_text="Trading month")
    return _layout(fig, height=390)


def sector_comparison_chart(summary: pd.DataFrame, metric: str) -> go.Figure:
    if metric not in ("Anomaly rate", "Anomaly count"):
        raise ValueError("Unknown sector comparison metric")
    column = "anomaly_rate_pct" if metric == "Anomaly rate" else "anomalies"
    ordered = summary.sort_values(column)
    labels = [f"{row.anomalies}/{row.eligible_observations} · {row.covered_stocks} stock(s)"
              for row in ordered.itertuples()]
    fig = go.Figure(go.Bar(x=ordered[column], y=ordered.sector, orientation="h",
                           marker_color=RED, text=labels, textposition="outside", cliponaxis=False,
                           customdata=ordered[["covered_stocks", "eligible_observations", "anomalies", "anomaly_rate_pct"]].to_numpy(),
                           hovertemplate="%{y}<br>Covered stocks: %{customdata[0]}<br>Eligible: %{customdata[1]}<br>Flags: %{customdata[2]}<br>Rate: %{customdata[3]:.2f}%<extra></extra>"))
    fig.update_xaxes(title_text="Rate of eligible sessions (%)" if metric == "Anomaly rate" else "Flagged sessions",
                     range=[0, max(1, ordered[column].max() * 1.62)])
    return _layout(fig, height=390)


def sector_timeline(monthly: pd.DataFrame, sector: str) -> go.Figure:
    selected = monthly.loc[monthly.sector == sector]
    fig = go.Figure()
    for partition, color, label in (("train", AMBER, "Training"), ("holdout", BLUE, "Holdout")):
        part = selected.loc[selected.partition == partition].sort_values("month")
        fig.add_trace(go.Bar(x=part.month, y=part.anomalies, name=label, marker_color=color,
                             customdata=part[["eligible_observations", "anomaly_rate_pct"]].to_numpy(),
                             hovertemplate="%{x}<br>Flags: %{y}<br>Eligible: %{customdata[0]}<br>Rate: %{customdata[1]:.2f}%<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Flagged stock-sessions")
    fig.update_xaxes(title_text="Trading month")
    return _layout(fig, height=340)


def normalized_stock_chart(stock_rows: pd.DataFrame, partitions: list[str]) -> go.Figure:
    """Normalize each covered stock to its first observed close in the view."""
    fig = go.Figure()
    palette = [BLUE, TEAL, "#74549a", "#bb7934", NAVY]
    for index, (symbol, stock) in enumerate(stock_rows.groupby("symbol", sort=True)):
        ordered = stock.sort_values("date").copy()
        ordered["index_level"] = 100 * ordered.close / ordered.close.iloc[0]
        color = palette[index % len(palette)]
        fig.add_trace(go.Scatter(x=ordered.date, y=ordered.index_level, name=symbol, mode="lines",
                                 line=dict(color=color, width=2),
                                 hovertemplate=f"{symbol} · %{{x}}<br>Index: %{{y:.1f}}<extra></extra>"))
        marked = ordered.loc[ordered.anomaly_flag.astype("boolean").fillna(False) & ordered.partition.isin(partitions)]
        fig.add_trace(go.Scatter(x=marked.date, y=marked.index_level, name=f"{symbol} anomaly",
                                 mode="markers", showlegend=False,
                                 marker=dict(color=RED, size=10, symbol="diamond",
                                             line=dict(color="white", width=1)),
                                 customdata=marked.partition,
                                 hovertemplate=f"{symbol} anomaly · %{{x}}<br>Index: %{{y:.1f}}<br>%{{customdata}}<extra></extra>"))
    fig.update_yaxes(title_text="Normalized close (first observed = 100)")
    fig.update_xaxes(type="date", title_text="Trading date")
    return _layout(fig, height=390)


def event_study_box(outcomes: pd.DataFrame) -> go.Figure:
    complete = outcomes.loc[(outcomes.partition == "holdout") & outcomes.next_five_session_abs_return.notna()].copy()
    flagged = complete.anomaly_flag.astype(bool)
    baseline = complete.baseline_volume_flag.astype(bool) | complete.baseline_return_flag.astype(bool)
    groups = (("Unflagged", ~flagged, BLUE), ("Flagged", flagged, RED),
              ("Model only", flagged & ~baseline, TEAL),
              ("Model + baseline", flagged & baseline, AMBER))
    fig = go.Figure()
    for label, mask, color in groups:
        values = complete.loc[mask, "next_five_session_abs_return"] * 100
        fig.add_trace(go.Box(y=values, name=f"{label} (n={len(values)})", marker_color=color,
                             boxpoints="outliers", hovertemplate="Absolute five-session return: %{y:.2f}%<extra></extra>"))
    fig.update_yaxes(title_text="Absolute next-five-session return (%)")
    fig.update_layout(showlegend=False)
    return _layout(fig, height=390)
