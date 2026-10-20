"""Streamlit views of saved historical anomaly detections."""

from datetime import date
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from market_radar.analysis.storage import AnalysisBundle, AnalysisDataError, FILES, load_analysis
from market_radar.config import ANALYSIS_DATA_DIR, ANOMALY_DATA_DIR, ARTIFACTS_DIR, FEATURE_DATA_DIR
from market_radar.visualization.charts import (
    baseline_chart, frequency_chart, partition_chart, price_volume_chart, score_chart, time_chart,
)
from market_radar.visualization.data import (
    DashboardData, DashboardDataError, baseline_overlap, event_context, filter_scored,
    frequency, grouped_frequency, load_dashboard_data,
)
from market_radar.visualization.research_views import (
    render_brief_download, render_event_followup, render_event_study, render_model_only_detail,
)
from market_radar.visualization.sector_page import render_sector_intelligence


FEATURE_DESCRIPTIONS = {
    "return_1d": "One-session close-to-close return (fraction)",
    "return_5d": "Five-session close-to-close return (fraction)",
    "intraday_range_pct": "High–low range divided by open (%)",
    "price_vs_prev_ma20_pct": "Close versus prior 20-session mean (%)",
    "volume_ratio_prev20": "Volume versus prior 20-session mean (ratio)",
    "volume_zscore_prev20": "Volume versus prior 20-session mean (standard deviations)",
    "return_volatility_prev20": "Prior 20-session return volatility (fraction)",
    "return_zscore_prev20": "Return versus prior 20-session mean (standard deviations)",
    "vnindex_return_1d": "Same-date VNINDEX return (fraction)",
    "excess_return_1d": "Stock return minus VNINDEX return (fraction)",
}


def _file_version(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size
    except OSError:
        return str(path), 0, 0


@st.cache_data(show_spinner=False)
def _cached_load(versions: tuple[tuple[str, int, int], ...]) -> DashboardData:
    paths = [Path(item[0]) for item in versions]
    return load_dashboard_data(*paths)


def _load_default() -> DashboardData:
    paths = (ANOMALY_DATA_DIR / "market_anomalies.csv", ANOMALY_DATA_DIR / "evaluation.json",
             FEATURE_DATA_DIR / "market_features.csv", ARTIFACTS_DIR / "isolation_forest.meta.json")
    return _cached_load(tuple(_file_version(path) for path in paths))


@st.cache_data(show_spinner=False)
def _cached_analysis(versions: tuple[tuple[str, int, int], ...], results: pd.DataFrame,
                     feature_sha256: str) -> AnalysisBundle:
    return load_analysis(ANALYSIS_DATA_DIR, ANOMALY_DATA_DIR / "market_anomalies.csv", results,
                         expected_feature_sha256=feature_sha256)


def _load_analysis(data: DashboardData) -> AnalysisBundle:
    paths = [ANALYSIS_DATA_DIR / name for name in FILES.values()]
    paths.append(ANOMALY_DATA_DIR / "market_anomalies.csv")
    return _cached_analysis(tuple(_file_version(path) for path in paths), data.results,
                            data.model_metadata["source_feature_sha256"])


def _analysis_or_message(data: DashboardData) -> AnalysisBundle | None:
    try:
        return _load_analysis(data)
    except (AnalysisDataError, OSError, ValueError) as exc:
        st.info(f"Research analysis unavailable: {exc}. Run `python -m market_radar.build_analysis` from the repository root.")
        return None


def _reset(*keys: str) -> None:
    for key in keys:
        st.session_state.pop(key, None)
        if key == "investigate_event":
            for saved_key in list(st.session_state):
                if saved_key.startswith("investigate_event_"):
                    st.session_state.pop(saved_key, None)


def _date_bounds(frame: pd.DataFrame, key: str) -> tuple[str, str]:
    start = date.fromisoformat(frame.date.min())
    end = date.fromisoformat(frame.date.max())
    selected = st.date_input("Trading-date range", value=(start, end), min_value=start,
                             max_value=end, key=key)
    if isinstance(selected, tuple) and len(selected) == 2:
        return selected[0].isoformat(), selected[1].isoformat()
    st.info("Select both the start and end dates to complete the range.")
    return "9999-12-31", "0001-01-01"


def _event_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "symbol", "sector", "partition", "anomaly_score", "return_1d",
               "volume_ratio_prev20", "excess_return_1d"]
    table = frame.loc[:, columns].copy()
    table["return_1d"] = (table["return_1d"] * 100).round(2)
    table["excess_return_1d"] = (table["excess_return_1d"] * 100).round(2)
    table["anomaly_score"] = table["anomaly_score"].round(4)
    table["volume_ratio_prev20"] = table["volume_ratio_prev20"].round(2)
    return table.rename(columns={"date": "Date", "symbol": "Stock", "sector": "Sector",
                                 "partition": "Partition", "anomaly_score": "Score",
                                 "return_1d": "Daily return (%)",
                                 "volume_ratio_prev20": "Volume / prior 20d",
                                 "excess_return_1d": "Versus VNINDEX (pp)"})


def _show_event(row: pd.Series) -> None:
    st.subheader(f"{row.symbol} · {row.date}")
    st.caption(f"{row.sector} · {row.partition.title()} observation · Historical session")
    cols = st.columns(4)
    cols[0].metric("Daily return", f"{row.return_1d:+.2%}")
    cols[1].metric("Volume / prior 20 sessions", f"{row.volume_ratio_prev20:.2f}×")
    cols[2].metric("Versus VNINDEX", f"{row.excess_return_1d * 100:+.2f} pp")
    cols[3].metric("Anomaly score", f"{row.anomaly_score:.4f}")
    st.info(str(row.anomaly_explanation))
    baseline_text = []
    if row.baseline_volume_flag:
        baseline_text.append("volume z-score > 3")
    if row.baseline_return_flag:
        baseline_text.append("absolute return z-score > 3")
    st.caption("Statistical baselines: " + (", ".join(baseline_text) if baseline_text else "neither rule flagged this event"))
    if row.volume_ratio_prev20 > 0:
        prior_average = row.volume / row.volume_ratio_prev20
        st.caption(f"Observed volume: {row.volume:,.0f} shares · Prior 20-session average: approximately {prior_average:,.0f} shares")


def render_overview(data: DashboardData) -> None:
    scored = filter_scored(data.results)
    all_stats = frequency(scored)
    holdout = filter_scored(data.results, partitions=["holdout"])
    holdout_stats = frequency(holdout)
    st.title("Market overview")
    st.caption("Historical Vietnamese equity sessions · Last available session: " + data.results.date.max())
    first = st.columns(3)
    first[0].metric("Stocks analyzed", f"{scored.symbol.nunique():,}")
    first[1].metric("Eligible stock-sessions", f"{all_stats['observations']:,}")
    first[2].metric("Detected anomalies", f"{all_stats['anomalies']:,}")
    second = st.columns(3)
    second[0].metric("Overall flag rate", f"{all_stats['rate_pct']:.2f}%")
    second[1].metric("Holdout flag rate", f"{holdout_stats['rate_pct']:.2f}%")
    second[2].metric("Latest dataset date", data.results.date.max())
    st.caption("Rates use model-ready sessions only. A higher flag rate is not a measure of risk or misconduct.")

    left, right = st.columns(2)
    with left:
        st.subheader("By stock")
        st.plotly_chart(frequency_chart(scored, "symbol"), width="stretch")
        stock_table = grouped_frequency(scored, "symbol")
        stock_table["rate_pct"] = stock_table["rate_pct"].round(2)
        st.dataframe(stock_table.rename(columns={
            "symbol": "Stock", "observations": "Eligible", "anomalies": "Flags", "rate_pct": "Rate (%)"}),
            width="stretch", hide_index=True)
    with right:
        st.subheader("By sector")
        st.plotly_chart(frequency_chart(scored, "sector"), width="stretch")
        sector_table = grouped_frequency(scored, "sector")
        sector_table["rate_pct"] = sector_table["rate_pct"].round(2)
        st.dataframe(sector_table.rename(columns={
            "sector": "Sector", "observations": "Eligible", "anomalies": "Flags", "rate_pct": "Rate (%)"}),
            width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Flag frequency over time")
        st.plotly_chart(time_chart(scored), width="stretch")
    with right:
        st.subheader("Score distribution")
        st.plotly_chart(score_chart(scored), width="stretch")
    st.subheader("Training versus holdout")
    st.plotly_chart(partition_chart(scored), width="stretch")

    st.subheader("Ranked historical anomalies")
    choice = st.selectbox("Dataset partition", ["Holdout", "Training", "Both"], key="overview_partition")
    partitions = {"Holdout": ["holdout"], "Training": ["train"], "Both": ["train", "holdout"]}[choice]
    table = filter_scored(data.results, partitions=partitions, anomalies_only=True)
    if table.empty:
        st.info("No anomalies match this partition.")
    else:
        st.dataframe(_event_table(table), width="stretch", hide_index=True)
        st.download_button("Download these anomalies (CSV)", table.to_csv(index=False),
                           file_name="market_overview_anomalies.csv", mime="text/csv")


def render_stock_explorer(data: DashboardData) -> None:
    st.title("Stock explorer")
    st.caption("Explore historical prices and flagged sessions. Close prices are in thousand VND; volume is shares.")
    symbols = sorted(data.results.symbol.unique())
    symbol = st.selectbox("Stock", symbols, key="explorer_stock")
    stock = data.results.loc[data.results.symbol == symbol].sort_values("date")
    sector = str(stock.sector.iloc[0])
    st.caption(f"{symbol} · {sector} · Available history: {stock.date.min()} to {stock.date.max()}")
    left, right = st.columns([2, 1])
    with left:
        start, end = _date_bounds(stock, "explorer_dates")
    with right:
        choice = st.selectbox("Anomaly partition", ["Both", "Holdout", "Training"], key="explorer_partition")
    st.button("Reset explorer filters", on_click=_reset, args=("explorer_dates", "explorer_partition"))
    partitions = {"Both": ["train", "holdout"], "Holdout": ["holdout"], "Training": ["train"]}[choice]
    eligible = filter_scored(data.results, symbols=[symbol], partitions=partitions, start=start, end=end)
    stats = frequency(eligible)
    plot_data = stock.loc[(stock.date >= start) & (stock.date <= end)].copy()
    plot_data.loc[~plot_data.partition.isin(partitions), "anomaly_flag"] = False
    if plot_data.empty:
        st.info("No trading sessions in this date range. Adjust the filters.")
        return
    latest = eligible.sort_values("date").iloc[-1] if not eligible.empty else None
    metrics = st.columns(5)
    metrics[0].metric("Eligible sessions", f"{stats['observations']:,}")
    metrics[1].metric("Anomalies", f"{stats['anomalies']:,} ({stats['rate_pct']:.1f}%)")
    metrics[2].metric("Mean daily return", f"{eligible.return_1d.mean():+.2%}" if latest is not None else "—")
    metrics[3].metric("Daily return volatility", f"{eligible.return_1d.std():.2%}" if len(eligible) > 1 else "—")
    metrics[4].metric("Latest eligible close", f"{latest.close:,.2f}" if latest is not None else "—",
                      help="Thousand VND per share; latest session within the selected filters")
    st.caption("Price and volume show every observed session in the selected date range. Metrics and anomaly markers follow the selected partition.")
    st.plotly_chart(price_volume_chart(plot_data), width="stretch")
    events = eligible.loc[eligible.anomaly_flag]
    if events.empty:
        st.info("No flagged sessions match these filters.")
        return
    st.subheader("Inspect a flagged session")
    labels = [f"{row.date} · score {row.anomaly_score:.4f} · {row.partition}" for row in events.itertuples()]
    selected = st.selectbox("Anomaly date", range(len(events)), format_func=lambda index: labels[index], key="explorer_event")
    _show_event(events.iloc[selected])


def render_investigation(data: DashboardData) -> None:
    st.title("Anomaly investigation")
    st.caption("Rank and inspect unusual historical observations. Default view is the out-of-sample holdout.")
    scored = filter_scored(data.results)
    symbols = sorted(scored.symbol.unique())
    sectors = sorted(scored.sector.unique())
    left, right = st.columns(2)
    with left:
        chosen_symbols = st.multiselect("Stocks", symbols, default=symbols, key="investigate_stocks")
        start, end = _date_bounds(scored, "investigate_dates")
    with right:
        chosen_sectors = st.multiselect("Sectors", sectors, default=sectors, key="investigate_sectors")
        choice = st.selectbox("Dataset partition", ["Holdout", "Training", "Both"], key="investigate_partition")
    first, second = st.columns(2)
    with first:
        model_only = st.checkbox("Model-only events (neither statistical baseline)", key="investigate_model_only")
    with second:
        min_score = st.number_input("Minimum anomaly score", min_value=0.0, value=0.0, step=0.01,
                                    format="%.3f", key="investigate_score")
    st.button("Reset investigation filters", on_click=_reset,
              args=("investigate_stocks", "investigate_sectors", "investigate_dates",
                    "investigate_partition", "investigate_model_only", "investigate_score",
                    "investigate_event", "investigate_event_version"))
    partitions = {"Holdout": ["holdout"], "Training": ["train"], "Both": ["train", "holdout"]}[choice]
    events = filter_scored(data.results, symbols=chosen_symbols, sectors=chosen_sectors,
                           partitions=partitions, start=start, end=end, anomalies_only=True,
                           model_only=model_only, min_score=min_score)
    st.caption(f"{len(events):,} matching flagged sessions · Sorted by anomaly score, highest first")
    if events.empty:
        st.info("No anomalies match these filters. Broaden the date range or reset the filters.")
        return
    st.dataframe(_event_table(events), width="stretch", hide_index=True)
    st.download_button("Download filtered events (CSV)", events.to_csv(index=False),
                       file_name="filtered_anomalies.csv", mime="text/csv")
    labels = [f"{row.date} · {row.symbol} · score {row.anomaly_score:.4f}"
              for row in events.itertuples()]
    option_version = sha256("\n".join(labels).encode("utf-8")).hexdigest()[:12]
    event_widget_key = f"investigate_event_{option_version}"
    if st.session_state.get("investigate_event_version") != option_version:
        st.session_state[event_widget_key] = None
        st.session_state["investigate_event_version"] = option_version
    selected = st.selectbox("Select an event", labels, index=None,
                            placeholder="Choose a flagged session", key=event_widget_key)
    if selected is None:
        st.info("Choose a flagged session to review its market context and later observed returns.")
        return
    event = events.iloc[labels.index(selected)]
    _show_event(event)
    if model_only:
        render_model_only_detail(event)
    st.subheader("Price and volume around the event")
    context = event_context(data.results, event.symbol, event.date, radius=10)
    st.plotly_chart(price_volume_chart(context, event_date=event.date), width="stretch")
    st.caption("Context uses up to ten observed sessions on each side of the selected date; missing sessions are not filled.")
    analysis = _analysis_or_message(data)
    if analysis is not None:
        outcome = render_event_followup(event, analysis)
        render_brief_download(event, outcome)


def render_methodology(data: DashboardData) -> None:
    st.title("Model & methodology")
    metadata = data.model_metadata
    scored = filter_scored(data.results)
    train = filter_scored(data.results, partitions=["train"])
    holdout = filter_scored(data.results, partitions=["holdout"])
    settings = metadata["hyperparameters"]
    st.markdown("A single pooled **Isolation Forest** analyzes combinations of returns, volume, volatility, and market-relative performance across the available stocks. A fitted StandardScaler is part of the saved model pipeline.")
    cols = st.columns(4)
    cols[0].metric("Trees", f"{settings['n_estimators']:,}")
    cols[1].metric("Training contamination", f"{settings['contamination']:.1%}")
    cols[2].metric("Random seed", str(settings["random_state"]))
    cols[3].metric("Holdout flag rate", f"{frequency(holdout)['rate_pct']:.2f}%")
    st.subheader("Chronological evaluation")
    split = pd.DataFrame([
        {"Partition": "Training (in-sample)", "Start": train.date.min(), "End": train.date.max(),
         "Eligible sessions": len(train), "Flagged": int(train.anomaly_flag.sum()),
         "Flag rate (%)": round(frequency(train)["rate_pct"], 2)},
        {"Partition": "Holdout (out-of-sample)", "Start": holdout.date.min(), "End": holdout.date.max(),
         "Eligible sessions": len(holdout), "Flagged": int(holdout.anomaly_flag.sum()),
         "Flag rate (%)": round(frequency(holdout)["rate_pct"], 2)},
    ])
    st.dataframe(split, width="stretch", hide_index=True)
    st.caption("The earliest 70% of model-ready trading dates fit the scaler and model. Every stock on a date shares its partition. The latest 30% are scored without refitting. This is one holdout, not a walk-forward backtest.")
    st.subheader("Ten model inputs")
    st.dataframe(pd.DataFrame([{"Feature": name, "Meaning / unit": FEATURE_DESCRIPTIONS.get(name, "")}
                               for name in metadata["feature_names"]]), width="stretch", hide_index=True)
    st.subheader("Holdout statistical baselines")
    overlap = baseline_overlap(holdout)
    st.plotly_chart(baseline_chart(overlap["model"], overlap["volume"], overlap["return"]), width="stretch")
    cols = st.columns(3)
    cols[0].metric("Model and either baseline", overlap["model_and_either"])
    cols[1].metric("Model only", overlap["model_only"])
    cols[2].metric("Either baseline", overlap["either"])
    st.caption("Fixed rules: volume z-score > 3 and absolute return z-score > 3. Their counts can overlap. These are comparisons between detectors, not accuracy measurements.")
    st.subheader("Interpretation")
    st.markdown("The score is the **negative Isolation Forest decision function**. Larger values are more unusual relative to the fitted training distribution; a score above zero crosses the model's flag boundary. It is not a probability or investment-confidence percentage. Event explanations are deterministic descriptions of observed values, not causal explanations or native feature attribution.")
    st.warning("There are no verified ground-truth anomaly labels. Flagged sessions may reflect normal volatility, corporate actions, or provider adjustments. This is historical research support, not a trading signal or evidence of misconduct.")
    st.caption(f"Scored observations: {len(scored):,} · Model created: {metadata['created_at_utc']} · Source feature SHA-256: {metadata['source_feature_sha256'][:12]}…")
    analysis = _analysis_or_message(data)
    if analysis is not None:
        render_event_study(analysis)


def render_app() -> None:
    st.set_page_config(page_title="Vietnam Market Intelligence Radar", page_icon="📈", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown("""<style>
      .block-container {max-width: 1420px; padding-top: 1.8rem;}
      h1, h2, h3 {color: #183153; letter-spacing: -0.02em;}
      [data-testid="stMetric"] {background: #f7f9fb; border: 1px solid #e3e9ef;
          border-radius: 8px; padding: 0.8rem 1rem;}
      [data-testid="stSidebar"] {background: #f5f8fa;}
    </style>""", unsafe_allow_html=True)
    st.sidebar.markdown("### Vietnam Market Radar")
    st.sidebar.caption("Historical research workspace")
    section = st.sidebar.radio("Navigate", ["Market Overview", "Stock Explorer",
                                            "Sector Intelligence", "Anomaly Investigation",
                                            "Model & Methodology"])
    try:
        data = _load_default()
    except DashboardDataError as exc:
        st.error(str(exc))
        st.info("Generate the offline inputs from the repository root: `python -m market_radar.build_features` then `python -m market_radar.detect_anomalies`. If files disagree, rerun anomaly detection against the current feature CSV.")
        st.stop()
    st.sidebar.caption("Latest available session: " + data.results.date.max())
    st.sidebar.caption("Offline · Saved historical results")
    if section == "Sector Intelligence":
        analysis = _analysis_or_message(data)
        if analysis is not None:
            render_sector_intelligence(data, analysis)
    else:
        pages = {"Market Overview": render_overview, "Stock Explorer": render_stock_explorer,
                 "Anomaly Investigation": render_investigation, "Model & Methodology": render_methodology}
        pages[section](data)
    st.divider()
    st.caption("Historical, unsupervised research output. Review original data and disclosures before drawing conclusions; no buy/sell recommendation is made.")
