"""Interactive sector intelligence for the stocks actually covered by the dataset."""

from datetime import date

import pandas as pd
import streamlit as st

from market_radar.analysis.research import sector_monthly, sector_summary
from market_radar.analysis.storage import AnalysisBundle
from market_radar.visualization.data import DashboardData, filter_scored
from market_radar.visualization.research_charts import (
    normalized_stock_chart, sector_comparison_chart, sector_heatmap, sector_timeline,
)


def _reset() -> None:
    for key in ("sector_partition", "sector_dates", "sector_metric", "sector_drilldown"):
        st.session_state.pop(key, None)


def _stock_drilldown(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, stock in frame.groupby("symbol", sort=True):
        count = len(stock)
        flags = int(stock.anomaly_flag.sum())
        rows.append({"Stock": symbol, "Eligible": count, "Flags": flags,
                     "Flag rate (%)": round(100 * flags / count, 2),
                     "Mean daily return (%)": round(100 * stock.return_1d.mean(), 2),
                     "Daily volatility (%)": round(100 * stock.return_1d.std(ddof=1), 2) if count > 1 else None})
    return pd.DataFrame(rows)


def render_sector_intelligence(data: DashboardData, analysis: AnalysisBundle) -> None:
    st.title("Sector intelligence")
    st.caption("Industry labels summarize only the stocks covered by this historical dataset, not entire sectors.")
    scored = filter_scored(data.results)
    first, second, third = st.columns([1, 2, 1])
    with first:
        choice = st.selectbox("Dataset partition", ["Both", "Holdout", "Training"], key="sector_partition")
    with second:
        start_date = date.fromisoformat(scored.date.min())
        end_date = date.fromisoformat(scored.date.max())
        dates = st.date_input("Eligible trading-date range", value=(start_date, end_date),
                              min_value=start_date, max_value=end_date, key="sector_dates")
    with third:
        metric = st.selectbox("Compare sectors by", ["Anomaly rate", "Anomaly count"], key="sector_metric")
    st.button("Reset sector filters", on_click=_reset)
    if not isinstance(dates, tuple) or len(dates) != 2:
        st.info("Choose both dates to complete the range.")
        return
    partitions = {"Both": ["train", "holdout"], "Holdout": ["holdout"], "Training": ["train"]}[choice]
    filtered = filter_scored(data.results, partitions=partitions,
                             start=dates[0].isoformat(), end=dates[1].isoformat())
    if filtered.empty:
        st.info("No eligible stock-sessions match these filters.")
        return
    full_view = choice == "Both" and dates == (start_date, end_date)
    summary = analysis.sectors if full_view else sector_summary(filtered)
    monthly = analysis.monthly if full_view else sector_monthly(filtered)
    overview = summary.copy()
    for column in ("anomaly_rate_pct", "mean_daily_return_pct", "daily_return_volatility_pct", "mean_volume_ratio"):
        overview[column] = overview[column].round(2)
    st.subheader("Covered-stock summary")
    st.dataframe(overview.rename(columns={
        "sector": "Sector", "covered_stocks": "Stocks", "symbols": "Covered symbols",
        "eligible_observations": "Eligible sessions", "anomalies": "Flags",
        "anomaly_rate_pct": "Flag rate (%)", "mean_daily_return_pct": "Mean daily return (%)",
        "daily_return_volatility_pct": "Daily volatility (%)",
        "mean_volume_ratio": "Mean volume / prior 20d",
        "latest_observation_date": "Latest observed date",
    }), width="stretch", hide_index=True)
    st.caption("Flag-rate denominator: model-ready stock-sessions among covered securities in the selected dates and partition. Daily volatility is the standard deviation of observed daily returns.")
    st.download_button("Download filtered sector summary (CSV)", summary.to_csv(index=False),
                       file_name="sector_summary.csv", mime="text/csv")
    left, right = st.columns(2)
    with left:
        st.subheader("Sector comparison")
        st.plotly_chart(sector_comparison_chart(summary, metric), width="stretch")
    with right:
        st.subheader("Monthly anomaly-rate heatmap")
        st.plotly_chart(sector_heatmap(monthly), width="stretch")
        st.caption("Each cell is flagged stock-sessions divided by eligible stock-sessions in that sector and month; blank cells have no covered observation.")
    sector = st.selectbox("Industry drilldown", sorted(summary.sector), key="sector_drilldown")
    selected = filtered.loc[filtered.sector == sector]
    st.subheader(f"{sector} activity")
    st.caption(f"{selected.symbol.nunique()} covered stock(s) · {len(selected):,} eligible sessions · {int(selected.anomaly_flag.sum())} flags")
    st.plotly_chart(sector_timeline(monthly, sector), width="stretch")
    st.caption("Training and holdout are separate bars. Compare frequencies alongside their eligible-session denominators in the hover labels.")
    st.subheader("Covered-stock comparison")
    st.dataframe(_stock_drilldown(selected), width="stretch", hide_index=True)
    historical = data.results.loc[(data.results.sector == sector) &
                                  data.results.date.between(dates[0].isoformat(), dates[1].isoformat())]
    st.plotly_chart(normalized_stock_chart(historical, partitions), width="stretch")
    st.caption("Each line starts at 100 on its first observed session in this view; diamond markers show selected-partition anomalies. No missing sessions are interpolated.")
    st.warning("Coverage is ten selected securities; several industries contain one stock. Differences in flag rates are descriptive sample patterns, not sector risk rankings.")

    with st.expander("Focused OHLCV review: GAS and VHM"):
        for symbol in ("GAS", "VHM"):
            report = analysis.diagnostics.get(symbol, {})
            if report.get("status") != "available":
                st.info(f"{symbol} is absent from this saved dataset.")
                continue
            train, holdout = report["partitions"]["train"], report["partitions"]["holdout"]
            st.markdown(f"**{symbol}** · Training {train['anomalies']}/{train['eligible']} ({train['rate_pct']:.2f}%) · "
                        f"Holdout {holdout['anomalies']}/{holdout['eligible']} ({holdout['rate_pct']:.2f}%)")
            st.caption(f"Maximum absolute daily return {report['maximum_absolute_daily_return_pct']:.2f}%; "
                       f"maximum volume ratio {report['maximum_volume_ratio']:.2f}×; "
                       f"inconsistent OHLC rows {report['ohlc_inconsistent_rows']}; "
                       f"invalid price rows {report['nonpositive_or_nonfinite_price_rows']}; "
                       f"invalid volume rows {report['negative_or_nonfinite_volume_rows']}.")
            st.caption("Monthly flagged counts: " + ", ".join(
                f"{month}: {count}" for month, count in report["flagged_months"].items()))
            candidates = pd.DataFrame(report["review_candidates"])
            if not candidates.empty:
                candidates["return_1d"] = (candidates["return_1d"] * 100).round(2)
                candidates["volume_ratio_prev20"] = candidates["volume_ratio_prev20"].round(2)
                st.dataframe(candidates[["date", "partition", "return_1d", "volume_ratio_prev20", "anomaly_score"]].rename(
                    columns={"date": "Date", "partition": "Partition", "return_1d": "Daily return (%)",
                             "volume_ratio_prev20": "Volume ratio", "anomaly_score": "Score"}),
                    width="stretch", hide_index=True)
        st.caption("Review the source's price-adjustment convention, public disclosures, and market context manually. These patterns do not identify a cause.")
