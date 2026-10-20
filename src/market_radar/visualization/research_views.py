"""Streamlit components for retrospective outcomes and research briefs."""

import pandas as pd
import streamlit as st

from market_radar.analysis.research import research_brief
from market_radar.analysis.storage import AnalysisBundle
from market_radar.visualization.research_charts import event_study_box


def render_model_only_detail(row: pd.Series) -> None:
    """Show observed features and the two unchanged Task 004 baseline tests."""
    st.subheader("Model-only pattern review")
    st.caption("Neither fixed statistical rule crossed its threshold. The pooled model considers all ten features jointly; this panel describes the observed pattern, not which features caused its flag.")
    detail = pd.DataFrame([
        {"Measure": "Prior five-session stock return", "Observed value": f"{row.return_5d:+.2%}"},
        {"Measure": "Daily stock return", "Observed value": f"{row.return_1d:+.2%}"},
        {"Measure": "Volume / prior 20-session average", "Observed value": f"{row.volume_ratio_prev20:.2f}×"},
        {"Measure": "Volume z-score", "Observed value": f"{row.volume_zscore_prev20:+.2f} (rule: > 3; not flagged)"},
        {"Measure": "Absolute return z-score", "Observed value": f"{abs(row.return_zscore_prev20):.2f} (rule: > 3; not flagged)"},
        {"Measure": "Prior daily return volatility", "Observed value": f"{row.return_volatility_prev20:.2%}"},
        {"Measure": "Versus VNINDEX", "Observed value": f"{row.excess_return_1d * 100:+.2f} percentage points"},
        {"Measure": "Isolation Forest anomaly score", "Observed value": f"{row.anomaly_score:.4f}"},
    ])
    st.dataframe(detail, width="stretch", hide_index=True)
    st.info("Analyst question: does this combined return, volume, volatility, and market-relative pattern reflect a genuine market event or a data adjustment? Check source history and public disclosures before drawing a conclusion.")


def render_event_followup(row: pd.Series, analysis: AnalysisBundle) -> pd.Series | None:
    matches = analysis.outcomes.loc[(analysis.outcomes.symbol == row.symbol) &
                                    (analysis.outcomes.date == row.date)]
    if len(matches) != 1:
        st.warning("Saved follow-up for this event is unavailable. Regenerate the analysis files.")
        return None
    outcome = matches.iloc[0]
    st.subheader("What happened afterward?")
    st.caption("Retrospective outcomes use later observed sessions of the same stock. They were never used in scoring or flag selection.")
    cols = st.columns(3)
    next_1 = outcome.next_session_return
    next_5 = outcome.next_five_session_return
    cols[0].metric("Next observed-session return", f"{next_1:+.2%}" if pd.notna(next_1) else "Unavailable")
    cols[1].metric("Next five-session return", f"{next_5:+.2%}" if pd.notna(next_5) else "Unavailable")
    cols[2].metric("Absolute five-session move", f"{abs(next_5):.2%}" if pd.notna(next_5) else "Unavailable")
    if pd.isna(next_5):
        st.caption("This event lacks a full five-observed-session future window and is excluded from five-session comparisons.")
    return outcome


def render_brief_download(row: pd.Series, outcome: pd.Series | None) -> None:
    note = research_brief(row, outcome, source="Vnstock historical OHLCV via validated local equity files")
    st.download_button("Download research brief (Markdown)", data=note,
                       file_name=f"market_radar_{row.symbol}_{row.date}.md", mime="text/markdown")


def render_event_study(analysis: AnalysisBundle) -> None:
    """Display saved descriptive holdout cohorts and sufficiently sized subgroups."""
    summary = analysis.comparisons
    cohorts = summary["cohorts"]
    st.subheader("Retrospective five-session event study")
    st.caption(f"{summary['complete_five_session_observations']:,} of {summary['holdout_scored_observations']:,} holdout rows have a full subsequent five-observed-session window; {summary['excluded_without_full_window']:,} are excluded.")
    names = {"flagged": "Flagged", "unflagged": "Unflagged", "model_only": "Model only",
             "model_and_baseline": "Model + baseline"}
    rows = []
    for key, label in names.items():
        stats = cohorts[key]
        rows.append({"Group": label, "Observations": stats["observations"],
                     "Mean absolute return (%)": stats["mean_abs_return_pct"],
                     "Median absolute return (%)": stats["median_abs_return_pct"],
                     "Mean signed return (%)": stats["mean_signed_return_pct"],
                     "Positive (%)": stats["positive_pct"], "Negative (%)": stats["negative_pct"]})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    differences = summary["mean_absolute_return_differences_pp"]
    st.caption("Mean absolute move differences: flagged minus unflagged "
               f"{differences['flagged_minus_unflagged']:+.2f} percentage points; model-only minus model + baseline "
               f"{differences['model_only_minus_model_and_baseline']:+.2f} percentage points." if all(
                   value is not None for value in differences.values()) else
               "At least one comparison group has no complete future window.")
    st.plotly_chart(event_study_box(analysis.outcomes), width="stretch")
    available = [item for item in summary["subgroups"] if item["comparison_available"]]
    if available:
        st.caption(f"Stock/sector comparisons require at least {summary['minimum_subgroup_size']} flagged and {summary['minimum_subgroup_size']} unflagged complete windows in a group.")
        st.dataframe(pd.DataFrame([{
            "Type": item["dimension"].title(), "Covered name": item["name"],
            "Flagged n": item["flagged"]["observations"],
            "Unflagged n": item["unflagged"]["observations"],
            "Flagged mean abs (%)": item["flagged"]["mean_abs_return_pct"],
            "Unflagged mean abs (%)": item["unflagged"]["mean_abs_return_pct"],
        } for item in available]), width="stretch", hide_index=True)
    else:
        st.caption("No individual stock or covered sector has enough complete flagged and unflagged windows for the minimum subgroup comparison.")
    st.warning("These are historical descriptions of a small, selected stock sample. Group differences do not establish predictive skill, causality, statistical significance, or trading profitability.")
