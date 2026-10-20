"""Known-value checks for sector coverage and retrospective research outputs."""

from dataclasses import replace
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from market_radar.analysis.research import (
    diagnose_symbols, future_outcomes, holdout_comparisons, outcome_summary,
    research_brief, sector_monthly, sector_summary,
)
from market_radar.analysis.storage import AnalysisBundle, AnalysisDataError, load_analysis, save_analysis
from market_radar.config import PROJECT_ROOT
from market_radar.visualization.data import DashboardData, filter_scored
from market_radar.visualization.research_charts import normalized_stock_chart, sector_heatmap


@pytest.fixture
def research_rows() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=12).strftime("%Y-%m-%d")
    rows = []
    for day, trading_date in enumerate(dates):
        for symbol, sector, offset in (("FPT", "Technology", 0), ("CTG", "Banking", 1)):
            if symbol == "CTG" and day < 2:
                continue
            close = 100 + 10 * day if symbol == "FPT" else 50 + 2 * day
            ready = day > (0 if symbol == "FPT" else 2)
            flagged = (symbol, day) in {("FPT", 4), ("CTG", 4)}
            volume_z = 4.2 if (symbol, day) == ("CTG", 4) else 2.0
            return_z = 2.1
            rows.append({
                "date": trading_date, "symbol": symbol, "sector": sector, "asset_type": "equity",
                "open": close, "high": close + 1, "low": close - 1, "close": close,
                "volume": 1000 + day * 100, "price_unit": "thousand_vnd",
                "model_ready": ready,
                "partition": "not_model_ready" if not ready else ("train" if day < 4 else "holdout"),
                "anomaly_flag": flagged if ready else pd.NA,
                "anomaly_score": 0.04 if flagged else (-0.1 if ready else np.nan),
                "baseline_volume_flag": volume_z > 3 if ready else pd.NA,
                "baseline_return_flag": False if ready else pd.NA,
                "return_1d": 0.02 + day * 0.001,
                "return_5d": 0.05,
                "volume_ratio_prev20": 1.3 + offset * 0.2,
                "volume_zscore_prev20": volume_z,
                "return_zscore_prev20": return_z,
                "return_volatility_prev20": 0.025,
                "vnindex_return_1d": 0.01,
                "excess_return_1d": 0.01 + day * 0.001,
                "anomaly_explanation": "Heuristic context: observed return and volume were unusual." if flagged else "",
            })
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def test_sector_coverage_denominators_and_unequal_histories(research_rows):
    summary = sector_summary(research_rows).set_index("sector")
    assert summary.loc["Technology", "covered_stocks"] == 1
    assert summary.loc["Banking", "covered_stocks"] == 1
    assert summary.loc["Technology", "eligible_observations"] == 11
    assert summary.loc["Banking", "eligible_observations"] == 9
    assert summary.loc["Technology", "anomaly_rate_pct"] == pytest.approx(100 / 11)
    assert summary.loc["Banking", "anomaly_rate_pct"] == pytest.approx(100 / 9)
    assert summary.eligible_observations.sum() == int(research_rows.model_ready.sum())
    assert summary.anomalies.sum() == 2
    assert summary.loc["Banking", "mean_volume_ratio"] == pytest.approx(1.5)
    monthly = sector_monthly(research_rows)
    assert monthly.eligible_observations.sum() == 20
    assert monthly.anomalies.sum() == 2
    assert set(monthly.partition) == {"train", "holdout"}
    assert sector_summary(research_rows.iloc[:0]).empty


def test_future_returns_use_own_stock_sessions_and_exclude_short_windows(research_rows):
    outcomes = future_outcomes(research_rows)
    fpt = outcomes.query("symbol == 'FPT' and date == '2026-01-09'").iloc[0]
    assert fpt.next_session_return == pytest.approx(150 / 140 - 1)
    assert fpt.next_five_session_return == pytest.approx(190 / 140 - 1)
    assert fpt.next_five_session_abs_return == pytest.approx(abs(190 / 140 - 1))
    ctg = outcomes.query("symbol == 'CTG' and date == '2026-01-09'").iloc[0]
    assert ctg.next_five_session_return == pytest.approx(68 / 58 - 1)
    last_fpt = outcomes[outcomes.symbol == "FPT"].iloc[-1]
    assert pd.isna(last_fpt.next_session_return)
    assert pd.isna(last_fpt.next_five_session_return)
    assert outcomes.query("symbol == 'FPT'").next_five_session_return.notna().sum() == 6
    original = research_rows.copy(deep=True)
    future_outcomes(research_rows)
    pd.testing.assert_frame_equal(research_rows, original)
    assert not {"next_session_return", "next_five_session_return"} & set(research_rows.columns)


def test_holdout_cohorts_model_only_and_empty_samples(research_rows):
    outcomes = future_outcomes(research_rows)
    comparison = holdout_comparisons(outcomes, min_group_size=2)
    assert comparison["holdout_scored_observations"] == 16
    assert comparison["complete_five_session_observations"] == 6
    assert comparison["excluded_without_full_window"] == 10
    assert comparison["cohorts"]["flagged"]["observations"] == 2
    assert comparison["cohorts"]["unflagged"]["observations"] == 4
    assert comparison["cohorts"]["model_only"]["observations"] == 1
    assert comparison["cohorts"]["model_and_baseline"]["observations"] == 1
    assert comparison["cohorts"]["flagged"]["positive_pct"] == 100
    assert comparison["cohorts"]["flagged"]["negative_pct"] == 0
    assert outcome_summary(outcomes.iloc[:0])["mean_abs_return_pct"] is None
    with pytest.raises(ValueError, match="positive"):
        holdout_comparisons(outcomes, min_group_size=0)


def test_model_only_baseline_rules_and_brief_values(research_rows):
    outcomes = future_outcomes(research_rows)
    holdout = filter_scored(research_rows, partitions=["holdout"], model_only=True)
    assert len(holdout) == 1 and holdout.iloc[0].symbol == "FPT"
    assert not holdout.iloc[0].baseline_volume_flag and not holdout.iloc[0].baseline_return_flag
    row = holdout.iloc[0]
    outcome = outcomes.query("symbol == 'FPT' and date == '2026-01-09'").iloc[0]
    note = research_brief(row, outcome, source="Synthetic test source")
    assert "FPT · 2026-01-09" in note
    assert "Model-only anomaly" in note
    assert "0.0400" in note
    assert "1.30×" in note
    assert "+35.71%" in note
    assert "Synthetic test source" in note
    assert "not a probability" in note
    assert "not a verified cause" in note
    assert "buy" not in note.lower()
    with pytest.raises(ValueError, match="flagged"):
        research_brief(research_rows.query("symbol == 'FPT' and date == '2026-01-12'").iloc[0], None)


def test_diagnostics_and_charts_have_real_date_alignment(research_rows):
    diagnostics = diagnose_symbols(research_rows, ("FPT", "ABSENT"))
    assert diagnostics["FPT"]["ohlc_inconsistent_rows"] == 0
    assert diagnostics["FPT"]["nonpositive_or_nonfinite_price_rows"] == 0
    assert diagnostics["FPT"]["negative_or_nonfinite_volume_rows"] == 0
    assert diagnostics["FPT"]["partitions"]["holdout"]["anomalies"] == 1
    assert diagnostics["ABSENT"]["status"] == "not_in_dataset"
    chart = normalized_stock_chart(research_rows[research_rows.sector == "Technology"], ["holdout"])
    line = next(trace for trace in chart.data if trace.name == "FPT")
    marker = next(trace for trace in chart.data if trace.name == "FPT anomaly")
    assert line.y[0] == pytest.approx(100)
    assert list(marker.x) == ["2026-01-09"]
    assert list(marker.y) == [pytest.approx(140)]
    heatmap = sector_heatmap(sector_monthly(research_rows))
    assert len(heatmap.data) == 1
    assert heatmap.data[0].z.shape[0] == 2


def test_analysis_save_load_and_stale_or_malformed_files(research_rows, tmp_path):
    source = tmp_path / "market_anomalies.csv"
    research_rows.to_csv(source, index=False)
    output = tmp_path / "analysis"
    paths = save_analysis(research_rows, source, output, feature_sha256="abc")
    loaded = load_analysis(output, source, research_rows)
    assert len(loaded.outcomes) == int(research_rows.model_ready.sum())
    assert loaded.metadata["parameters"]["future_horizons_observed_sessions"] == [1, 5]
    paths["monthly"].unlink()
    with pytest.raises(AnalysisDataError, match="Missing analysis inputs"):
        load_analysis(output, source, research_rows)
    paths = save_analysis(research_rows, source, output, feature_sha256="abc")
    source.write_text(source.read_text() + "\n")
    with pytest.raises(AnalysisDataError, match="stale"):
        load_analysis(output, source, research_rows)
    source.write_text(source.read_text().rstrip("\n") + "\n")
    paths = save_analysis(research_rows, source, output, feature_sha256="abc")
    malformed = pd.read_csv(paths["sectors"]).drop(columns=["eligible_observations"])
    malformed.to_csv(paths["sectors"], index=False)
    with pytest.raises(AnalysisDataError, match="lacks columns"):
        load_analysis(output, source, research_rows)
    paths = save_analysis(research_rows, source, output, feature_sha256="abc")
    paths["metadata"].write_text("[]", encoding="utf-8")
    with pytest.raises(AnalysisDataError, match="Malformed analysis metadata"):
        load_analysis(output, source, research_rows)


def test_sector_page_and_research_components_in_apptest(research_rows, tmp_path):
    source = tmp_path / "market_anomalies.csv"
    research_rows.to_csv(source, index=False)
    paths = save_analysis(research_rows, source, tmp_path / "analysis", feature_sha256="abc")
    bundle = load_analysis(tmp_path / "analysis", source, research_rows)
    data = DashboardData(research_rows, pd.DataFrame(), {}, {})
    with patch("market_radar.visualization.pages._load_default", return_value=data), \
         patch("market_radar.visualization.pages._load_analysis", return_value=bundle):
        app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=25).run()
        app.radio[0].set_value("Sector Intelligence").run()
        assert not app.exception
        assert any("Covered-stock summary" in x.value for x in app.subheader)
        app.radio[0].set_value("Anomaly Investigation").run()
        assert not app.exception
        picker = next(x for x in app.selectbox if x.label == "Select an event")
        picker.set_value(picker.options[0]).run()
        assert any("What happened afterward" in x.value for x in app.subheader)
        mode = next(x for x in app.checkbox if "Model-only events" in x.label)
        mode.set_value(True).run()
        assert not app.exception
        picker = next(x for x in app.selectbox if x.label == "Select an event")
        assert picker.value is None
        picker.set_value(picker.options[0]).run()
        assert any("Model-only pattern review" in x.value for x in app.subheader)


def test_dashboard_keeps_existing_views_when_analysis_is_missing(research_rows):
    data = DashboardData(research_rows, pd.DataFrame(), {}, {})
    with patch("market_radar.visualization.pages._load_default", return_value=data), \
         patch("market_radar.visualization.pages._load_analysis",
               side_effect=AnalysisDataError("Missing analysis inputs")):
        app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=25).run()
        app.radio[0].set_value("Sector Intelligence").run()
        assert not app.exception
        assert any("build_analysis" in item.value for item in app.info)
        app.radio[0].set_value("Anomaly Investigation").run()
        assert not app.exception
        picker = next(x for x in app.selectbox if x.label == "Select an event")
        picker.set_value(picker.options[0]).run()
        assert not app.exception
        assert any("build_analysis" in item.value for item in app.info)
