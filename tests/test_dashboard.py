"""Offline dashboard data, chart, and Streamlit interaction checks."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from market_radar.config import PROJECT_ROOT
from market_radar.features import MODEL_FEATURE_COLUMNS
from market_radar.modeling.detection import DetectionConfig, detect_anomalies
from market_radar.modeling.evaluation import evaluate_run
from market_radar.modeling.persistence import save_run
from market_radar.visualization.charts import price_volume_chart
from market_radar.visualization.data import (
    DashboardDataError, baseline_overlap, event_context, filter_scored, frequency,
    grouped_frequency, load_dashboard_data,
)


@pytest.fixture
def saved_data(tmp_path):
    dates = pd.bdate_range("2026-01-05", periods=14).strftime("%Y-%m-%d")
    rows = []
    for day, trading_date in enumerate(dates):
        for symbol, sector, offset in (("FPT", "Technology", 0), ("CTG", "Banking", 1)):
            close = 50 + day + 5 * offset
            row = {"date": trading_date, "symbol": symbol, "asset_type": "equity",
                   "sector": sector, "open": close, "high": close + 1,
                   "low": close - 1, "close": close, "volume": 1000 + 100 * day,
                   "price_unit": "thousand_vnd", "model_ready": day != 0}
            row.update({name: day * 0.02 + offset * 0.1 + i * 0.01
                        for i, name in enumerate(MODEL_FEATURE_COLUMNS)})
            if day == 0:
                row["return_1d"] = np.nan
            if day == 12 and symbol == "FPT":
                row["volume_zscore_prev20"] = 4.5
                row["return_zscore_prev20"] = -4.1
            rows.append(row)
    frame = pd.DataFrame(rows)
    feature_path = tmp_path / "features.csv"
    frame.to_csv(feature_path, index=False)
    run = detect_anomalies(pd.read_csv(feature_path), DetectionConfig(
        n_estimators=24, contamination=0.15, random_state=42, n_jobs=1))
    paths = save_run(run, evaluate_run(run), feature_path, tmp_path / "artifacts", tmp_path / "outputs")
    loaded = load_dashboard_data(paths["results"], paths["evaluation"],
                                 feature_path, paths["model_metadata"])
    return loaded, paths, feature_path


def test_loading_and_missing_file_messages(saved_data, tmp_path):
    data, paths, feature_path = saved_data
    assert len(data.results) == 28
    assert data.results.model_ready.sum() == 26
    assert data.model_metadata["feature_names"] == list(MODEL_FEATURE_COLUMNS)
    with pytest.raises(DashboardDataError, match="Missing dashboard inputs"):
        load_dashboard_data(tmp_path / "absent.csv", paths["evaluation"],
                            feature_path, paths["model_metadata"])


def test_validation_catches_malformed_and_stale_outputs(saved_data, tmp_path):
    _, paths, feature_path = saved_data
    bad_results = pd.read_csv(paths["results"])
    bad_results = bad_results.drop(columns=["anomaly_score"])
    bad_path = tmp_path / "bad.csv"
    bad_results.to_csv(bad_path, index=False)
    with pytest.raises(DashboardDataError, match="missing columns"):
        load_dashboard_data(bad_path, paths["evaluation"], feature_path, paths["model_metadata"])
    feature_path.write_text(feature_path.read_text() + "\n")
    with pytest.raises(DashboardDataError, match="hash differs"):
        load_dashboard_data(paths["results"], paths["evaluation"],
                            feature_path, paths["model_metadata"])


def test_filters_counts_rates_and_partitions(saved_data):
    data, _, _ = saved_data
    scored = filter_scored(data.results)
    assert len(scored) == 26
    assert len(filter_scored(data.results, partitions=["train"])) == 18
    assert len(filter_scored(data.results, partitions=["holdout"])) == 8
    fpt = filter_scored(data.results, symbols=["FPT"])
    assert len(fpt) == 13
    assert len(filter_scored(data.results, sectors=["Technology"])) == 13
    assert len(filter_scored(data.results, symbols=["FPT"], sectors=["Banking"])) == 0
    assert filter_scored(data.results, symbols=[]).empty
    assert filter_scored(data.results, start="2099-01-01").empty
    day = scored.date.min()
    assert len(filter_scored(data.results, start=day, end=day)) == 2
    flagged = int(scored.anomaly_flag.sum())
    stats = frequency(scored)
    assert stats["observations"] == 26 and stats["anomalies"] == flagged
    assert stats["rate_pct"] == pytest.approx(100 * flagged / 26)
    assert frequency(filter_scored(data.results, symbols=[]))["rate_pct"] == 0
    stock_groups = grouped_frequency(scored, "symbol")
    sector_groups = grouped_frequency(scored, "sector")
    assert int(stock_groups.observations.sum()) == 26
    assert int(sector_groups.anomalies.sum()) == flagged


def test_baselines_match_fixed_rules_and_model_only_filter(saved_data):
    data, _, _ = saved_data
    holdout = filter_scored(data.results, partitions=["holdout"])
    overlap = baseline_overlap(holdout)
    volume = holdout.volume_zscore_prev20 > 3
    returns = holdout.return_zscore_prev20.abs() > 3
    model = holdout.anomaly_flag
    assert overlap["volume"] == int(volume.sum())
    assert overlap["return"] == int(returns.sum())
    assert overlap["model_and_either"] == int((model & (volume | returns)).sum())
    only = filter_scored(data.results, partitions=["holdout"], model_only=True)
    assert len(only) == overlap["model_only"]
    assert not (only.baseline_volume_flag | only.baseline_return_flag).any()


def test_price_chart_marker_alignment_and_symbol_context(saved_data):
    data, _, _ = saved_data
    stock = data.results[data.results.symbol == "FPT"]
    chart = price_volume_chart(stock)
    close_trace = next(trace for trace in chart.data if trace.name == "Close")
    assert list(close_trace.x) == stock.sort_values("date").date.tolist()
    assert list(close_trace.y) == stock.sort_values("date").close.tolist()
    for label, partition in (("Training anomaly", "train"), ("Holdout anomaly", "holdout")):
        trace = next(trace for trace in chart.data if trace.name == label)
        expected = stock[(stock.partition == partition) & stock.anomaly_flag.fillna(False)]
        assert list(trace.x) == expected.date.tolist()
        assert list(trace.y) == expected.close.tolist()
    target = stock.iloc[7]
    context = event_context(data.results, "FPT", target.date, radius=2)
    assert len(context) == 5 and context.symbol.eq("FPT").all()
    with pytest.raises(DashboardDataError, match="absent"):
        event_context(data.results, "FPT", "1900-01-01")


def test_streamlit_pages_and_empty_filter_without_network(saved_data):
    data, _, _ = saved_data
    with patch("market_radar.visualization.pages._load_default", return_value=data):
        app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=25).run()
        assert not app.exception
        assert [m.value for m in app.metric if m.label == "Eligible stock-sessions"] == ["26"]
        for section in ("Stock Explorer", "Anomaly Investigation", "Model & Methodology"):
            app.radio[0].set_value(section).run()
            assert not app.exception, section
        app.radio[0].set_value("Anomaly Investigation").run()
        stocks = next(widget for widget in app.multiselect if widget.label == "Stocks")
        stocks.set_value([]).run()
        assert not app.exception
        assert any("No anomalies match" in item.value for item in app.info)


def test_streamlit_missing_input_state():
    with patch("market_radar.visualization.pages._load_default",
               side_effect=DashboardDataError("Missing dashboard inputs: test.csv")):
        app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=20).run()
    assert not app.exception
    assert any("Missing dashboard inputs" in item.value for item in app.error)
    assert any("build_features" in item.value for item in app.info)
