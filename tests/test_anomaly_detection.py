"""Offline behavior checks for the pooled Isolation Forest workflow."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from market_radar.detect_anomalies import main
from market_radar.features import MODEL_FEATURE_COLUMNS
from market_radar.modeling.detection import (
    DetectionConfig, chronological_split, detect_anomalies, explain_anomaly, validate_feature_table,
)
from market_radar.modeling.evaluation import evaluate_run
from market_radar.modeling.persistence import load_trusted_model, save_run, score_with_model


def sample_features() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2026-01-05", periods=12).strftime("%Y-%m-%d")
    for day, date in enumerate(dates):
        for symbol, sector, offset in (("FPT", "Technology", 0), ("CTG", "Banking", 1)):
            if symbol == "CTG" and day in (2, 10):
                continue  # Symbols may have different histories.
            close = 100 + day + offset * 10
            row = {"date": date, "symbol": symbol, "asset_type": "equity", "sector": sector,
                   "open": close - 0.5, "high": close + 1, "low": close - 1,
                   "close": close, "volume": 1000 + day * 25 + offset * 30,
                   "price_unit": "thousand_vnd", "model_ready": day != 0}
            row.update({name: 0.1 * (day + 1) + offset * 0.03 + j * 0.01
                        for j, name in enumerate(MODEL_FEATURE_COLUMNS)})
            if day == 11:
                row["return_1d"] = 0.07
                row["volume_ratio_prev20"] = 4.2
                row["volume_zscore_prev20"] = 4.1
                row["return_zscore_prev20"] = 3.2
            if day == 0:
                row["return_1d"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows).sample(frac=1, random_state=9).reset_index(drop=True)


def config() -> DetectionConfig:
    return DetectionConfig(n_estimators=32, contamination=0.1, random_state=42, n_jobs=1)


def test_date_split_ready_selection_and_training_only_scaler():
    frame = sample_features()
    run = detect_anomalies(frame, config())
    result = run.results
    assert len(result) == len(frame)
    assert int(result["model_ready"].sum()) == len(run.split.train_indices) + len(run.split.holdout_indices)
    assert result.loc[~result.model_ready, "anomaly_score"].isna().all()
    assert result.loc[~result.model_ready, "anomaly_flag"].isna().all()
    assert set(result.loc[~result.model_ready, "scoring_status"]) == {"insufficient_features"}
    assert max(run.split.train_dates) < min(run.split.holdout_dates)
    assert result.groupby("date")["partition"].nunique().max() == 1
    expected = frame.loc[run.split.train_indices, list(MODEL_FEATURE_COLUMNS)].mean().to_numpy()
    np.testing.assert_allclose(run.model.named_steps["scaler"].mean_, expected)
    assert not np.allclose(expected, frame.loc[frame.model_ready, list(MODEL_FEATURE_COLUMNS)].mean())


def test_future_rows_do_not_affect_fitted_model_or_earlier_scores():
    frame = sample_features()
    first = detect_anomalies(frame, config())
    changed = frame.copy()
    changed.loc[changed.date.isin(first.split.holdout_dates), "volume_ratio_prev20"] += 10000
    second = detect_anomalies(changed, config())
    np.testing.assert_allclose(first.model.named_steps["scaler"].mean_, second.model.named_steps["scaler"].mean_)
    earlier = first.results[first.results.partition == "train"].anomaly_score.to_numpy()
    changed_earlier = second.results[second.results.partition == "train"].anomaly_score.to_numpy()
    np.testing.assert_allclose(earlier, changed_earlier)


def test_seed_scores_boundary_and_deterministic_explanations():
    frame = sample_features()
    first = detect_anomalies(frame, config())
    second = detect_anomalies(frame, config())
    pd.testing.assert_frame_equal(first.results, second.results)
    scored = first.results[first.results.model_ready]
    features = scored.loc[:, list(MODEL_FEATURE_COLUMNS)]
    np.testing.assert_allclose(scored.anomaly_score, -first.model.decision_function(features))
    assert scored.anomaly_flag.tolist() == (first.model.predict(features) == -1).tolist()
    assert (scored.loc[scored.anomaly_flag, "anomaly_score"] > 0).all()
    assert (scored.loc[~scored.anomaly_flag, "anomaly_score"] <= 0).all()
    row = scored.iloc[-1].copy()
    row["volume_ratio_prev20"] = 3.7
    row["return_1d"] = -0.041
    row["excess_return_1d"] = -0.04
    explanation = explain_anomaly(row)
    assert "3.7×" in explanation and "-4.1%" in explanation
    assert "not model feature attribution" in explanation
    row["volume_ratio_prev20"] = np.nan
    assert explain_anomaly(row) == ""


def test_diagnostics_baselines_groups_and_event_study():
    run = detect_anomalies(sample_features(), config())
    results = run.results.copy()
    holdout = results.index[results.partition == "holdout"]
    results.loc[holdout, "anomaly_flag"] = False
    results.loc[holdout, "baseline_volume_flag"] = False
    results.loc[holdout, "baseline_return_flag"] = False
    results.loc[holdout[:3], "anomaly_flag"] = True
    results.loc[holdout[0], "baseline_volume_flag"] = True
    results.loc[holdout[1], "baseline_return_flag"] = True
    results.loc[holdout[3], "baseline_volume_flag"] = True
    summary = evaluate_run(replace(run, results=results))
    baseline = summary["holdout_baselines"]
    assert baseline["model_anomalies"] == 3
    assert baseline["model_and_either"] == 2
    assert baseline["model_only"] == 1
    assert baseline["either_baseline"] == 3
    assert set(summary["by_stock"]) == {"CTG", "FPT"}
    assert set(summary["by_sector"]) == {"Banking", "Technology"}
    assert sum(item["observations"] for item in summary["by_stock"].values()) == summary["all_scored"]["observations"]
    study = summary["five_session_event_study"]
    assert study["excluded_without_full_window"] >= 5
    assert study["eligible_holdout_observations"] + study["excluded_without_full_window"] == len(holdout)


def test_save_load_order_and_hash(tmp_path):
    frame = sample_features()
    source = tmp_path / "features.csv"
    frame.to_csv(source, index=False)
    run = detect_anomalies(pd.read_csv(source), config())
    paths = save_run(run, evaluate_run(run), source, tmp_path / "artifacts", tmp_path / "outputs")
    model, metadata = load_trusted_model(paths["model"], paths["model_metadata"])
    assert metadata["feature_names"] == list(MODEL_FEATURE_COLUMNS)
    assert metadata["training_samples"] == len(run.split.train_indices)
    shuffled = frame[frame.columns[::-1]]
    inferred = score_with_model(model, metadata, shuffled).sort_values(["date", "symbol"])
    expected = run.results[run.results.model_ready].sort_values(["date", "symbol"])
    np.testing.assert_allclose(inferred.anomaly_score, expected.anomaly_score)
    assert inferred.anomaly_flag.tolist() == expected.anomaly_flag.tolist()
    saved = pd.read_csv(paths["results"])
    assert len(saved) == len(frame)
    assert json.loads(paths["evaluation"].read_text())["all_scored"]["observations"] == int(frame.model_ready.sum())
    bad = dict(metadata)
    bad["feature_names"] = list(reversed(MODEL_FEATURE_COLUMNS))
    paths["model_metadata"].write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="feature names or ordering"):
        load_trusted_model(paths["model"], paths["model_metadata"])


@pytest.mark.parametrize("change, error", [
    (lambda f: f.drop(columns=["volume_ratio_prev20"]), "missing columns"),
    (lambda f: f.assign(model_ready="yes"), "must be boolean"),
    (lambda f: f.assign(date="bad"), "YYYY-MM-DD"),
    (lambda f: pd.concat([f, f.iloc[[0]]]), "Duplicate"),
    (lambda f: f.assign(asset_type="index"), "equity"),
    (lambda f: f.assign(close=-1), "prices must be positive"),
    (lambda f: f.assign(volume=np.nan), "must be finite"),
])
def test_invalid_inputs(change, error):
    with pytest.raises(ValueError, match=error):
        validate_feature_table(change(sample_features()))


def test_missing_ready_value_and_insufficient_dates():
    frame = sample_features()
    frame.loc[frame.model_ready.idxmax(), "return_1d"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        detect_anomalies(frame, config())
    with pytest.raises(ValueError, match="two model-ready trading dates"):
        chronological_split(sample_features().query("date == '2026-01-06'"))


def test_cli_with_synthetic_input(tmp_path, capsys):
    source = tmp_path / "features.csv"
    sample_features().to_csv(source, index=False)
    assert main(["--input", str(source), "--artifact-dir", str(tmp_path / "artifacts"),
                 "--output-dir", str(tmp_path / "outputs"), "--n-estimators", "32"]) == 0
    assert "Anomalies:" in capsys.readouterr().out
    assert (tmp_path / "outputs" / "market_anomalies.csv").exists()
    assert (tmp_path / "artifacts" / "isolation_forest.joblib").exists()
