"""Descriptive diagnostics for unlabeled anomaly detections."""

import pandas as pd

from market_radar.modeling.detection import DetectionRun


def _frequency(frame: pd.DataFrame) -> dict[str, float | int]:
    count = int(len(frame))
    anomalies = int(frame["anomaly_flag"].sum())
    return {"observations": count, "anomalies": anomalies,
            "anomaly_pct": round(100 * anomalies / count, 3) if count else 0.0}


def _by_group(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float | int]]:
    return {str(value): _frequency(group) for value, group in frame.groupby(column, sort=True)}


def _score_distribution(frame: pd.DataFrame) -> dict[str, float] | None:
    if frame.empty:
        return None
    scores = frame["anomaly_score"]
    return {key: float(value) for key, value in {
        "min": scores.min(), "median": scores.median(), "p90": scores.quantile(0.9),
        "p95": scores.quantile(0.95), "max": scores.max(),
    }.items()}


def _holdout_baselines(frame: pd.DataFrame) -> dict[str, int]:
    model = frame["anomaly_flag"].astype(bool)
    volume = frame["baseline_volume_flag"].astype(bool)
    returns = frame["baseline_return_flag"].astype(bool)
    either = volume | returns
    return {
        "holdout_observations": int(len(frame)),
        "model_anomalies": int(model.sum()),
        "volume_z_gt_3": int(volume.sum()),
        "absolute_return_z_gt_3": int(returns.sum()),
        "either_baseline": int(either.sum()),
        "model_and_volume": int((model & volume).sum()),
        "model_and_return": int((model & returns).sum()),
        "model_and_either": int((model & either).sum()),
        "model_only": int((model & ~either).sum()),
    }


def _event_study(results: pd.DataFrame) -> dict[str, object]:
    """Use later closes for description only; never feed them into the model."""
    ordered = results.sort_values(["symbol", "date"]).copy()
    ordered["forward_close_5"] = ordered.groupby("symbol")["close"].shift(-5)
    eligible = ordered.loc[
        (ordered["partition"] == "holdout") & ordered["forward_close_5"].notna()
    ].copy()
    eligible["forward_abs_return_5"] = (eligible["forward_close_5"] / eligible["close"] - 1).abs()
    groups = {}
    for name, mask in (("flagged", eligible["anomaly_flag"].astype(bool)),
                       ("unflagged", ~eligible["anomaly_flag"].astype(bool))):
        values = eligible.loc[mask, "forward_abs_return_5"]
        groups[name] = {"observations": int(len(values)),
                        "mean_absolute_return_pct": round(float(values.mean() * 100), 3) if len(values) else None,
                        "median_absolute_return_pct": round(float(values.median() * 100), 3) if len(values) else None}
    return {
        "horizon": "five subsequent observed sessions of the same symbol",
        "eligible_holdout_observations": int(len(eligible)),
        "excluded_without_full_window": int((results["partition"] == "holdout").sum() - len(eligible)),
        "groups": groups,
        "limitation": "Retrospective description only; no labels, causal or predictive inference."
    }


def evaluate_run(run: DetectionRun, top_n: int = 10) -> dict[str, object]:
    """Summarize rates, baseline overlap, and top events without accuracy claims."""
    scored = run.results.loc[run.results["scoring_status"] == "scored"]
    train = scored.loc[scored["partition"] == "train"]
    holdout = scored.loc[scored["partition"] == "holdout"]
    top = scored.loc[scored["anomaly_flag"].astype(bool)].nlargest(top_n, "anomaly_score")
    event_columns = ["date", "symbol", "sector", "partition", "anomaly_score",
                     "return_1d", "return_5d", "volume_ratio_prev20", "volume_zscore_prev20",
                     "return_zscore_prev20", "excess_return_1d", "intraday_range_pct",
                     "anomaly_explanation"]
    return {
        "evaluation_type": "descriptive single chronological holdout; no ground-truth anomaly labels",
        "all_scored": _frequency(scored),
        "not_model_ready": int((run.results["scoring_status"] != "scored").sum()),
        "by_partition": {"train": _frequency(train), "holdout": _frequency(holdout)},
        "by_stock": _by_group(scored, "symbol"),
        "by_sector": _by_group(scored, "sector"),
        "score_distribution": {"all": _score_distribution(scored), "train": _score_distribution(train),
                               "holdout": _score_distribution(holdout)},
        "holdout_baselines": _holdout_baselines(holdout),
        "top_events": top[event_columns].to_dict(orient="records"),
        "five_session_event_study": _event_study(run.results),
    }
