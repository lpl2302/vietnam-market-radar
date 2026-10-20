"""Descriptive sector and retrospective analyses; never used for model scoring."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _scored(results: pd.DataFrame) -> pd.DataFrame:
    return results.loc[results["model_ready"]].copy()


def sector_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize only the covered stocks, using ready rows for rates and features."""
    columns = ["sector", "covered_stocks", "symbols", "eligible_observations", "anomalies",
               "anomaly_rate_pct", "mean_daily_return_pct", "daily_return_volatility_pct",
               "mean_volume_ratio", "latest_observation_date"]
    if results.empty:
        return pd.DataFrame(columns=columns)
    ready = _scored(results)
    rows = []
    for sector, all_rows in results.groupby("sector", sort=True):
        sample = ready.loc[ready.sector == sector]
        count = len(sample)
        flags = int(sample.anomaly_flag.sum()) if count else 0
        rows.append({
            "sector": sector, "covered_stocks": int(all_rows.symbol.nunique()),
            "symbols": ", ".join(sorted(all_rows.symbol.unique())),
            "eligible_observations": count, "anomalies": flags,
            "anomaly_rate_pct": 100 * flags / count if count else 0.0,
            "mean_daily_return_pct": 100 * sample.return_1d.mean() if count else np.nan,
            "daily_return_volatility_pct": 100 * sample.return_1d.std(ddof=1) if count > 1 else np.nan,
            "mean_volume_ratio": sample.volume_ratio_prev20.mean() if count else np.nan,
            "latest_observation_date": all_rows.date.max(),
        })
    return pd.DataFrame(rows, columns=columns).sort_values("sector").reset_index(drop=True)


def sector_monthly(results: pd.DataFrame) -> pd.DataFrame:
    """Month/sector/partition frequencies with eligible stock-session denominators."""
    columns = ["month", "sector", "partition", "eligible_observations", "anomalies", "anomaly_rate_pct"]
    ready = _scored(results)
    if ready.empty:
        return pd.DataFrame(columns=columns)
    ready["month"] = ready.date.str.slice(0, 7)
    grouped = ready.groupby(["month", "sector", "partition"], sort=True).anomaly_flag.agg(
        eligible_observations="size", anomalies="sum").reset_index()
    grouped["anomalies"] = grouped.anomalies.astype(int)
    grouped["anomaly_rate_pct"] = 100 * grouped.anomalies / grouped.eligible_observations
    return grouped.loc[:, columns]


def future_outcomes(results: pd.DataFrame) -> pd.DataFrame:
    """Use subsequent observed sessions within each symbol; preserve model outputs."""
    required = {"date", "symbol", "sector", "close", "model_ready", "partition", "anomaly_flag",
                "baseline_volume_flag", "baseline_return_flag", "anomaly_score"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results are missing columns for event study: {sorted(missing)}")
    if results.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate symbol/date observations")
    ordered = results.sort_values(["symbol", "date"]).copy()
    close = pd.to_numeric(ordered.close, errors="coerce")
    if not np.isfinite(close.to_numpy(dtype=float)).all() or (close <= 0).any():
        raise ValueError("Event study needs finite, positive closing prices")
    ordered["next_session_date"] = ordered.groupby("symbol").date.shift(-1)
    ordered["next_five_session_date"] = ordered.groupby("symbol").date.shift(-5)
    next_1 = ordered.groupby("symbol").close.shift(-1)
    next_5 = ordered.groupby("symbol").close.shift(-5)
    ordered["next_session_return"] = next_1 / close - 1
    ordered["next_five_session_return"] = next_5 / close - 1
    ordered["next_five_session_abs_return"] = ordered.next_five_session_return.abs()
    columns = ["date", "symbol", "sector", "partition", "anomaly_score", "anomaly_flag",
               "baseline_volume_flag", "baseline_return_flag", "next_session_date",
               "next_session_return", "next_five_session_date", "next_five_session_return",
               "next_five_session_abs_return"]
    return ordered.loc[ordered.model_ready, columns].sort_values(["date", "symbol"]).reset_index(drop=True)


def outcome_summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Summarize rows with full five-session follow-up; return fractions as percent."""
    values = frame.loc[frame.next_five_session_return.notna(), "next_five_session_return"]
    count = len(values)
    if not count:
        return {"observations": 0, "mean_abs_return_pct": None, "median_abs_return_pct": None,
                "mean_signed_return_pct": None, "positive_pct": None, "negative_pct": None,
                "zero_pct": None}
    return {"observations": count,
            "mean_abs_return_pct": round(float(values.abs().mean() * 100), 3),
            "median_abs_return_pct": round(float(values.abs().median() * 100), 3),
            "mean_signed_return_pct": round(float(values.mean() * 100), 3),
            "positive_pct": round(float((values > 0).mean() * 100), 3),
            "negative_pct": round(float((values < 0).mean() * 100), 3),
            "zero_pct": round(float((values == 0).mean() * 100), 3)}


def holdout_comparisons(outcomes: pd.DataFrame, min_group_size: int = 5) -> dict:
    """Compare descriptive holdout cohorts without fitting or significance tests."""
    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")
    holdout = outcomes.loc[outcomes.partition == "holdout"].copy()
    complete = holdout.loc[holdout.next_five_session_return.notna()].copy()
    flagged = complete.anomaly_flag.astype(bool)
    baseline = complete.baseline_volume_flag.astype(bool) | complete.baseline_return_flag.astype(bool)
    cohorts = {
        "flagged": outcome_summary(complete.loc[flagged]),
        "unflagged": outcome_summary(complete.loc[~flagged]),
        "model_only": outcome_summary(complete.loc[flagged & ~baseline]),
        "model_and_baseline": outcome_summary(complete.loc[flagged & baseline]),
    }
    differences = {}
    for label, a, b in (("flagged_minus_unflagged", "flagged", "unflagged"),
                        ("model_only_minus_model_and_baseline", "model_only", "model_and_baseline")):
        left, right = cohorts[a]["mean_abs_return_pct"], cohorts[b]["mean_abs_return_pct"]
        differences[label] = round(left - right, 3) if left is not None and right is not None else None
    subgroup_rows = []
    for dimension in ("symbol", "sector"):
        for key, group in complete.groupby(dimension, sort=True):
            is_flagged = group.anomaly_flag.astype(bool)
            yes, no = outcome_summary(group.loc[is_flagged]), outcome_summary(group.loc[~is_flagged])
            subgroup_rows.append({"dimension": dimension, "name": key, "flagged": yes,
                                  "unflagged": no,
                                  "comparison_available": yes["observations"] >= min_group_size and no["observations"] >= min_group_size})
    return {
        "holdout_scored_observations": int(len(holdout)),
        "complete_five_session_observations": int(len(complete)),
        "excluded_without_full_window": int(len(holdout) - len(complete)),
        "horizon": "five subsequent observed sessions within the same stock history",
        "cohorts": cohorts,
        "mean_absolute_return_differences_pp": differences,
        "subgroups": subgroup_rows,
        "minimum_subgroup_size": min_group_size,
        "interpretation": "Retrospective descriptive comparison only; no verified labels, causal inference, predictive validation, or trading-profitability claim.",
    }


def diagnose_symbols(results: pd.DataFrame, symbols: Iterable[str] = ("GAS", "VHM")) -> dict:
    """Flag observations worth review without asserting a corporate-action cause."""
    report = {}
    for symbol in symbols:
        stock = results.loc[results.symbol == symbol].sort_values("date")
        if stock.empty:
            report[symbol] = {"status": "not_in_dataset"}
            continue
        ready = stock.loc[stock.model_ready]
        partition = {}
        for name in ("train", "holdout"):
            group = ready.loc[ready.partition == name]
            count = len(group)
            flags = int(group.anomaly_flag.sum()) if count else 0
            partition[name] = {"eligible": count, "anomalies": flags,
                               "rate_pct": round(100 * flags / count, 3) if count else 0.0}
        flags = ready.loc[ready.anomaly_flag.astype("boolean").fillna(False)].copy()
        flags["month"] = flags.date.str.slice(0, 7)
        high_low_bad = (stock.high < stock[["open", "low", "close"]].max(axis=1)) | (
            stock.low > stock[["open", "high", "close"]].min(axis=1))
        prices = stock[["open", "high", "low", "close"]].to_numpy(dtype=float)
        nonpositive_price = (~np.isfinite(prices) | (prices <= 0)).any(axis=1)
        volumes = stock.volume.to_numpy(dtype=float)
        invalid_volume = ~np.isfinite(volumes) | (volumes < 0)
        review = flags.reindex(flags.anomaly_score.abs().sort_values(ascending=False).index).head(8)
        report[symbol] = {
            "status": "available", "covered_sessions": int(len(stock)),
            "partitions": partition,
            "flagged_months": {str(k): int(v) for k, v in flags.groupby("month").size().items()},
            "maximum_absolute_daily_return_pct": round(float(ready.return_1d.abs().max() * 100), 3),
            "maximum_volume_ratio": round(float(ready.volume_ratio_prev20.max()), 3),
            "ohlc_inconsistent_rows": int(high_low_bad.sum()),
            "nonpositive_or_nonfinite_price_rows": int(nonpositive_price.sum()),
            "negative_or_nonfinite_volume_rows": int(invalid_volume.sum()),
            "review_candidates": review[["date", "partition", "anomaly_score", "return_1d",
                                         "return_5d", "volume_ratio_prev20", "volume_zscore_prev20",
                                         "close"]].to_dict(orient="records"),
            "caveat": "Review price-adjustment history, corporate disclosures, and market context manually; the data alone does not identify a cause.",
        }
    return report


def research_brief(row: pd.Series, outcome: pd.Series | None,
                   *, source: str = "Vnstock historical OHLCV") -> str:
    """Create a deterministic Markdown note from one saved anomaly and follow-up."""
    if not bool(row.anomaly_flag):
        raise ValueError("A research brief requires a flagged observation")
    model_only = not (bool(row.baseline_volume_flag) or bool(row.baseline_return_flag))
    classification = "Model-only anomaly" if model_only else "Model anomaly also flagged by a statistical baseline"
    volume_rule = "flagged" if bool(row.baseline_volume_flag) else "not flagged"
    return_rule = "flagged" if bool(row.baseline_return_flag) else "not flagged"
    future = "No full five-session follow-up is available in the saved stock history."
    if outcome is not None and pd.notna(outcome.get("next_five_session_return")):
        next_1 = outcome.get("next_session_return")
        one = f"{next_1:+.2%}" if pd.notna(next_1) else "unavailable"
        future = (f"Next observed session ({outcome['next_session_date']}): {one}. "
                  f"Five observed sessions later ({outcome['next_five_session_date']}): "
                  f"{outcome['next_five_session_return']:+.2%} cumulative; "
                  f"absolute move {outcome['next_five_session_abs_return']:.2%}.")
    elif outcome is not None and pd.notna(outcome.get("next_session_return")):
        future = (f"Next observed session ({outcome['next_session_date']}): "
                  f"{outcome['next_session_return']:+.2%}. A full five-session window is unavailable.")
    return f"""# Historical anomaly research brief — {row.symbol} · {row.date}

**Industry:** {row.sector}

**Dataset partition:** {row.partition}

**Classification:** {classification}

**Isolation Forest score:** {row.anomaly_score:.4f} (higher means more unusual; not a probability)

## Observed session

- Close: {row.close:,.2f} thousand VND per share; daily return: {row.return_1d:+.2%}; prior five-session return: {row.return_5d:+.2%}.
- Volume: {row.volume:,.0f} shares; {row.volume_ratio_prev20:.2f}× its preceding 20-session average; volume z-score {row.volume_zscore_prev20:+.2f}.
- VNINDEX return on the same date: {row.vnindex_return_1d:+.2%}; stock excess return: {row.excess_return_1d * 100:+.2f} percentage points.
- Prior return volatility: {row.return_volatility_prev20:.2%} per session.

**Saved heuristic context:** {row.anomaly_explanation}

## Fixed statistical baselines

- Volume z-score > 3: {volume_rule} ({row.volume_zscore_prev20:+.2f}).
- Absolute return z-score > 3: {return_rule} ({row.return_zscore_prev20:+.2f}).

## Retrospective follow-up

{future} These later prices were not inputs to the anomaly model or used to select this event.

## Source and limits

Source: {source}. This note uses locally saved historical equity bars, engineered features, and a previously fitted Isolation Forest. Price history follows the source's existing adjustment convention; check source metadata and public disclosures before interpreting a discontinuity. Heuristic context describes observed values and is not a verified cause or model attribution. The covered stocks are not representative of entire industries. No verified anomaly labels or investment recommendation are provided.
"""
