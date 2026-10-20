"""Pooled Isolation Forest with a date-based holdout and heuristic context."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_radar.features import MODEL_FEATURE_COLUMNS
from market_radar.ingestion.schema import CANONICAL_COLUMNS


@dataclass(frozen=True)
class DetectionConfig:
    n_estimators: int = 200
    contamination: float = 0.05
    random_state: int = 42
    n_jobs: int = -1
    train_fraction: float = 0.70

    def validate(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if not 0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")


@dataclass(frozen=True)
class ChronologicalSplit:
    train_indices: pd.Index
    holdout_indices: pd.Index
    train_dates: tuple[str, ...]
    holdout_dates: tuple[str, ...]


@dataclass(frozen=True)
class DetectionRun:
    results: pd.DataFrame
    model: Pipeline
    split: ChronologicalSplit
    config: DetectionConfig


def validate_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and ready-row numerics without changing the source."""
    required = set(CANONICAL_COLUMNS) | set(MODEL_FEATURE_COLUMNS) | {"model_ready"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature dataset is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Feature dataset is empty")
    result = frame.copy()
    if not pd.api.types.is_bool_dtype(result["model_ready"]):
        raise ValueError("model_ready must be boolean")
    if result[["date", "symbol", "sector"]].isna().any().any() or result[["date", "symbol", "sector"]].eq("").any().any():
        raise ValueError("date, symbol, and sector must be present")
    parsed = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any() or (parsed.dt.strftime("%Y-%m-%d") != result["date"]).any():
        raise ValueError("date must use YYYY-MM-DD")
    if result.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate symbol/date rows in feature dataset")
    if (result["asset_type"] != "equity").any():
        raise ValueError("Feature dataset must contain only equity rows")
    for name in ("open", "high", "low", "close", "volume"):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    bars = result[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(bars.to_numpy(dtype=float)).all():
        raise ValueError("OHLCV values must be finite")
    if (result[["open", "high", "low", "close"]] <= 0).any().any() or (result["volume"] < 0).any():
        raise ValueError("OHLCV prices must be positive and volume nonnegative")
    if (result["high"] < result[["open", "low", "close"]].max(axis=1)).any() or (result["low"] > result[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("OHLCV high/low values are inconsistent")
    for name in MODEL_FEATURE_COLUMNS:
        original = result[name]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"Feature {name} contains nonnumeric values")
        result[name] = numeric.astype("float64")
    ready_values = result.loc[result.model_ready, MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(ready_values).all():
        raise ValueError("model_ready rows contain missing or nonfinite model features")
    if len(ready_values) == 0:
        raise ValueError("No model-ready observations")
    return result


def chronological_split(ready: pd.DataFrame, train_fraction: float = 0.70) -> ChronologicalSplit:
    """Split unique dates, keeping every symbol on a date in one partition."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    dates = sorted(ready["date"].unique())
    if len(dates) < 2:
        raise ValueError("At least two model-ready trading dates are required")
    cut = min(max(int(len(dates) * train_fraction), 1), len(dates) - 1)
    train_dates = tuple(dates[:cut])
    holdout_dates = tuple(dates[cut:])
    train_mask = ready["date"].isin(train_dates)
    return ChronologicalSplit(ready.index[train_mask], ready.index[~train_mask], train_dates, holdout_dates)


def make_model(config: DetectionConfig) -> Pipeline:
    """Persist scaling and Isolation Forest as one fitted object."""
    config.validate()
    return Pipeline([
        ("scaler", StandardScaler()),
        ("isolation_forest", IsolationForest(
            n_estimators=config.n_estimators,
            contamination=config.contamination,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )),
    ])


def explain_anomaly(row: pd.Series) -> str:
    """Describe notable observed values; this is not model attribution."""
    needed = (
        "return_1d", "excess_return_1d", "volume_ratio_prev20",
        "volume_zscore_prev20", "return_zscore_prev20", "intraday_range_pct",
    )
    if any(pd.isna(row.get(name)) or not np.isfinite(float(row[name])) for name in needed):
        return ""

    candidates: list[tuple[float, str]] = []
    volume_ratio = float(row["volume_ratio_prev20"])
    volume_z = float(row["volume_zscore_prev20"])
    daily_return = float(row["return_1d"])
    excess_return = float(row["excess_return_1d"])
    return_z = float(row["return_zscore_prev20"])
    intraday_range = float(row["intraday_range_pct"])
    if volume_ratio >= 2 or volume_ratio <= 0.25:
        strength = volume_ratio / 2 if volume_ratio >= 2 else 0.25 / max(volume_ratio, 0.01)
        candidates.append((strength, f"volume was {volume_ratio:.1f}× its prior 20-session average"))
    elif volume_z >= 3:
        candidates.append((volume_z / 3, f"volume was {volume_z:.1f} standard deviations above its prior 20-session mean"))
    if abs(daily_return) >= 0.03:
        detail = f"daily return was {daily_return:+.1%}"
        if abs(return_z) >= 3:
            detail += f" ({abs(return_z):.1f} standard deviations from its prior mean)"
        candidates.append((abs(daily_return) / 0.03, detail))
    if abs(excess_return) >= 0.025:
        direction = "outperformed" if excess_return > 0 else "underperformed"
        candidates.append((abs(excess_return) / 0.025, f"the stock {direction} VNINDEX by {abs(excess_return) * 100:.1f} percentage points"))
    if intraday_range >= 5:
        candidates.append((intraday_range / 5, f"intraday range was {intraday_range:.1f}% of opening price"))
    if abs(return_z) >= 3 and abs(daily_return) < 0.03:
        candidates.append((abs(return_z) / 3, f"daily return was {abs(return_z):.1f} standard deviations from its prior mean"))

    if not candidates:
        context = "the joint pattern of the financial features was unusual relative to training sessions"
    else:
        context = "; ".join(text for _, text in sorted(candidates, key=lambda item: -item[0])[:3])
    return f"Heuristic context: {context}. This describes observed values, not model feature attribution."


def detect_anomalies(frame: pd.DataFrame, config: DetectionConfig = DetectionConfig()) -> DetectionRun:
    """Fit on early dates and score all ready rows using a fixed model boundary."""
    config.validate()
    result = validate_feature_table(frame)
    ready = result.loc[result["model_ready"]]
    split = chronological_split(ready, config.train_fraction)
    model = make_model(config)
    model.fit(result.loc[split.train_indices, list(MODEL_FEATURE_COLUMNS)])

    result["partition"] = "not_model_ready"
    result.loc[split.train_indices, "partition"] = "train"
    result.loc[split.holdout_indices, "partition"] = "holdout"
    result["scoring_status"] = np.where(result["model_ready"], "scored", "insufficient_features")
    result["anomaly_score"] = np.nan
    result["anomaly_flag"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["baseline_volume_flag"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["baseline_return_flag"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["anomaly_explanation"] = ""

    features = result.loc[ready.index, list(MODEL_FEATURE_COLUMNS)]
    scores = -model.decision_function(features)
    flags = model.predict(features) == -1
    result.loc[ready.index, "anomaly_score"] = scores
    result.loc[ready.index, "anomaly_flag"] = flags
    result.loc[ready.index, "baseline_volume_flag"] = result.loc[ready.index, "volume_zscore_prev20"].gt(3)
    result.loc[ready.index, "baseline_return_flag"] = result.loc[ready.index, "return_zscore_prev20"].abs().gt(3)
    flagged = result["anomaly_flag"].fillna(False)
    result.loc[flagged, "anomaly_explanation"] = result.loc[flagged].apply(explain_anomaly, axis=1)
    return DetectionRun(result.sort_values(["date", "symbol"]).reset_index(drop=True), model, split, config)
