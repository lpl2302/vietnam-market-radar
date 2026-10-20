"""Read and summarize previously generated, local research results."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_radar.features import MODEL_FEATURE_COLUMNS


class DashboardDataError(ValueError):
    """A dashboard input is absent, stale, or malformed."""


@dataclass(frozen=True)
class DashboardData:
    results: pd.DataFrame
    features: pd.DataFrame
    evaluation: dict
    model_metadata: dict


REQUIRED_RESULT_COLUMNS = {
    "date", "symbol", "sector", "asset_type", "open", "high", "low", "close",
    "volume", "price_unit", "model_ready", "partition", "scoring_status",
    "anomaly_score", "anomaly_flag", "anomaly_explanation", *MODEL_FEATURE_COLUMNS,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DashboardDataError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DashboardDataError(f"Expected a JSON object in {path}")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DashboardDataError(f"Cannot read {path}: {exc}") from exc


def _boolean_column(frame: pd.DataFrame, column: str, *, nullable: bool) -> pd.Series:
    values = frame[column].astype("string")
    valid = values.isin(["True", "False"]) | (values.isna() if nullable else False)
    if not valid.all():
        raise DashboardDataError(f"{column} must contain True/False" + (" or blank" if nullable else ""))
    return values.map({"True": True, "False": False}).astype("boolean" if nullable else "bool")


def validate_dashboard_data(data: DashboardData) -> DashboardData:
    """Reject inconsistent saved outputs before showing analyst-facing numbers."""
    frame = data.results.copy()
    features = data.features.copy()
    missing = REQUIRED_RESULT_COLUMNS - set(frame.columns)
    if missing:
        raise DashboardDataError(f"Anomaly results are missing columns: {sorted(missing)}")
    if frame.empty:
        raise DashboardDataError("Anomaly results are empty")
    required_features = {"date", "symbol", "close", "model_ready", *MODEL_FEATURE_COLUMNS}
    if required_features - set(features.columns):
        raise DashboardDataError(f"Feature table is missing columns: {sorted(required_features - set(features.columns))}")
    for table, name in ((frame, "anomaly results"), (features, "feature table")):
        parsed = pd.to_datetime(table["date"], format="%Y-%m-%d", errors="coerce")
        if parsed.isna().any() or (parsed.dt.strftime("%Y-%m-%d") != table["date"]).any():
            raise DashboardDataError(f"Invalid trading dates in {name}")
        if table.duplicated(["date", "symbol"]).any():
            raise DashboardDataError(f"Duplicate symbol/date observations in {name}")
    frame["model_ready"] = _boolean_column(frame, "model_ready", nullable=False)
    frame["anomaly_flag"] = _boolean_column(frame, "anomaly_flag", nullable=True)
    features["model_ready"] = _boolean_column(features, "model_ready", nullable=False)
    if frame[["date", "symbol", "sector"]].isna().any().any() or frame[["symbol", "sector"]].eq("").any().any():
        raise DashboardDataError("Results need a date, stock symbol, and sector on every row")
    ready = frame["model_ready"]
    if (frame.loc[ready, "partition"].isin(["train", "holdout"]) == False).any():
        raise DashboardDataError("Scored rows need train or holdout partition")
    if (frame.loc[~ready, "partition"] != "not_model_ready").any():
        raise DashboardDataError("Unscored rows need not_model_ready partition")
    if (frame.loc[ready, "scoring_status"] != "scored").any() or (frame.loc[~ready, "scoring_status"] != "insufficient_features").any():
        raise DashboardDataError("Scoring status disagrees with model_ready")
    if frame.loc[ready, "anomaly_flag"].isna().any() or frame.loc[~ready, "anomaly_flag"].notna().any():
        raise DashboardDataError("Anomaly flags disagree with model_ready")
    if frame.loc[ready & frame["anomaly_flag"].fillna(False), "anomaly_explanation"].fillna("").eq("").any():
        raise DashboardDataError("Flagged observations need saved heuristic explanations")
    frame["anomaly_score"] = pd.to_numeric(frame["anomaly_score"], errors="coerce")
    if not np.isfinite(frame.loc[ready, "anomaly_score"].to_numpy(dtype=float)).all() or frame.loc[~ready, "anomaly_score"].notna().any():
        raise DashboardDataError("Anomaly scores disagree with model_ready")
    numeric = ["open", "high", "low", "close", "volume", *MODEL_FEATURE_COLUMNS]
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if not np.isfinite(frame.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)).all():
        raise DashboardDataError("OHLCV values must be finite")
    if not np.isfinite(frame.loc[ready, list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise DashboardDataError("Model-ready feature values must be finite")
    if (frame["asset_type"] != "equity").any() or (frame["price_unit"] != "thousand_vnd").any():
        raise DashboardDataError("Results must contain equity prices in thousand VND")
    for name, rule in (("baseline_volume_flag", frame["volume_zscore_prev20"].gt(3)),
                       ("baseline_return_flag", frame["return_zscore_prev20"].abs().gt(3))):
        expected = rule.where(ready, pd.NA).astype("boolean")
        if name in frame:
            supplied = _boolean_column(frame, name, nullable=True)
            if not supplied.equals(expected):
                raise DashboardDataError(f"{name} disagrees with the fixed Task 004 rule")
        frame[name] = expected
    joined = frame[["date", "symbol", "close", "model_ready"]].merge(
        features[["date", "symbol", "close", "model_ready"]],
        on=["date", "symbol"], how="outer", indicator=True, suffixes=("_result", "_feature"),
    )
    if (joined["_merge"] != "both").any() or not joined["model_ready_result"].equals(joined["model_ready_feature"]) or not np.allclose(joined["close_result"], joined["close_feature"], equal_nan=True):
        raise DashboardDataError("Feature table and anomaly results do not describe the same observations")
    scored = frame.loc[ready]
    evaluation = data.evaluation
    metadata = data.model_metadata
    try:
        if int(evaluation["all_scored"]["observations"]) != len(scored) or int(evaluation["all_scored"]["anomalies"]) != int(scored["anomaly_flag"].sum()):
            raise DashboardDataError("Evaluation summary disagrees with anomaly results")
        for partition, key in (("train", "training_samples"), ("holdout", "holdout_samples")):
            part = scored.loc[scored.partition == partition]
            if (int(metadata[key]) != len(part) or
                    int(evaluation["by_partition"][partition]["observations"]) != len(part) or
                    int(evaluation["by_partition"][partition]["anomalies"]) != int(part.anomaly_flag.sum())):
                raise DashboardDataError(f"{partition} counts disagree across saved outputs")
            if [part.date.min(), part.date.max()] != metadata[f"{'train' if partition == 'train' else 'holdout'}_date_range"]:
                raise DashboardDataError(f"{partition} dates disagree with model metadata")
        if metadata["feature_names"] != list(MODEL_FEATURE_COLUMNS):
            raise DashboardDataError("Model feature names or ordering differ from this package")
        if not isinstance(metadata["created_at_utc"], str) or not isinstance(metadata["source_feature_sha256"], str):
            raise DashboardDataError("Model provenance fields are malformed")
        settings = metadata["hyperparameters"]
        for key in ("n_estimators", "contamination", "random_state", "n_jobs", "train_fraction"):
            if key not in settings:
                raise DashboardDataError(f"Model hyperparameters lack {key}")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, DashboardDataError):
            raise
        raise DashboardDataError(f"Invalid evaluation or model metadata: {exc}") from exc
    return DashboardData(frame.sort_values(["date", "symbol"]).reset_index(drop=True), features, evaluation, metadata)


def load_dashboard_data(results_path: Path, evaluation_path: Path,
                        features_path: Path, metadata_path: Path) -> DashboardData:
    """Load four offline files; never deserialize the model or fetch market data."""
    paths = (results_path, evaluation_path, features_path, metadata_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise DashboardDataError("Missing dashboard inputs: " + ", ".join(missing))
    data = DashboardData(_read_csv(results_path), _read_csv(features_path),
                         _read_json(evaluation_path), _read_json(metadata_path))
    expected_hash = data.model_metadata.get("source_feature_sha256")
    if expected_hash and _sha256(features_path) != expected_hash:
        raise DashboardDataError("Feature file hash differs from model metadata; regenerate anomaly results")
    return validate_dashboard_data(data)


def filter_scored(frame: pd.DataFrame, *, symbols: list[str] | None = None,
                  sectors: list[str] | None = None, partitions: list[str] | None = None,
                  start: str | None = None, end: str | None = None,
                  anomalies_only: bool = False, model_only: bool = False,
                  min_score: float | None = None) -> pd.DataFrame:
    """Filter model-ready observations; empty lists intentionally match nothing."""
    subset = frame.loc[frame.model_ready].copy()
    if symbols is not None:
        subset = subset.loc[subset.symbol.isin(symbols)]
    if sectors is not None:
        subset = subset.loc[subset.sector.isin(sectors)]
    if partitions is not None:
        subset = subset.loc[subset.partition.isin(partitions)]
    if start is not None:
        subset = subset.loc[subset.date >= start]
    if end is not None:
        subset = subset.loc[subset.date <= end]
    if anomalies_only or model_only:
        subset = subset.loc[subset.anomaly_flag]
    if model_only:
        subset = subset.loc[~(subset.baseline_volume_flag | subset.baseline_return_flag)]
    if min_score is not None:
        subset = subset.loc[subset.anomaly_score >= min_score]
    return subset.sort_values(["anomaly_score", "date"], ascending=[False, False]).copy()


def frequency(frame: pd.DataFrame) -> dict[str, float | int]:
    """Use eligible rows only as the denominator."""
    count = len(frame)
    anomalies = int(frame.anomaly_flag.sum()) if count else 0
    return {"observations": count, "anomalies": anomalies,
            "rate_pct": 100 * anomalies / count if count else 0.0}


def grouped_frequency(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[column, "observations", "anomalies", "rate_pct"])
    counts = frame.groupby(column, sort=True).anomaly_flag.agg(observations="size", anomalies="sum").reset_index()
    counts["anomalies"] = counts.anomalies.astype(int)
    counts["rate_pct"] = 100 * counts.anomalies / counts.observations
    return counts.sort_values(["anomalies", column], ascending=[False, True]).reset_index(drop=True)


def monthly_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["month", "observations", "anomalies", "rate_pct"])
    copied = frame.copy()
    copied["month"] = copied.date.str.slice(0, 7)
    return grouped_frequency(copied, "month").sort_values("month").reset_index(drop=True)


def baseline_overlap(frame: pd.DataFrame) -> dict[str, int]:
    """Use the same strict >3 rules persisted in Task 004."""
    model = frame.anomaly_flag.astype(bool)
    volume = frame.baseline_volume_flag.astype(bool)
    returns = frame.baseline_return_flag.astype(bool)
    either = volume | returns
    return {"model": int(model.sum()), "volume": int(volume.sum()),
            "return": int(returns.sum()), "either": int(either.sum()),
            "model_and_either": int((model & either).sum()),
            "model_only": int((model & ~either).sum())}


def event_context(frame: pd.DataFrame, symbol: str, date: str, radius: int = 10) -> pd.DataFrame:
    """Return actual neighboring sessions of one symbol, without interpolation."""
    stock = frame.loc[frame.symbol == symbol].sort_values("date").reset_index(drop=True)
    matches = stock.index[stock.date == date]
    if len(matches) != 1:
        raise DashboardDataError(f"Event {symbol} {date} is absent")
    index = int(matches[0])
    return stock.iloc[max(0, index - radius):index + radius + 1].copy()
