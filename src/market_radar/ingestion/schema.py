"""Validate and normalize daily Vnstock OHLCV observations."""

from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_radar.ingestion.universe import Asset


VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
PROVIDER_COLUMNS = ("time", "open", "high", "low", "close", "volume")
CANONICAL_COLUMNS = (
    "date", "symbol", "asset_type", "sector", "open", "high", "low", "close",
    "volume", "price_unit",
)


class DataValidationError(ValueError):
    """Provider or cached data cannot safely be used for research."""


@dataclass(frozen=True)
class NormalizationResult:
    frame: pd.DataFrame
    duplicates_removed: int


def _market_dates(values: pd.Series) -> pd.Series:
    """Interpret naive provider timestamps as Vietnam time; convert aware ones."""
    try:
        timestamps = pd.to_datetime(values, errors="coerce", format="mixed")
        if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
            timestamps = timestamps.dt.tz_convert(VIETNAM_TZ)
        else:
            timestamps = timestamps.dt.tz_localize(VIETNAM_TZ)
        return timestamps.dt.date
    except (TypeError, ValueError, AttributeError) as exc:
        raise DataValidationError("Invalid or mixed timestamp formats") from exc


def validate_canonical(frame: pd.DataFrame, asset: Asset, start: date, end: date) -> pd.DataFrame:
    """Reject missing, impossible, out-of-window, or ambiguous daily bars."""
    if tuple(frame.columns) != CANONICAL_COLUMNS:
        raise DataValidationError(f"Expected columns {CANONICAL_COLUMNS}; got {tuple(frame.columns)}")
    if frame.empty:
        raise DataValidationError(f"{asset.symbol}: no OHLCV observations returned")

    result = frame.copy()
    parsed_dates = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="coerce")
    if parsed_dates.isna().any() or (parsed_dates.dt.strftime("%Y-%m-%d") != result["date"]).any():
        raise DataValidationError(f"{asset.symbol}: invalid date")
    if (parsed_dates.dt.date < start).any() or (parsed_dates.dt.date > end).any():
        raise DataValidationError(f"{asset.symbol}: observations outside requested date range")
    if result["date"].duplicated().any():
        raise DataValidationError(f"{asset.symbol}: conflicting observations for one trading date")
    if (result["symbol"] != asset.symbol).any() or (result["asset_type"] != asset.asset_type).any() or (result["sector"] != asset.sector).any():
        raise DataValidationError(f"{asset.symbol}: incorrect symbol, asset type, or sector")
    expected_unit = "thousand_vnd" if asset.asset_type == "equity" else "points"
    if (result["price_unit"] != expected_unit).any():
        raise DataValidationError(f"{asset.symbol}: incorrect price unit")

    for column in (*PROVIDER_COLUMNS[1:5], "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    numeric = result[["open", "high", "low", "close", "volume"]]
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise DataValidationError(f"{asset.symbol}: missing or nonnumeric OHLCV value")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError(f"{asset.symbol}: prices must be positive")
    if (result["volume"] < 0).any() or (result["volume"] % 1 != 0).any():
        raise DataValidationError(f"{asset.symbol}: volume must be a nonnegative integer")
    if (result["high"] + 1e-9 < result[["open", "low", "close"]].max(axis=1)).any():
        raise DataValidationError(f"{asset.symbol}: high is below another OHLC price")
    if (result["low"] - 1e-9 > result[["open", "high", "close"]].min(axis=1)).any():
        raise DataValidationError(f"{asset.symbol}: low is above another OHLC price")

    for column in ("open", "high", "low", "close"):
        result[column] = result[column].astype("float64")
    result["volume"] = result["volume"].astype("int64")
    return result.sort_values("date").reset_index(drop=True)


def normalize_ohlcv(raw: pd.DataFrame, asset: Asset, start: date, end: date) -> NormalizationResult:
    """Normalize Vnstock's six-column daily response without filling absent days."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataValidationError(f"{asset.symbol}: no OHLCV observations returned")
    missing = set(PROVIDER_COLUMNS) - set(raw.columns)
    if missing:
        raise DataValidationError(f"{asset.symbol}: missing provider columns {sorted(missing)}")

    frame = raw.loc[:, list(PROVIDER_COLUMNS)].copy()
    frame.insert(0, "date", _market_dates(frame.pop("time")).astype("string"))
    frame.insert(1, "symbol", asset.symbol)
    frame.insert(2, "asset_type", asset.asset_type)
    frame.insert(3, "sector", asset.sector)
    frame["price_unit"] = "thousand_vnd" if asset.asset_type == "equity" else "points"
    if frame["date"].isna().any():
        raise DataValidationError(f"{asset.symbol}: missing or invalid timestamp")
    before = len(frame)
    frame = frame.drop_duplicates()
    validated = validate_canonical(frame.loc[:, CANONICAL_COLUMNS], asset, start, end)
    return NormalizationResult(validated, before - len(frame))
