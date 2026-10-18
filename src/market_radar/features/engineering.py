"""Pure, session-based financial features with no future-data leakage."""

import numpy as np
import pandas as pd

from market_radar.ingestion.schema import CANONICAL_COLUMNS


# The only numerical inputs intended for Task 004. Identifiers and OHLCV are
# retained in the output for auditability but are not model predictors.
MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "return_1d",
    "return_5d",
    "intraday_range_pct",
    "price_vs_prev_ma20_pct",
    "volume_ratio_prev20",
    "volume_zscore_prev20",
    "return_volatility_prev20",
    "return_zscore_prev20",
    "vnindex_return_1d",
    "excess_return_1d",
)


def _check_input(frame: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    missing = set(CANONICAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{asset_type}: missing normalized columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{asset_type}: no normalized observations")
    if (frame["asset_type"] != asset_type).any():
        raise ValueError(f"Expected only {asset_type} rows")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError(f"{asset_type}: duplicate symbol/date observations")
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True).copy()


def _stock_features(stock: pd.DataFrame) -> pd.DataFrame:
    """Calculate one symbol's features using trading rows, never calendar fill."""
    result = stock.copy()
    close = result["close"].astype(float)
    volume = result["volume"].astype(float)
    daily_return = close.div(close.shift(1)).sub(1)

    previous_close = close.shift(1)
    previous_volume = volume.shift(1)
    previous_returns = daily_return.shift(1)
    close_mean_20 = previous_close.rolling(20, min_periods=20).mean()
    volume_mean_20 = previous_volume.rolling(20, min_periods=20).mean()
    volume_std_20 = previous_volume.rolling(20, min_periods=20).std(ddof=1)
    return_mean_20 = previous_returns.rolling(20, min_periods=20).mean()
    return_std_20 = previous_returns.rolling(20, min_periods=20).std(ddof=1)

    result["return_1d"] = daily_return
    result["return_5d"] = close.div(close.shift(5)).sub(1)
    result["intraday_range_pct"] = result["high"].sub(result["low"]).div(result["open"]).mul(100)
    result["price_vs_prev_ma20_pct"] = close.div(close_mean_20).sub(1).mul(100)
    result["volume_ratio_prev20"] = volume.div(volume_mean_20.where(volume_mean_20.ne(0)))
    result["volume_zscore_prev20"] = volume.sub(volume_mean_20).div(volume_std_20.where(volume_std_20.ne(0)))
    result["return_volatility_prev20"] = return_std_20
    result["return_zscore_prev20"] = daily_return.sub(return_mean_20).div(return_std_20.where(return_std_20.ne(0)))
    return result


def build_feature_table(stocks: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Join stock features to same-date VNINDEX returns and mark ML-ready rows.

    All observations describe completed sessions. Rolling comparison windows
    exclude the current row; no benchmark dates are invented or forward-filled.
    """
    equity = _check_input(stocks, "equity")
    market = _check_input(benchmark, "index")
    if set(market["symbol"]) != {"VNINDEX"}:
        raise ValueError("Benchmark must contain VNINDEX only")

    market_return = market["close"].astype(float).div(market["close"].astype(float).shift(1)).sub(1)
    market_by_date = pd.DataFrame({"date": market["date"], "vnindex_return_1d": market_return})
    frames = [_stock_features(group) for _, group in equity.groupby("symbol", sort=False)]
    result = pd.concat(frames, ignore_index=True)
    result = result.merge(market_by_date, on="date", how="left", validate="many_to_one", sort=False)
    result["excess_return_1d"] = result["return_1d"] - result["vnindex_return_1d"]
    result.loc[:, MODEL_FEATURE_COLUMNS] = result.loc[:, MODEL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    result["model_ready"] = np.isfinite(result.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    return result.sort_values(["symbol", "date"]).reset_index(drop=True)
