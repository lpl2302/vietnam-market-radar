"""Known-value, offline checks for financial features and leakage boundaries."""

import numpy as np
import pandas as pd
import pytest

from market_radar.features import MODEL_FEATURE_COLUMNS, build_feature_table


def make_bars(symbol: str, closes: list[float], volumes: list[int], *, index: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=len(closes)).strftime("%Y-%m-%d")
    return pd.DataFrame({
        "date": dates,
        "symbol": symbol,
        "asset_type": "index" if index else "equity",
        "sector": "Market benchmark" if index else "Technology",
        "open": closes,
        "high": np.asarray(closes) + 2,
        "low": np.asarray(closes) - 2,
        "close": closes,
        "volume": volumes,
        "price_unit": "points" if index else "thousand_vnd",
    })


def test_known_financial_formulas_and_prior_session_windows():
    close = [100.0 + i * i / 10 for i in range(30)]
    volume = [100 + i * 10 for i in range(30)]
    stock = make_bars("FPT", close, volume)
    market_close = [1000 + i * 3 for i in range(30)]
    benchmark = make_bars("VNINDEX", market_close, [1000] * 30, index=True)
    result = build_feature_table(stock, benchmark)

    row = result.iloc[21]
    previous_volumes = np.array(volume[1:21], dtype=float)
    returns = np.array([close[i] / close[i - 1] - 1 for i in range(1, 21)])
    today_return = close[21] / close[20] - 1
    market_return = market_close[21] / market_close[20] - 1

    assert row["return_1d"] == pytest.approx(today_return)
    assert row["return_5d"] == pytest.approx(close[21] / close[16] - 1)
    assert row["intraday_range_pct"] == pytest.approx(4 / close[21] * 100)
    assert row["price_vs_prev_ma20_pct"] == pytest.approx((close[21] / np.mean(close[1:21]) - 1) * 100)
    assert row["volume_ratio_prev20"] == pytest.approx(volume[21] / previous_volumes.mean())
    assert row["volume_zscore_prev20"] == pytest.approx((volume[21] - previous_volumes.mean()) / previous_volumes.std(ddof=1))
    assert row["return_volatility_prev20"] == pytest.approx(returns.std(ddof=1))
    assert row["return_zscore_prev20"] == pytest.approx((today_return - returns.mean()) / returns.std(ddof=1))
    assert row["vnindex_return_1d"] == pytest.approx(market_return)
    assert row["excess_return_1d"] == pytest.approx(today_return - market_return)
    assert bool(row["model_ready"])
    assert not bool(result.iloc[20]["model_ready"])
    assert pd.isna(result.iloc[20]["return_volatility_prev20"])
    assert len(MODEL_FEATURE_COLUMNS) == 10


def test_grouping_uses_each_stocks_own_history_even_when_input_is_shuffled():
    fpt_close = [100.0 + i * i / 10 for i in range(25)]
    hpg_close = [20.0 + i / 2 for i in range(25)]
    fpt = make_bars("FPT", fpt_close, list(range(100, 125)))
    hpg = make_bars("HPG", hpg_close, list(range(1000, 1025)))
    benchmark = make_bars("VNINDEX", [1000 + i for i in range(25)], [1000] * 25, index=True)
    shuffled = pd.concat([hpg, fpt], ignore_index=True).sample(frac=1, random_state=7)
    result = build_feature_table(shuffled, benchmark)
    fpt_result = result[result.symbol == "FPT"].reset_index(drop=True)
    hpg_result = result[result.symbol == "HPG"].reset_index(drop=True)

    assert len(result) == 50
    assert fpt_result.loc[21, "return_1d"] == pytest.approx(fpt_close[21] / fpt_close[20] - 1)
    assert hpg_result.loc[21, "return_1d"] == pytest.approx(hpg_close[21] / hpg_close[20] - 1)
    assert fpt_result.loc[21, "price_vs_prev_ma20_pct"] != hpg_result.loc[21, "price_vs_prev_ma20_pct"]
    assert fpt_result.loc[0, "date"] == hpg_result.loc[0, "date"]


def test_benchmark_alignment_does_not_fill_missing_session():
    close = [100 + i * i / 10 for i in range(30)]
    stock = make_bars("FPT", close, [100 + i for i in range(30)])
    benchmark = make_bars("VNINDEX", [1000 + i for i in range(30)], [1000] * 30, index=True)
    absent_date = stock.loc[25, "date"]
    benchmark = benchmark[benchmark.date != absent_date]

    result = build_feature_table(stock, benchmark)
    absent = result[result.date == absent_date].iloc[0]
    assert len(result) == 30
    assert pd.isna(absent["vnindex_return_1d"])
    assert pd.isna(absent["excess_return_1d"])
    assert not bool(absent["model_ready"])
    assert pd.notna(result.loc[26, "vnindex_return_1d"])


def test_zero_volume_constant_prices_and_missing_history_are_explicit():
    constant = make_bars("FPT", [10.0] * 30, [0] * 30)
    benchmark = make_bars("VNINDEX", [1000 + i for i in range(30)], [1000] * 30, index=True)
    result = build_feature_table(constant, benchmark)
    row = result.iloc[21]
    assert row["return_volatility_prev20"] == pytest.approx(0)
    assert pd.isna(row["volume_ratio_prev20"])
    assert pd.isna(row["volume_zscore_prev20"])
    assert pd.isna(row["return_zscore_prev20"])
    assert not result.model_ready.any()
    assert np.isfinite(result.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)[~result.loc[:, MODEL_FEATURE_COLUMNS].isna().to_numpy()]).all()

    variable = make_bars("FPT", [10 + i * i / 100 for i in range(30)], [100 + i for i in range(30)])
    variable.loc[21, "volume"] = 0
    zero_today = build_feature_table(variable, benchmark).iloc[21]
    assert zero_today["volume_ratio_prev20"] == pytest.approx(0)
    assert bool(zero_today["model_ready"])


def test_missing_stock_session_is_not_synthesized():
    stock = make_bars("FPT", [100 + i * i / 10 for i in range(30)], [100 + i for i in range(30)])
    missing_date = stock.loc[10, "date"]
    stock = stock[stock.date != missing_date]
    benchmark = make_bars("VNINDEX", [1000 + i for i in range(30)], [1000] * 30, index=True)
    result = build_feature_table(stock, benchmark)
    assert len(result) == 29
    assert missing_date not in set(result.date)
    assert result.loc[20, "price_vs_prev_ma20_pct"] == pytest.approx(
        (result.loc[20, "close"] / result.loc[:19, "close"].mean() - 1) * 100
    )


def test_changing_future_stock_and_index_rows_cannot_change_earlier_features():
    stock = make_bars("FPT", [100 + i * i / 10 for i in range(30)], [100 + i for i in range(30)])
    benchmark = make_bars("VNINDEX", [1000 + i * i / 10 for i in range(30)], [1000] * 30, index=True)
    original = build_feature_table(stock, benchmark)

    changed_stock = stock.copy()
    changed_benchmark = benchmark.copy()
    changed_stock.loc[29, ["open", "high", "low", "close", "volume"]] = [500, 510, 490, 505, 100000]
    changed_benchmark.loc[29, ["open", "high", "low", "close"]] = [2000, 2010, 1990, 2005]
    changed = build_feature_table(changed_stock, changed_benchmark)

    pd.testing.assert_frame_equal(original.iloc[:29], changed.iloc[:29])
