"""Offline tests for Vnstock response handling and cache behavior."""

from datetime import date, datetime
import json
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import market_radar.ingestion.pipeline as pipeline_module
from market_radar.ingestion.pipeline import HistoricalDataPipeline, cache_paths, default_window
from market_radar.ingestion.schema import DataValidationError, normalize_ohlcv
from market_radar.ingestion.universe import build_universe


START = date(2026, 10, 1)
END = date(2026, 10, 16)
FPT = build_universe(["FPT"])[0]


def provider_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-10-02 07:00", "2026-10-01 07:00"]).tz_localize("Asia/Ho_Chi_Minh"),
            "open": [10.0, 9.5],
            "high": [11.0, 10.0],
            "low": [9.0, 9.0],
            "close": [10.5, 9.7],
            "volume": [100, 200],
        }
    )
    frame.attrs["source"] = "vci"
    return frame


def test_normalization_schema_order_types_and_exact_duplicates():
    raw = provider_frame()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    normalized = normalize_ohlcv(raw, FPT, START, END)

    assert normalized.duplicates_removed == 1
    assert normalized.frame["date"].tolist() == ["2026-10-01", "2026-10-02"]
    assert normalized.frame["symbol"].tolist() == ["FPT", "FPT"]
    assert normalized.frame["price_unit"].tolist() == ["thousand_vnd"] * 2
    assert normalized.frame["volume"].dtype == "int64"
    assert normalized.frame["close"].dtype == "float64"


def test_timestamp_conversion_uses_vietnam_market_date():
    raw = provider_frame().iloc[[0]].copy()
    raw["time"] = pd.to_datetime(["2026-10-01 18:00:00+00:00"], utc=True)
    assert normalize_ohlcv(raw, FPT, START, END).frame["date"].iloc[0] == "2026-10-02"

    raw["time"] = ["2026-10-02 07:00:00"]
    assert normalize_ohlcv(raw, FPT, START, END).frame["date"].iloc[0] == "2026-10-02"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda df: df.assign(close=None), "missing or nonnumeric"),
        (lambda df: df.assign(open=0), "prices must be positive"),
        (lambda df: df.assign(volume=-1), "nonnegative integer"),
        (lambda df: df.assign(volume=1.5), "nonnegative integer"),
        (lambda df: df.assign(high=8), "high is below"),
        (lambda df: df.assign(low=12, high=13), "low is above"),
        (lambda df: df.assign(time="invalid"), "timestamp"),
    ],
)
def test_invalid_observations_fail_instead_of_being_filled(change, message):
    with pytest.raises(DataValidationError, match=message):
        normalize_ohlcv(change(provider_frame()), FPT, START, END)


def test_empty_missing_columns_and_conflicting_same_day_fail():
    with pytest.raises(DataValidationError, match="no OHLCV"):
        normalize_ohlcv(pd.DataFrame(), FPT, START, END)
    with pytest.raises(DataValidationError, match="missing provider columns"):
        normalize_ohlcv(provider_frame().drop(columns="volume"), FPT, START, END)
    conflicting = pd.concat([provider_frame(), provider_frame().iloc[[0]].assign(close=10.8)])
    with pytest.raises(DataValidationError, match="conflicting observations"):
        normalize_ohlcv(conflicting, FPT, START, END)


def test_exact_window_cache_and_refresh(tmp_path):
    calls = []

    def fetch(asset, start, end):
        calls.append((asset.symbol, start, end))
        return provider_frame()

    pipeline = HistoricalDataPipeline(
        raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed",
        fetch=fetch, request_delay_seconds=0, pause=lambda _: None, report=lambda _: None,
    )
    first = pipeline.download_asset(FPT, START, END)
    second = pipeline.download_asset(FPT, START, END)
    third = pipeline.download_asset(FPT, START, END, refresh=True)
    paths = cache_paths(FPT, START, END, tmp_path / "raw", tmp_path / "processed")

    assert (first.cached, second.cached, third.cached) == (False, True, False)
    assert first.complete and second.complete and third.complete
    assert len(calls) == 2
    assert paths.raw.is_file() and paths.normalized.is_file()
    assert pd.read_csv(paths.normalized)["date"].tolist() == ["2026-10-01", "2026-10-02"]
    metadata = json.loads(paths.metadata.read_text())
    assert metadata["provider_source"] == "vci"
    assert metadata["raw_rows"] == 2 and metadata["normalized_rows"] == 2
    assert metadata["start"] == START.isoformat() and metadata["end"] == END.isoformat()


def test_tier_capped_response_is_cached_and_reported_partial(tmp_path):
    calls = []

    def fetch(asset, start, end):
        calls.append(asset.symbol)
        frame = provider_frame()
        frame.attrs["tier_cap_history_days"] = 365
        return frame

    pipeline = HistoricalDataPipeline(
        tmp_path / "raw", tmp_path / "processed", fetch=fetch,
        request_delay_seconds=0, pause=lambda _: None, report=lambda _: None,
    )
    first = pipeline.run([FPT], START, END)
    second = pipeline.run([FPT], START, END)
    assert calls == ["FPT"]
    assert first.incomplete == second.incomplete
    assert not first.results[0].complete and second.results[0].cached


def test_transient_failures_retry_but_invalid_requests_do_not(tmp_path):
    attempts = []
    pauses = []

    def transient(asset, start, end):
        attempts.append(asset.symbol)
        if len(attempts) == 1:
            raise TimeoutError("temporary timeout")
        return provider_frame()

    pipeline = HistoricalDataPipeline(
        tmp_path / "raw", tmp_path / "processed", fetch=transient,
        request_delay_seconds=1, pause=pauses.append, report=lambda _: None,
    )
    assert pipeline.download_asset(FPT, START, END).rows == 2
    assert len(attempts) == 2 and pauses == [2, 1]

    def invalid(asset, start, end):
        attempts.append(asset.symbol)
        raise ValueError("bad symbol")

    failing = HistoricalDataPipeline(
        tmp_path / "raw2", tmp_path / "processed2", fetch=invalid,
        request_delay_seconds=0, pause=lambda _: None, report=lambda _: None,
    )
    summary = failing.run([FPT], START, END)
    assert len(summary.failures) == 1 and len(attempts) == 3


def test_only_minute_entitlement_error_is_retried(monkeypatch, tmp_path):
    class FakeEntitlementError(Exception):
        def __init__(self, code):
            self.code = code
            super().__init__(code)

    monkeypatch.setattr(pipeline_module, "EntitlementError", FakeEntitlementError)
    attempts = []
    pauses = []

    def minute_then_success(asset, start, end):
        attempts.append(asset.symbol)
        if len(attempts) == 1:
            raise FakeEntitlementError("VNSTOCK_AUTH_MINUTE_LIMIT")
        return provider_frame()

    pipeline = HistoricalDataPipeline(
        tmp_path / "raw", tmp_path / "processed", fetch=minute_then_success,
        request_delay_seconds=0, pause=pauses.append, report=lambda _: None,
    )
    assert pipeline.download_asset(FPT, START, END).rows == 2
    assert attempts == ["FPT", "FPT"] and pauses == [60, 0]

    def auth_failure(asset, start, end):
        attempts.append(asset.symbol)
        raise FakeEntitlementError("VNSTOCK_TIER_SOURCE")

    failing = HistoricalDataPipeline(
        tmp_path / "raw2", tmp_path / "processed2", fetch=auth_failure,
        request_delay_seconds=0, pause=pauses.append, report=lambda _: None,
    )
    assert "FPT" in failing.run([FPT], START, END).failures
    assert len(attempts) == 3


def test_default_window_excludes_unfinished_and_weekend_sessions():
    vietnam = ZoneInfo("Asia/Ho_Chi_Minh")
    start, end = default_window(datetime(2026, 10, 18, 12, tzinfo=vietnam))
    assert end == date(2026, 10, 16)
    assert 1094 <= (end - start).days <= 1096
    assert default_window(datetime(2026, 10, 19, 14, tzinfo=vietnam))[1] == date(2026, 10, 16)
