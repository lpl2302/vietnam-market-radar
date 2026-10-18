"""Offline cache selection, provenance, and feature CLI behavior."""

from datetime import date
import json

import pandas as pd
import pytest

from market_radar import build_features
from market_radar.config import PROJECT_ROOT
from market_radar.features.storage import build_from_cache, load_processed_asset
from market_radar.ingestion.universe import build_universe


def bars(symbol: str, count: int, *, index: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=count).strftime("%Y-%m-%d")
    close = [100.0 + i * i / 10 for i in range(count)]
    return pd.DataFrame({
        "date": dates, "symbol": symbol,
        "asset_type": "index" if index else "equity",
        "sector": "Market benchmark" if index else "Technology",
        "open": close, "high": [price + 1 for price in close],
        "low": [price - 1 for price in close], "close": close,
        "volume": [100 + i for i in range(count)],
        "price_unit": "points" if index else "thousand_vnd",
    })


def save_cache(directory, frame: pd.DataFrame, request_start: str):
    symbol = frame.symbol.iloc[0]
    kind = frame.asset_type.iloc[0]
    end = frame.date.iloc[-1]
    stem = f"{kind}_{symbol}_{request_start}_{end}"
    csv_path = directory / f"{stem}.csv"
    frame.to_csv(csv_path, index=False)
    metadata = {
        "schema_version": 2, "symbol": symbol, "asset_type": kind,
        "start": request_start, "end": end,
        "first_observation": frame.date.iloc[0],
        "last_observation": end, "normalized_rows": len(frame),
        "retrieved_at_utc": "2026-02-01T00:00:00+00:00",
        "provider": "Vnstock", "provider_source": "vci",
        "complete": True, "limitation": None,
    }
    (directory / f"{stem}.meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    return csv_path


def test_longest_observed_cache_is_selected_and_sources_are_recorded(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    full = bars("FPT", 25)
    shorter = full.iloc[1:].reset_index(drop=True)
    full_path = save_cache(processed, full, "2025-01-01")
    save_cache(processed, shorter, "2026-01-02")
    save_cache(processed, bars("VNINDEX", 25, index=True), "2025-01-01")
    original_bytes = full_path.read_bytes()
    fpt, _, chosen = load_processed_asset(build_universe(["FPT"])[0], processed)
    assert len(fpt) == 25 and chosen == full_path

    output = tmp_path / "features" / "market_features.csv"
    feature_frame = build_from_cache(build_universe(["FPT"]), processed, output)
    assert len(feature_frame) == 25
    assert full_path.read_bytes() == original_bytes
    assert output.is_file()
    metadata = json.loads(output.with_suffix(".meta.json").read_text())
    assert metadata["feature_rows"] == 25
    assert metadata["sources"][0]["csv"] == full_path.name
    assert len(metadata["sources"][0]["sha256"]) == 64


def test_missing_cached_symbol_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No processed OHLCV cache"):
        load_processed_asset(build_universe(["FPT"])[0], tmp_path)


def test_cli_symbol_and_output_arguments(monkeypatch, capsys):
    seen = {}

    def fake_build(assets, processed_dir, output):
        seen.update(assets=assets, output=output)
        return pd.DataFrame({
            "symbol": ["FPT"], "date": [date(2026, 1, 1).isoformat()],
            "model_ready": [True],
        })

    monkeypatch.setattr(build_features, "build_from_cache", fake_build)
    assert build_features.main(["--symbols", "fpt", "--output", "data/features/custom.csv"]) == 0
    assert [asset.symbol for asset in seen["assets"]] == ["FPT", "VNINDEX"]
    assert seen["output"] == PROJECT_ROOT / "data/features/custom.csv"
    assert "1 model-ready" in capsys.readouterr().out

    with pytest.raises(SystemExit) as invalid:
        build_features.main(["--output", "features.txt"])
    assert invalid.value.code == 2


def test_cli_returns_nonzero_when_cache_is_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(build_features, "PROCESSED_DATA_DIR", tmp_path)
    assert build_features.main(["--symbols", "FPT", "--output", str(tmp_path / "features.csv")]) == 1
    assert "No processed OHLCV cache" in capsys.readouterr().err
