"""Offline tests for command parsing and required-download failure status."""

from datetime import date

import pytest

from market_radar import download_data
from market_radar.ingestion.pipeline import RunSummary
from market_radar.ingestion.universe import build_universe


def test_universe_defaults_and_custom_symbols():
    defaults = build_universe()
    assert len(defaults) == 11
    assert defaults[-1].symbol == "VNINDEX" and defaults[-1].asset_type == "index"
    assert defaults[-1].source == "vci"
    custom = build_universe(["fpt", "SSI", "fpt"])
    assert [asset.symbol for asset in custom] == ["FPT", "SSI", "VNINDEX"]
    assert custom[0].sector == "Technology"
    with pytest.raises(ValueError, match="Invalid stock symbol"):
        build_universe(["../secrets"])


def test_cli_parses_window_symbols_and_refresh(monkeypatch):
    observed = {}

    class FakePipeline:
        def run(self, assets, start, end, refresh):
            observed.update(assets=assets, start=start, end=end, refresh=refresh)
            return RunSummary()

    monkeypatch.setattr(download_data, "HistoricalDataPipeline", FakePipeline)
    code = download_data.main([
        "--symbols", "fpt,SSI", "VCB", "--start", "2025-01-01",
        "--end", "2025-02-01", "--refresh",
    ])
    assert code == 0
    assert [asset.symbol for asset in observed["assets"]] == ["FPT", "SSI", "VCB", "VNINDEX"]
    assert observed["start"] == date(2025, 1, 1)
    assert observed["end"] == date(2025, 2, 1)
    assert observed["refresh"] is True


def test_end_without_start_uses_three_year_lookback(monkeypatch):
    observed = {}

    class FakePipeline:
        def run(self, assets, start, end, refresh):
            observed.update(start=start, end=end)
            return RunSummary()

    monkeypatch.setattr(download_data, "HistoricalDataPipeline", FakePipeline)
    assert download_data.main(["--end", "2025-02-01"]) == 0
    assert (observed["end"] - observed["start"]).days == 1095


def test_cli_returns_failure_status_and_rejects_bad_dates(monkeypatch):
    class FakePipeline:
        def run(self, assets, start, end, refresh):
            return RunSummary(failures={"FPT": "provider unavailable"})

    monkeypatch.setattr(download_data, "HistoricalDataPipeline", FakePipeline)
    assert download_data.main(["--start", "2025-01-01", "--end", "2025-02-01"]) == 1
    with pytest.raises(SystemExit) as exc:
        download_data.main(["--start", "01/01/2025"])
    assert exc.value.code == 2


def test_cli_marks_tier_capped_history_incomplete(monkeypatch):
    class FakePipeline:
        def run(self, assets, start, end, refresh):
            return RunSummary(incomplete={"FPT": "tier cap"})

    monkeypatch.setattr(download_data, "HistoricalDataPipeline", FakePipeline)
    assert download_data.main(["--start", "2025-01-01", "--end", "2025-02-01"]) == 1
