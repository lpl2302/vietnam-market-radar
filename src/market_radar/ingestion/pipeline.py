"""Sequential Vnstock downloads, validation, provenance, and local caching."""

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
from time import sleep
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
from vnstock import Market
from vnstock.core.exceptions import EntitlementError, NetworkError, RateLimitedError

from market_radar.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from market_radar.ingestion.schema import DataValidationError, normalize_ohlcv, validate_canonical
from market_radar.ingestion.universe import Asset


RETRYABLE_ERRORS = (RateLimitedError, NetworkError, TimeoutError, ConnectionError)
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
Fetch = Callable[[Asset, date, date], pd.DataFrame]


def latest_completed_session_cutoff(now: datetime | None = None) -> date:
    """Return the latest weekday whose Vietnam market close is safely past.

    Exchange holidays are left to the provider: no artificial bar is created.
    """
    clock = (now or datetime.now(VIETNAM_TZ)).astimezone(VIETNAM_TZ)
    candidate = clock.date() if clock.time() >= time(16, 0) else clock.date() - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def default_window(now: datetime | None = None) -> tuple[date, date]:
    end = latest_completed_session_cutoff(now)
    return end - timedelta(days=3 * 365), end


def fetch_vnstock(asset: Asset, start: date, end: date) -> pd.DataFrame:
    """Fetch Community-tier daily bars from the pinned Vnstock 5 API."""
    market = Market()
    route = market.equity(asset.symbol) if asset.asset_type == "equity" else market.index(asset.symbol)
    return route.ohlcv(
        start=start.isoformat(), end=end.isoformat(), interval="1D",
        source=asset.source, floating=None,
    )


@dataclass(frozen=True)
class CachePaths:
    raw: Path
    normalized: Path
    metadata: Path


def cache_paths(asset: Asset, start: date, end: date, raw_dir: Path, processed_dir: Path) -> CachePaths:
    stem = f"{asset.asset_type}_{asset.symbol}_{start.isoformat()}_{end.isoformat()}"
    return CachePaths(
        raw_dir / f"{stem}.csv",
        processed_dir / f"{stem}.csv",
        processed_dir / f"{stem}.meta.json",
    )


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        writer(temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class DownloadResult:
    asset: Asset
    rows: int
    path: Path
    cached: bool
    complete: bool
    limitation: str | None = None


@dataclass
class RunSummary:
    results: list[DownloadResult] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    incomplete: dict[str, str] = field(default_factory=dict)


class HistoricalDataPipeline:
    """Download assets one by one; reuse exact-window caches unless refreshed."""

    def __init__(
        self,
        raw_dir: Path = RAW_DATA_DIR,
        processed_dir: Path = PROCESSED_DATA_DIR,
        fetch: Fetch = fetch_vnstock,
        request_delay_seconds: float = 7.0,
        max_attempts: int = 3,
        pause: Callable[[float], None] = sleep,
        report: Callable[[str], None] = print,
    ) -> None:
        if request_delay_seconds < 0 or max_attempts < 1:
            raise ValueError("Delay must be nonnegative and max_attempts must be positive")
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.fetch = fetch
        self.request_delay_seconds = request_delay_seconds
        self.max_attempts = max_attempts
        self.pause = pause
        self.report = report
        self._made_request = False

    def _fetch_with_retries(self, asset: Asset, start: date, end: date) -> pd.DataFrame:
        for attempt in range(1, self.max_attempts + 1):
            if self._made_request:
                self.pause(self.request_delay_seconds)
            self._made_request = True
            try:
                return self.fetch(asset, start, end)
            except RETRYABLE_ERRORS as exc:
                if attempt == self.max_attempts:
                    raise
                wait = min(2 ** attempt, 30)
                self.report(f"{asset.symbol}: transient {type(exc).__name__}; retry {attempt + 1}/{self.max_attempts} in {wait}s")
                self.pause(wait)
            except EntitlementError as exc:
                # Only the short minute quota is worth retrying in this run.
                if getattr(exc, "code", None) != "VNSTOCK_AUTH_MINUTE_LIMIT" or attempt == self.max_attempts:
                    raise
                self.report(f"{asset.symbol}: minute quota reached; retry {attempt + 1}/{self.max_attempts} in 60s")
                self.pause(60)
        raise AssertionError("retry loop exhausted")

    def download_asset(self, asset: Asset, start: date, end: date, refresh: bool = False) -> DownloadResult:
        if start > end:
            raise ValueError("Start date must be on or before end date")
        paths = cache_paths(asset, start, end, self.raw_dir, self.processed_dir)
        if not refresh and all(path.is_file() for path in (paths.raw, paths.normalized, paths.metadata)):
            try:
                metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
                if metadata.get("schema_version") == 2:
                    if metadata["symbol"] != asset.symbol or metadata["start"] != start.isoformat() or metadata["end"] != end.isoformat() or metadata["provider_source"] != asset.source:
                        raise ValueError("cache metadata does not match request")
                    cached = pd.read_csv(paths.normalized, keep_default_na=False)
                    cached = validate_canonical(cached, asset, start, end)
                    return DownloadResult(asset, len(cached), paths.normalized, True, metadata["complete"], metadata["limitation"])
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                raise DataValidationError(f"{asset.symbol}: invalid cache; rerun with --refresh: {exc}") from exc

        raw = self._fetch_with_retries(asset, start, end)
        normalized = normalize_ohlcv(raw, asset, start, end)
        provider_source = raw.attrs.get("source", asset.source)
        caps = [key for key in ("tier_cap_history_days", "tier_cap_rows") if key in raw.attrs]
        limitation = (
            f"Vnstock access tier truncated the requested history ({', '.join(caps)})"
            if caps else None
        )
        metadata = {
            "schema_version": 2,
            "provider": "Vnstock",
            "provider_version": version("vnstock"),
            "provider_source": str(provider_source),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": asset.symbol,
            "asset_type": asset.asset_type,
            "sector": asset.sector,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "interval": "1D",
            "raw_rows": len(raw),
            "normalized_rows": len(normalized.frame),
            "exact_duplicates_removed": normalized.duplicates_removed,
            "first_observation": normalized.frame["date"].iloc[0],
            "last_observation": normalized.frame["date"].iloc[-1],
            "price_unit": "thousand_vnd" if asset.asset_type == "equity" else "points",
            "price_adjustment": "Provider-adjusted for corporate actions" if asset.asset_type == "equity" else "Index points; no equity adjustment",
            "complete": limitation is None,
            "limitation": limitation,
        }
        for key in caps:
            metadata[key] = str(raw.attrs[key])

        _atomic_write(paths.raw, lambda path: raw.to_csv(path, index=False))
        _atomic_write(paths.normalized, lambda path: normalized.frame.to_csv(path, index=False))
        _atomic_write(paths.metadata, lambda path: path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8"))
        return DownloadResult(asset, len(normalized.frame), paths.normalized, False, limitation is None, limitation)

    def run(self, assets: list[Asset], start: date, end: date, refresh: bool = False) -> RunSummary:
        summary = RunSummary()
        for position, asset in enumerate(assets, start=1):
            self.report(f"[{position}/{len(assets)}] {asset.symbol}: checking {start} to {end}")
            try:
                result = self.download_asset(asset, start, end, refresh)
            except Exception as exc:
                summary.failures[asset.symbol] = f"{type(exc).__name__}: {exc}"
                self.report(f"{asset.symbol}: FAILED — {summary.failures[asset.symbol]}")
            else:
                summary.results.append(result)
                state = "cache" if result.cached else "downloaded"
                if not result.complete:
                    summary.incomplete[asset.symbol] = result.limitation or "provider returned an incomplete window"
                    state += " PARTIAL"
                self.report(f"{asset.symbol}: {state} {result.rows} rows -> {result.path}")
                if result.limitation:
                    self.report(f"{asset.symbol}: {result.limitation}")
        return summary
