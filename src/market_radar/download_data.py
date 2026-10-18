"""Command line entry point for the historical daily OHLCV pipeline."""

import argparse
from datetime import date, timedelta

from market_radar.ingestion.pipeline import HistoricalDataPipeline, default_window, latest_completed_session_cutoff
from market_radar.ingestion.universe import build_universe


def _date_argument(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
        return parsed
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected date in YYYY-MM-DD format") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download daily Vietnamese stock OHLCV and VNINDEX")
    parser.add_argument("--symbols", nargs="+", metavar="SYMBOL", help="Stock symbols, separated by spaces or commas; VNINDEX is always included")
    parser.add_argument("--start", type=_date_argument, help="First requested date (YYYY-MM-DD)")
    parser.add_argument("--end", type=_date_argument, help="Last requested date (YYYY-MM-DD)")
    parser.add_argument("--refresh", action="store_true", help="Ignore matching local caches and download again")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _, default_end = default_window()
    end = args.end or default_end
    start = args.start or end - timedelta(days=3 * 365)
    if start > end:
        parser.error("--start must be on or before --end")
    if end > latest_completed_session_cutoff():
        parser.error("--end cannot be after the latest completed Vietnam market day")
    symbols = None if args.symbols is None else [piece for item in args.symbols for piece in item.split(",")]
    try:
        assets = build_universe(symbols)
    except ValueError as exc:
        parser.error(str(exc))
    summary = HistoricalDataPipeline().run(assets, start, end, refresh=args.refresh)
    complete = len(summary.results) - len(summary.incomplete)
    print(f"Finished: {complete} complete, {len(summary.incomplete)} partial, {len(summary.failures)} failed")
    return 1 if summary.failures or summary.incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
