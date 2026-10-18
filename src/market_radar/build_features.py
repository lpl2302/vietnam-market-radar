"""Offline CLI for building the Task 003 financial feature table."""

import argparse
from pathlib import Path
import sys

from market_radar.config import FEATURE_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from market_radar.features.storage import build_from_cache
from market_radar.ingestion.universe import build_universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build offline financial features from cached OHLCV CSVs")
    parser.add_argument("--symbols", nargs="+", metavar="SYMBOL", help="Stocks separated by spaces or commas; default is the ten-stock universe")
    parser.add_argument("--output", type=Path, default=FEATURE_DATA_DIR / "market_features.csv", help="Feature CSV path (relative paths start at project root)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    symbols = None if args.symbols is None else [piece for item in args.symbols for piece in item.split(",")]
    try:
        assets = build_universe(symbols)
    except ValueError as exc:
        parser.error(str(exc))
    if len(assets) < 2:
        parser.error("Select at least one stock symbol")
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if output.suffix.lower() != ".csv":
        parser.error("--output must be a .csv path")
    try:
        frame = build_from_cache(assets, PROCESSED_DATA_DIR, output)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Feature build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} rows ({int(frame['model_ready'].sum())} model-ready) to {output}")
    for symbol, group in frame.groupby("symbol", sort=False):
        print(f"{symbol}: {len(group)} rows, {int(group['model_ready'].sum())} model-ready, {group['date'].min()} to {group['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
