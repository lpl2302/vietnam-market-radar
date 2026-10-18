"""Load validated local caches and write reproducible feature outputs."""

from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import pandas as pd

from market_radar.features.engineering import MODEL_FEATURE_COLUMNS, build_feature_table
from market_radar.ingestion.schema import validate_canonical
from market_radar.ingestion.universe import Asset


def _csv_path(metadata_path: Path) -> Path:
    return metadata_path.with_name(metadata_path.name.removesuffix(".meta.json") + ".csv")


def load_processed_asset(asset: Asset, processed_dir: Path) -> tuple[pd.DataFrame, dict[str, object], Path]:
    """Choose the newest ending session, then the longest observed coverage."""
    candidates: list[tuple[tuple[str, int, int, str], Path, dict[str, object]]] = []
    for path in processed_dir.glob(f"{asset.asset_type}_{asset.symbol}_*.meta.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 2 or metadata.get("symbol") != asset.symbol or metadata.get("asset_type") != asset.asset_type:
            raise ValueError(f"Unexpected ingestion metadata: {path}")
        first = date.fromisoformat(str(metadata["first_observation"]))
        last = date.fromisoformat(str(metadata["last_observation"]))
        rank = (last.isoformat(), (last - first).days, int(metadata["normalized_rows"]), str(metadata["retrieved_at_utc"]))
        candidates.append((rank, path, metadata))
    if not candidates:
        raise FileNotFoundError(f"No processed OHLCV cache for {asset.symbol} in {processed_dir}")

    _, metadata_path, metadata = max(candidates, key=lambda item: item[0])
    csv_path = _csv_path(metadata_path)
    frame = pd.read_csv(csv_path, keep_default_na=False)
    frame = validate_canonical(
        frame, asset,
        date.fromisoformat(str(metadata["start"])),
        date.fromisoformat(str(metadata["end"])),
    )
    if len(frame) != metadata["normalized_rows"] or frame["date"].iloc[0] != metadata["first_observation"] or frame["date"].iloc[-1] != metadata["last_observation"]:
        raise ValueError(f"Processed CSV and metadata disagree: {csv_path}")
    return frame, metadata, csv_path


def _atomic_write(path: Path, writer: Callable[[Path], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_from_cache(assets: list[Asset], processed_dir: Path, output_path: Path) -> pd.DataFrame:
    """Build features entirely offline and write CSV plus provenance sidecar."""
    equities = [asset for asset in assets if asset.asset_type == "equity"]
    indices = [asset for asset in assets if asset.asset_type == "index"]
    if not equities or len(indices) != 1 or indices[0].symbol != "VNINDEX":
        raise ValueError("Choose at least one stock and the VNINDEX benchmark")

    input_frames: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    for asset in [*equities, indices[0]]:
        frame, metadata, csv_path = load_processed_asset(asset, processed_dir)
        input_frames.append(frame)
        sources.append({
            "symbol": asset.symbol,
            "csv": csv_path.name,
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "provider": metadata["provider"],
            "provider_source": metadata["provider_source"],
            "retrieved_at_utc": metadata["retrieved_at_utc"],
            "requested_start": metadata["start"],
            "requested_end": metadata["end"],
            "first_observation": metadata["first_observation"],
            "last_observation": metadata["last_observation"],
            "rows": len(frame),
            "source_complete": metadata["complete"],
            "source_limitation": metadata["limitation"],
        })

    combined_stocks = pd.concat(input_frames[:-1], ignore_index=True)
    result = build_feature_table(combined_stocks, input_frames[-1])
    sidecar = output_path.with_suffix(".meta.json")
    provenance = {
        "feature_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_rows": len(result),
        "model_ready_rows": int(result["model_ready"].sum()),
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "rolling_windows": "20 preceding trading observations per stock; current session excluded from comparison baselines",
        "benchmark_alignment": "exact VNINDEX trading date; no forward fill",
        "sources": sources,
    }
    _atomic_write(output_path, lambda path: result.to_csv(path, index=False))
    _atomic_write(sidecar, lambda path: path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8"))
    return result
