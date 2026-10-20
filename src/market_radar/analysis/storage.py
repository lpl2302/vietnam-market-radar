"""Persist offline research tables with source identity and reproducible settings."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import numpy as np

from market_radar.analysis.research import (
    diagnose_symbols, future_outcomes, holdout_comparisons, sector_monthly, sector_summary,
)


class AnalysisDataError(ValueError):
    """Saved analysis is missing, malformed, or based on other anomaly results."""


@dataclass(frozen=True)
class AnalysisBundle:
    sectors: pd.DataFrame
    monthly: pd.DataFrame
    outcomes: pd.DataFrame
    comparisons: dict
    diagnostics: dict
    metadata: dict


FILES = {
    "sectors": "sector_summary.csv",
    "monthly": "sector_monthly.csv",
    "outcomes": "event_outcomes.csv",
    "comparisons": "holdout_comparisons.json",
    "diagnostics": "quality_diagnostics.json",
    "metadata": "analysis.meta.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        write(Path(name))
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _json(path: Path, value: dict) -> None:
    _atomic(path, lambda temp: temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"))


def save_analysis(results: pd.DataFrame, source_path: Path, output_dir: Path,
                  *, feature_sha256: str, min_group_size: int = 5) -> dict[str, Path]:
    """Generate separate future outcomes; original anomaly file is read only."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Anomaly results not found: {source_path}")
    sectors = sector_summary(results)
    monthly = sector_monthly(results)
    outcomes = future_outcomes(results)
    comparisons = holdout_comparisons(outcomes, min_group_size=min_group_size)
    diagnostics = diagnose_symbols(results)
    metadata = {
        "analysis_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_results_file": source_path.name,
        "source_results_sha256": sha256_file(source_path),
        "source_feature_sha256": feature_sha256,
        "parameters": {"future_horizons_observed_sessions": [1, 5],
                       "min_subgroup_size": min_group_size,
                       "volume_z_threshold_strictly_above": 3,
                       "absolute_return_z_threshold_strictly_above": 3},
        "scored_observations": int(results.model_ready.sum()),
        "outcome_rows": int(len(outcomes)),
        "sector_coverage": int(len(sectors)),
        "limitation": "Descriptive retrospective analysis of covered stocks; no model fitting or ground-truth labels.",
    }
    paths = {key: output_dir / filename for key, filename in FILES.items()}
    for key, table in (("sectors", sectors), ("monthly", monthly), ("outcomes", outcomes)):
        _atomic(paths[key], lambda temp, table=table: table.to_csv(temp, index=False))
    _json(paths["comparisons"], comparisons)
    _json(paths["diagnostics"], diagnostics)
    _json(paths["metadata"], metadata)
    return paths


def load_analysis(output_dir: Path, results_path: Path, results: pd.DataFrame,
                  *, expected_feature_sha256: str | None = None) -> AnalysisBundle:
    """Load persisted tables and reject outputs from a different anomaly run."""
    paths = {key: output_dir / filename for key, filename in FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise AnalysisDataError("Missing analysis inputs: " + ", ".join(missing))
    try:
        sectors = pd.read_csv(paths["sectors"])
        monthly = pd.read_csv(paths["monthly"])
        outcomes = pd.read_csv(paths["outcomes"])
        comparisons = json.loads(paths["comparisons"].read_text(encoding="utf-8"))
        diagnostics = json.loads(paths["diagnostics"].read_text(encoding="utf-8"))
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise AnalysisDataError(f"Cannot read saved analysis: {exc}") from exc
    if not isinstance(metadata, dict) or not isinstance(diagnostics, dict):
        raise AnalysisDataError("Malformed analysis metadata or diagnostics")
    if (not isinstance(comparisons, dict) or
            not {"holdout_scored_observations", "complete_five_session_observations",
                 "excluded_without_full_window", "cohorts", "subgroups",
                 "mean_absolute_return_differences_pp", "minimum_subgroup_size"} <= comparisons.keys()):
        raise AnalysisDataError("Malformed holdout comparisons")
    if metadata.get("source_results_sha256") != sha256_file(results_path):
        raise AnalysisDataError("Saved analysis is stale for the current anomaly results; rerun build_analysis")
    if expected_feature_sha256 is not None and metadata.get("source_feature_sha256") != expected_feature_sha256:
        raise AnalysisDataError("Saved analysis uses a different feature source; rerun build_analysis")
    required_sectors = {"sector", "covered_stocks", "symbols", "eligible_observations", "anomalies",
                        "anomaly_rate_pct", "mean_daily_return_pct", "daily_return_volatility_pct",
                        "mean_volume_ratio", "latest_observation_date"}
    required_monthly = {"month", "sector", "partition", "eligible_observations", "anomalies", "anomaly_rate_pct"}
    for name, table, required in (("Sector summary", sectors, required_sectors),
                                  ("Sector timeline", monthly, required_monthly)):
        if required - set(table.columns):
            raise AnalysisDataError(f"{name} lacks columns: {sorted(required - set(table.columns))}")
    needed = {"date", "symbol", "sector", "partition", "anomaly_score", "anomaly_flag",
              "baseline_volume_flag", "baseline_return_flag", "next_session_date",
              "next_session_return", "next_five_session_date", "next_five_session_return",
              "next_five_session_abs_return"}
    if needed - set(outcomes.columns):
        raise AnalysisDataError(f"Event outcomes lack columns: {sorted(needed - set(outcomes.columns))}")
    if outcomes.duplicated(["date", "symbol"]).any():
        raise AnalysisDataError("Duplicate event outcome keys")
    scored = results.loc[results.model_ready]
    merged = outcomes[["date", "symbol", "sector", "partition", "anomaly_flag",
                       "baseline_volume_flag", "baseline_return_flag"]].merge(
        scored[["date", "symbol", "sector", "partition", "anomaly_flag",
                "baseline_volume_flag", "baseline_return_flag"]], on=["date", "symbol"],
        how="outer", indicator=True, suffixes=("_outcome", "_result"))
    if ((merged._merge != "both").any() or
            not merged.sector_outcome.equals(merged.sector_result) or
            not merged.partition_outcome.equals(merged.partition_result) or
            any(not merged[f"{name}_outcome"].astype("string").equals(merged[f"{name}_result"].astype("string"))
                for name in ("anomaly_flag", "baseline_volume_flag", "baseline_return_flag"))):
        raise AnalysisDataError("Saved outcomes disagree with current scored observations")
    signed = pd.to_numeric(outcomes.next_five_session_return, errors="coerce")
    absolute = pd.to_numeric(outcomes.next_five_session_abs_return, errors="coerce")
    if not np.allclose(signed.abs(), absolute, equal_nan=True):
        raise AnalysisDataError("Five-session absolute returns disagree with signed returns")
    if int(sectors.eligible_observations.sum()) != len(scored) or int(sectors.anomalies.sum()) != int(scored.anomaly_flag.sum()):
        raise AnalysisDataError("Sector totals disagree with current scored observations")
    if int(monthly.eligible_observations.sum()) != len(scored) or int(monthly.anomalies.sum()) != int(scored.anomaly_flag.sum()):
        raise AnalysisDataError("Sector timeline totals disagree with current scored observations")
    if comparisons["holdout_scored_observations"] != int((scored.partition == "holdout").sum()):
        raise AnalysisDataError("Holdout comparison sample differs from current results")
    return AnalysisBundle(sectors, monthly, outcomes, comparisons, diagnostics, metadata)
