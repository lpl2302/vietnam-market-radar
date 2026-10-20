"""Generate offline sector and retrospective event-study tables."""

import argparse
from pathlib import Path

from market_radar.analysis.storage import save_analysis
from market_radar.config import ANALYSIS_DATA_DIR, ANOMALY_DATA_DIR, ARTIFACTS_DIR, FEATURE_DATA_DIR
from market_radar.visualization.data import DashboardDataError, load_dashboard_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ANOMALY_DATA_DIR / "market_anomalies.csv")
    parser.add_argument("--evaluation", type=Path, default=ANOMALY_DATA_DIR / "evaluation.json")
    parser.add_argument("--features", type=Path, default=FEATURE_DATA_DIR / "market_features.csv")
    parser.add_argument("--model-metadata", type=Path, default=ARTIFACTS_DIR / "isolation_forest.meta.json")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DATA_DIR)
    args = parser.parse_args(argv)
    try:
        data = load_dashboard_data(args.results, args.evaluation, args.features, args.model_metadata)
        paths = save_analysis(data.results, args.results, args.output_dir,
                              feature_sha256=data.model_metadata["source_feature_sha256"])
    except (OSError, ValueError, DashboardDataError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(f"Saved {len(data.results[data.results.model_ready]):,} scored outcomes and "
          f"{data.results.sector.nunique()} covered-sector summaries to {args.output_dir}")
    print(f"Event outcomes: {paths['outcomes']}")
    print(f"Holdout comparisons: {paths['comparisons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
