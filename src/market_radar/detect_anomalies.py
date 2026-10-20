"""Train and evaluate the offline pooled anomaly detector."""

import argparse
from pathlib import Path

import pandas as pd

from market_radar.config import ANOMALY_DATA_DIR, ARTIFACTS_DIR, FEATURE_DATA_DIR
from market_radar.modeling.detection import DetectionConfig, detect_anomalies
from market_radar.modeling.evaluation import evaluate_run
from market_radar.modeling.persistence import save_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FEATURE_DATA_DIR / "market_features.csv")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=ANOMALY_DATA_DIR)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        config = DetectionConfig(n_estimators=args.n_estimators, contamination=args.contamination,
                                 random_state=args.random_state)
        frame = pd.read_csv(args.input)
        run = detect_anomalies(frame, config)
        evaluation = evaluate_run(run)
        paths = save_run(run, evaluation, args.input, args.artifact_dir, args.output_dir)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.exit(1, f"Anomaly detection failed: {exc}\n")

    print(f"Training: {evaluation['by_partition']['train']['observations']} observations, "
          f"{run.split.train_dates[0]} to {run.split.train_dates[-1]}")
    print(f"Holdout: {evaluation['by_partition']['holdout']['observations']} observations, "
          f"{run.split.holdout_dates[0]} to {run.split.holdout_dates[-1]}")
    print(f"Anomalies: {evaluation['all_scored']['anomalies']} total; "
          f"{evaluation['by_partition']['holdout']['anomalies']} holdout "
          f"({evaluation['by_partition']['holdout']['anomaly_pct']:.3f}%)")
    print(f"Saved model: {paths['model']}")
    print(f"Saved results: {paths['results']}")
    print(f"Saved diagnostics: {paths['evaluation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
