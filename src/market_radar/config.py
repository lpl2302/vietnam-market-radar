"""Project-relative paths used by future research workflows."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DATA_DIR = PROJECT_ROOT / "data" / "features"
ANOMALY_DATA_DIR = PROJECT_ROOT / "data" / "anomalies"
ANALYSIS_DATA_DIR = PROJECT_ROOT / "data" / "analysis"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
