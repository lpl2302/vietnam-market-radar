"""Save and reload trusted local anomaly models and research outputs."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_radar.features import MODEL_FEATURE_COLUMNS
from market_radar.config import PROJECT_ROOT
from market_radar.modeling.detection import DetectionRun, validate_feature_table


MODEL_FILENAME = "isolation_forest.joblib"
MODEL_METADATA_FILENAME = "isolation_forest.meta.json"
RESULT_FILENAME = "market_anomalies.csv"
EVALUATION_FILENAME = "evaluation.json"
RESULT_METADATA_FILENAME = "market_anomalies.meta.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    try:
        writer(Path(temp_name))
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _write_json(path: Path, data: dict[str, object]) -> None:
    _atomic_write(path, lambda temp: temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"))


def build_model_metadata(run: DetectionRun, source_path: Path) -> dict[str, object]:
    """Record training configuration and source identity for later inference."""
    config = run.config
    return {
        "algorithm": "sklearn.ensemble.IsolationForest",
        "preprocessing": "sklearn.preprocessing.StandardScaler",
        "feature_names": list(MODEL_FEATURE_COLUMNS),
        "hyperparameters": {"n_estimators": config.n_estimators, "contamination": config.contamination,
                            "random_state": config.random_state, "n_jobs": config.n_jobs,
                            "train_fraction": config.train_fraction},
        "random_seed": config.random_state,
        "train_date_range": [run.split.train_dates[0], run.split.train_dates[-1]],
        "holdout_date_range": [run.split.holdout_dates[0], run.split.holdout_dates[-1]],
        "training_samples": len(run.split.train_indices),
        "holdout_samples": len(run.split.holdout_indices),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_versions": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__, "joblib": joblib.__version__},
        "source_feature_csv": str(source_path.resolve().relative_to(PROJECT_ROOT)) if source_path.resolve().is_relative_to(PROJECT_ROOT) else source_path.name,
        "source_feature_sha256": sha256_file(source_path),
        "score_definition": "negative sklearn Pipeline.decision_function; larger means more unusual",
        "flag_definition": "Pipeline.predict == -1 using training-fitted Isolation Forest boundary",
    }


def save_run(run: DetectionRun, evaluation: dict[str, object], source_path: Path,
             artifact_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Save model, metadata, scored table, and descriptive diagnostics."""
    metadata = build_model_metadata(run, source_path)
    paths = {
        "model": artifact_dir / MODEL_FILENAME,
        "model_metadata": artifact_dir / MODEL_METADATA_FILENAME,
        "results": output_dir / RESULT_FILENAME,
        "evaluation": output_dir / EVALUATION_FILENAME,
        "results_metadata": output_dir / RESULT_METADATA_FILENAME,
    }
    _atomic_write(paths["model"], lambda temp: joblib.dump(run.model, temp))
    metadata["model_sha256"] = sha256_file(paths["model"])
    _write_json(paths["model_metadata"], metadata)
    _atomic_write(paths["results"], lambda temp: run.results.to_csv(temp, index=False))
    _write_json(paths["evaluation"], evaluation)
    _write_json(paths["results_metadata"], {
        "model_file": MODEL_FILENAME,
        "model_metadata_file": MODEL_METADATA_FILENAME,
        "model_sha256": metadata["model_sha256"],
        "source_feature_sha256": metadata["source_feature_sha256"],
        "feature_names": metadata["feature_names"],
        "created_at_utc": metadata["created_at_utc"],
        "score_definition": metadata["score_definition"],
    })
    return paths


def load_trusted_model(model_path: Path, metadata_path: Path) -> tuple[Pipeline, dict[str, object]]:
    """Load only a model created locally by a trusted run; joblib is executable data."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_names") != list(MODEL_FEATURE_COLUMNS):
        raise ValueError("Saved model feature names or ordering do not match this package")
    if metadata.get("model_sha256") != sha256_file(model_path):
        raise ValueError("Saved model hash does not match metadata")
    model = joblib.load(model_path)
    if (not isinstance(model, Pipeline) or
            not isinstance(model.named_steps.get("scaler"), StandardScaler) or
            not isinstance(model.named_steps.get("isolation_forest"), IsolationForest)):
        raise ValueError("Saved model is not the expected Isolation Forest pipeline")
    if list(model.feature_names_in_) != metadata["feature_names"]:
        raise ValueError("Saved model feature names do not match metadata")
    return model, metadata


def score_with_model(model: Pipeline, metadata: dict[str, object], frame: pd.DataFrame) -> pd.DataFrame:
    """Apply persisted preprocessing and feature order to ready observations."""
    if metadata.get("feature_names") != list(MODEL_FEATURE_COLUMNS):
        raise ValueError("Saved model feature names or ordering do not match this package")
    validated = validate_feature_table(frame)
    ready = validated.loc[validated["model_ready"]].copy()
    features = ready.loc[:, metadata["feature_names"]]
    ready["anomaly_score"] = -model.decision_function(features)
    ready["anomaly_flag"] = model.predict(features) == -1
    return ready
