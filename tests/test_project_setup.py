"""Checks that the package layout and project paths work after installation."""

import importlib

from market_radar import __version__
from market_radar.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR


def test_package_and_project_paths_are_independent_of_working_directory(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    assert __version__ == "0.1.0"
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert RAW_DATA_DIR == PROJECT_ROOT / "data" / "raw"
    assert PROCESSED_DATA_DIR == PROJECT_ROOT / "data" / "processed"
    assert ARTIFACTS_DIR == PROJECT_ROOT / "artifacts"

    for module in ("ingestion", "features", "modeling", "visualization"):
        importlib.import_module(f"market_radar.{module}")
