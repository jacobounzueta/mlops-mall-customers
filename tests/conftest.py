"""Fixtures compartidas por la suite de tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Debe configurarse antes de importar api.main: la API lee estas variables al
# cargar el módulo. Evitamos que los tests intenten conectarse a MLflow.
_TMP_DIR = Path(tempfile.mkdtemp(prefix="mlops-tests-"))
os.environ["MODEL_AUTOLOAD"] = "false"
os.environ["PREDICTIONS_LOG_PATH"] = str(_TMP_DIR / "predictions.csv")


@pytest.fixture(scope="session")
def predictions_log_path() -> Path:
    return Path(os.environ["PREDICTIONS_LOG_PATH"])


@pytest.fixture(scope="session")
def processed_df() -> pd.DataFrame:
    from src.data import preprocess_data

    return preprocess_data(download=False)


@pytest.fixture(autouse=True)
def clean_predictions_log(predictions_log_path: Path):
    predictions_log_path.unlink(missing_ok=True)
    yield
    predictions_log_path.unlink(missing_ok=True)
