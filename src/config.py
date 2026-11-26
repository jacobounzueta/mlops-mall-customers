"""Configuración centralizada del proyecto.

Todas las rutas, nombres y parámetros del pipeline viven aquí para evitar
valores duplicados entre entrenamiento, API y frontend. Las variables de
entorno permiten sobrescribir la configuración sin tocar código (por
ejemplo dentro de Docker Compose).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# La consola de Windows usa cp1252 por defecto y MLflow imprime emojis en las
# URLs de los runs; forzamos UTF-8 para evitar UnicodeEncodeError.
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - entornos sin consola real
        pass

ROOT_DIR = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MONITORING_DIR = DATA_DIR / "monitoring"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "Mall_Customers.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "mall_customers_processed.csv"
SEGMENTED_DATA_PATH = PROCESSED_DATA_DIR / "mall_customers_segmented.csv"

SEGMENT_LABELS_PATH = METRICS_DIR / "segment_labels.json"
REFERENCE_DISTRIBUTIONS_PATH = METRICS_DIR / "reference_distributions.json"
CLUSTERING_METRICS_PATH = METRICS_DIR / "clustering_metrics.json"

PREDICTIONS_LOG_PATH = Path(
    os.getenv(
        "PREDICTIONS_LOG_PATH",
        str(MONITORING_DIR / "api_predictions_log.csv"),
    )
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://raw.githubusercontent.com/tirthajyoti/"
    "Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
)
DATASET_URL_FALLBACK = (
    "https://raw.githubusercontent.com/kennethleungty/"
    "Mall-Customer-Segmentation/master/Mall_Customers.csv"
)

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mall-customers-classification")
REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL", "mall-customer-segment-classifier"
)
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
MODEL_ARTIFACT_PATH = "model"

# ---------------------------------------------------------------------------
# API / Frontend
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://{API_HOST}:{API_PORT}")

# ---------------------------------------------------------------------------
# Parámetros del pipeline
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CLUSTERS = 4
K_RANGE = range(2, 9)

# El PSI es inestable con muestras pequeñas: no se publica hasta tener al menos
# esta cantidad de predicciones registradas.
MIN_DRIFT_SAMPLES = int(os.getenv("MIN_DRIFT_SAMPLES", "30"))

# Variables usadas para construir los segmentos. Se excluye gender a propósito:
# al incluirla, KMeans separa los clusters casi perfectamente por género
# (silhouette 0.298), lo que produce segmentos sin valor de negocio y un
# clasificador trivial. Con variables conductuales el silhouette sube a ~0.49.
CLUSTERING_FEATURES = ["annual_income_k", "spending_score"]
