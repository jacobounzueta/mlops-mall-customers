"""API REST de clasificación de segmentos de clientes.

Expone el modelo ``champion`` registrado en MLflow, instrumenta métricas de
Prometheus, registra cada predicción en un CSV (fuente para el análisis de
drift) y documenta el contrato con Pydantic.

Uso:
    uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from mlflow.tracking import MlflowClient
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from src.config import (
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    PREDICTIONS_LOG_PATH,
    REGISTERED_MODEL_NAME,
    SEGMENT_LABELS_PATH,
)
from src.drift import compute_drift_for_monitoring, prediction_log_size
from src.features import FEATURE_COLUMNS

START_TIME = time.time()

# =============================================================================
# Métricas de Prometheus
# =============================================================================
REQUEST_COUNT = Counter(
    "mall_customer_http_requests_total",
    "Peticiones HTTP recibidas por endpoint y código de estado",
    ["method", "endpoint", "status_code"],
)
PREDICTION_COUNT = Counter(
    "mall_customer_predictions_total",
    "Predicciones realizadas por segmento",
    ["segment", "label"],
)
PREDICTION_LATENCY = Histogram(
    "mall_customer_prediction_latency_seconds",
    "Latencia de la predicción del modelo en segundos",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTION_ERRORS = Counter(
    "mall_customer_prediction_errors_total",
    "Errores de predicción por motivo",
    ["reason"],
)
FEATURE_PSI = Gauge(
    "mall_customer_feature_psi",
    "Population Stability Index por feature (producción vs entrenamiento)",
    ["feature"],
)
DRIFT_SAMPLES = Gauge(
    "mall_customer_drift_samples",
    "Cantidad de predicciones usadas para calcular el PSI",
)
MODEL_LOADED = Gauge(
    "mall_customer_model_loaded",
    "1 si el modelo champion está cargado, 0 en caso contrario",
)
BATCH_SIZE = Histogram(
    "mall_customer_batch_size",
    "Cantidad de clientes enviados en peticiones por lote",
    buckets=(1, 5, 10, 25, 50, 100),
)


# =============================================================================
# Estado del modelo
# =============================================================================
class ModelState:
    def __init__(self) -> None:
        self.model = None
        self.error: str | None = None
        self.loaded_at: str | None = None
        self.version: str | None = None
        self.run_id: str | None = None
        self.alias: str = MODEL_ALIAS
        self.name: str = REGISTERED_MODEL_NAME


state = ModelState()
_log_lock = threading.Lock()


def _load_segment_labels() -> dict:
    if not SEGMENT_LABELS_PATH.exists():
        return {}
    return json.loads(SEGMENT_LABELS_PATH.read_text(encoding="utf-8"))


def load_champion_model() -> None:
    """Carga el alias champion con reintentos (MLflow puede tardar en arrancar)."""
    attempts = int(os.getenv("MODEL_LOAD_ATTEMPTS", "5"))
    delay = float(os.getenv("MODEL_LOAD_DELAY_SECONDS", "1.0"))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

    for attempt in range(1, attempts + 1):
        try:
            state.model = mlflow.sklearn.load_model(model_uri)
            state.error = None
            state.loaded_at = datetime.now(UTC).isoformat()

            try:
                client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
                version_info = client.get_model_version_by_alias(
                    REGISTERED_MODEL_NAME, MODEL_ALIAS
                )
                state.version = str(version_info.version)
                state.run_id = version_info.run_id
            except Exception:
                state.version = None
                state.run_id = None

            MODEL_LOADED.set(1)
            print(f"Modelo {model_uri} cargado (versión {state.version})")
            return
        except Exception as exc:
            state.error = str(exc)
            MODEL_LOADED.set(0)
            print(f"Intento {attempt}/{attempts} de cargar el modelo falló: {exc}")
            if attempt < attempts:
                time.sleep(delay)

    print(
        "La API arranca sin modelo. /predict responderá 503 hasta que MLflow "
        "esté disponible y se reinicie el servicio."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("MODEL_AUTOLOAD", "true").lower() == "true":
        load_champion_model()
    yield


app = FastAPI(
    title="Mall Customers Segmentation API",
    description=(
        "Clasifica clientes en segmentos construidos con KMeans y entrenados "
        "con modelos supervisados registrados en MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Schemas
# =============================================================================
class CustomerFeatures(BaseModel):
    """Datos de entrada de un cliente."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "gender": "Male",
                    "age": 28,
                    "annual_income_k": 60,
                    "spending_score": 80,
                }
            ]
        },
    )

    gender: Literal["Male", "Female"] = Field(description="Género del cliente")
    age: int = Field(ge=18, le=100, description="Edad en años")
    annual_income_k: int = Field(
        ge=0, le=200, description="Ingreso anual en miles de dólares"
    )
    spending_score: int = Field(
        ge=1, le=100, description="Puntaje de gasto asignado por el mall (1-100)"
    )


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    customer_segment: int
    segment_label: str
    segment_description: str
    probabilities: dict[str, float]
    model_name: str
    model_version: str | None
    latency_ms: float
    timestamp: str


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model: str
    alias: str
    model_version: str | None
    model_loaded: bool
    error: str | None = None
    uptime_seconds: float


# =============================================================================
# Utilidades internas
# =============================================================================
def _to_feature_frame(customer: CustomerFeatures) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gender": 1 if customer.gender == "Male" else 0,
                "age": customer.age,
                "annual_income_k": customer.annual_income_k,
                "spending_score": customer.spending_score,
            }
        ]
    )[FEATURE_COLUMNS]


def _predict_frame(frame: pd.DataFrame) -> tuple[int, dict[str, float]]:
    if state.model is None:
        PREDICTION_ERRORS.labels(reason="model_not_loaded").inc()
        raise HTTPException(
            status_code=503,
            detail="El modelo champion todavía no está cargado.",
        )

    try:
        prediction = int(state.model.predict(frame)[0])
        probabilities: dict[str, float] = {}

        if hasattr(state.model, "predict_proba"):
            proba = state.model.predict_proba(frame)[0]
            classes = getattr(state.model, "classes_", range(len(proba)))
            probabilities = {
                str(int(label)): round(float(value), 6)
                for label, value in zip(classes, proba, strict=True)
            }
    except Exception as exc:
        PREDICTION_ERRORS.labels(reason="inference_error").inc()
        raise HTTPException(
            status_code=500, detail=f"Error de inferencia: {exc}"
        ) from exc

    return prediction, probabilities


def _log_prediction(
    frame: pd.DataFrame,
    prediction: int,
    label: str,
    latency_ms: float,
) -> None:
    log_path = Path(PREDICTIONS_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = frame.iloc[0].to_dict()
    row.update(
        {
            "customer_segment": prediction,
            "segment_label": label,
            "latency_ms": round(latency_ms, 3),
            "model_version": state.version,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )

    with _log_lock:
        write_header = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _build_response(
    frame: pd.DataFrame,
    prediction: int,
    probabilities: dict[str, float],
    latency_ms: float,
) -> PredictionResponse:
    labels = _load_segment_labels().get("segments", {})
    segment_info = labels.get(str(prediction), {})
    label = segment_info.get("label", f"Segmento {prediction}")
    description = segment_info.get("description", "")

    PREDICTION_COUNT.labels(segment=str(prediction), label=label).inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000)
    _log_prediction(frame, prediction, label, latency_ms)

    return PredictionResponse(
        customer_segment=prediction,
        segment_label=label,
        segment_description=description,
        probabilities=probabilities,
        model_name=state.name,
        model_version=state.version,
        latency_ms=round(latency_ms, 3),
        timestamp=datetime.now(UTC).isoformat(),
    )


# =============================================================================
# Middleware de métricas HTTP
# =============================================================================
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    if request.url.path != "/metrics":
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()
        response.headers["X-Process-Time-Ms"] = f"{elapsed * 1000:.2f}"

    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    PREDICTION_ERRORS.labels(reason="validation_error").inc()
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Datos de entrada inválidos",
            "errors": jsonable_encoder(exc.errors()),
        },
    )


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/", tags=["info"])
def root():
    return {
        "message": "Mall Customers Segmentation API activa",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/model/info",
            "/predict",
            "/predict/batch",
            "/metrics",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    return HealthResponse(
        status="ok" if state.model is not None else "degraded",
        model=state.name,
        alias=state.alias,
        model_version=state.version,
        model_loaded=state.model is not None,
        error=None if state.model is not None else state.error,
        uptime_seconds=round(time.time() - START_TIME, 3),
    )


@app.get("/model/info", tags=["info"])
def model_info():
    labels = _load_segment_labels()
    return {
        "registered_model": state.name,
        "alias": state.alias,
        "version": state.version,
        "run_id": state.run_id,
        "loaded_at": state.loaded_at,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "global_means": labels.get("global_means", {}),
        "segments": labels.get("segments", {}),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(customer: CustomerFeatures):
    start = time.perf_counter()
    frame = _to_feature_frame(customer)
    prediction, probabilities = _predict_frame(frame)
    latency_ms = (time.perf_counter() - start) * 1000
    return _build_response(frame, prediction, probabilities, latency_ms)


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(batch: BatchRequest):
    start = time.perf_counter()
    BATCH_SIZE.observe(len(batch.customers))

    predictions = []
    for customer in batch.customers:
        frame = _to_feature_frame(customer)
        item_start = time.perf_counter()
        prediction, probabilities = _predict_frame(frame)
        latency_ms = (time.perf_counter() - item_start) * 1000
        predictions.append(
            _build_response(frame, prediction, probabilities, latency_ms)
        )

    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
    )


@app.get("/predictions/recent", tags=["monitoring"])
def recent_predictions(limit: int = Query(default=20, ge=1, le=200)):
    log_path = Path(PREDICTIONS_LOG_PATH)
    if not log_path.exists():
        return {"predictions": [], "count": 0}

    try:
        df = pd.read_csv(log_path)
    except Exception:
        # El archivo puede estar siendo escrito o corrupto: no rompemos la API.
        return {"predictions": [], "count": 0}

    if df.empty:
        return {"predictions": [], "count": 0}

    records = df.tail(limit).iloc[::-1].to_dict(orient="records")
    return {"predictions": records, "count": len(records)}


@app.get("/metrics", tags=["monitoring"], include_in_schema=False)
def metrics():
    MODEL_LOADED.set(1 if state.model is not None else 0)
    DRIFT_SAMPLES.set(prediction_log_size())

    for feature, values in compute_drift_for_monitoring().items():
        FEATURE_PSI.labels(feature=feature).set(values["psi"])

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
