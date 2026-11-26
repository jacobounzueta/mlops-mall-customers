"""Registro del mejor modelo en MLflow Model Registry.

Selecciona el run con mejor ``f1_weighted`` del experimento, lo registra y le
asigna el alias ``champion``, que es el que consume la API.

Uso:
    python -m src.register_model
"""

from __future__ import annotations

import json

import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    EXPERIMENT_NAME,
    METRICS_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_ARTIFACT_PATH,
    REGISTERED_MODEL_NAME,
)


def register_best_model() -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(
            f"No existe el experimento '{EXPERIMENT_NAME}'. "
            "Ejecuta primero: python -m src.train"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_weighted DESC", "metrics.log_loss ASC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No hay runs disponibles en el experimento.")

    best_run = runs[0]
    run_id = best_run.info.run_id
    metrics = best_run.data.metrics

    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
        version=result.version,
    )

    labels_path = METRICS_DIR / "segment_labels.json"
    labels = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))["segments"]

    description = (
        f"Mejor modelo del experimento '{EXPERIMENT_NAME}' "
        f"(run {run_id}). "
        f"f1_weighted={metrics.get('f1_weighted', 0):.4f}, "
        f"accuracy={metrics.get('accuracy', 0):.4f}, "
        f"log_loss={metrics.get('log_loss', 0):.4f}. "
        "Clasifica clientes en los segmentos creados con KMeans."
    )
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=result.version,
        description=description,
    )
    client.update_registered_model(
        name=REGISTERED_MODEL_NAME,
        description=(
            "Clasificador de segmentos de clientes (dataset Mall Customers). "
            "Los segmentos se construyen con KMeans y el modelo final es un "
            "clasificador supervisado con alias 'champion'."
        ),
    )

    for key, value in metrics.items():
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=result.version,
            key=f"metric_{key}",
            value=round(float(value), 6),
        )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=result.version,
        key="run_name",
        value=best_run.data.tags.get("mlflow.runName", ""),
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=result.version,
        key="segment_labels",
        value=json.dumps(
            {key: value["label"] for key, value in labels.items()},
            ensure_ascii=False,
        ),
    )

    summary = {
        "run_id": run_id,
        "model_name": REGISTERED_MODEL_NAME,
        "version": result.version,
        "alias": MODEL_ALIAS,
        "metrics": {key: round(float(value), 6) for key, value in metrics.items()},
    }

    print("Modelo registrado correctamente")
    print(f"  Run ID:  {run_id}")
    print(f"  Modelo:  {REGISTERED_MODEL_NAME}")
    print(f"  Versión: {result.version}")
    print(f"  Alias:   {MODEL_ALIAS}")
    print(f"  F1:      {metrics.get('f1_weighted', 0):.4f}")

    output_path = METRICS_DIR / "registered_model.json"
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Resumen: {output_path}")

    return summary


if __name__ == "__main__":
    register_best_model()
