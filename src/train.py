"""Entrenamiento y tracking de modelos con MLflow.

Entrena tres clasificadores supervisados sobre los segmentos generados por
KMeans, registra parámetros, métricas y artefactos en MLflow y guarda una
tabla comparativa en ``reports/metrics``.

Requiere el servidor de MLflow activo (docker compose up -d mlflow).

Uso:
    python -m src.train
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    EXPERIMENT_NAME,
    FIGURES_DIR,
    METRICS_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_ARTIFACT_PATH,
    RANDOM_STATE,
    SEGMENTED_DATA_PATH,
    TEST_SIZE,
)
from src.features import FEATURE_COLUMNS, TARGET_COLUMN

sns.set_theme(style="whitegrid")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def load_segmented_data() -> pd.DataFrame:
    if not SEGMENTED_DATA_PATH.exists():
        raise FileNotFoundError(
            "No existe el dataset segmentado. Ejecuta primero: python -m src.clustering"
        )
    return pd.read_csv(SEGMENTED_DATA_PATH)


def save_confusion_matrix(y_test, y_pred, model_name: str) -> Path:
    labels = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Matriz de confusión - {model_name}")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.tight_layout()

    output_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path, dpi=140)
    plt.close()
    return output_path


def evaluate(y_test, y_pred, y_proba=None, classes=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    # La accuracy puede empatar en 1.0 porque el target se deriva de dos
    # features; log_loss desempata usando la calidad de las probabilidades.
    if y_proba is not None and classes is not None:
        metrics["log_loss"] = log_loss(y_test, y_proba, labels=classes)
    return metrics


def train_and_log_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    data_version: dict,
) -> dict:
    with mlflow.start_run(run_name=model_name) as run:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        metrics = evaluate(y_test, y_pred, y_proba, classes=pipeline.classes_)

        # ------------------------------------------------------------------
        # Parámetros, tags y métricas
        # ------------------------------------------------------------------
        loggable_params = {
            key: value
            for key, value in model.get_params().items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        loggable_params["model_name"] = model_name
        mlflow.log_params(loggable_params)
        mlflow.set_tags(
            {
                "model_family": model_name,
                "git_commit": git_commit(),
                "dataset_rows": data_version["n_rows"],
                "n_clusters": data_version["n_clusters"],
                "author": "Jacobo Unzueta / Rommel Prieto",
            }
        )
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})

        # ------------------------------------------------------------------
        # Artefactos: reporte, matriz de confusión y dataset
        # ------------------------------------------------------------------
        report = classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        report_path = METRICS_DIR / f"classification_report_{model_name}.csv"
        report_df.to_csv(report_path)
        mlflow.log_artifact(str(report_path), artifact_path="metrics")

        cm_path = save_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(str(cm_path), artifact_path="figures")

        schema = {
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "n_clusters": data_version["n_clusters"],
            "segment_labels": data_version["segment_labels"],
        }
        mlflow.log_dict(schema, "schema.json")

        # ------------------------------------------------------------------
        # Modelo con firma e input_example (contrato para el serving)
        # ------------------------------------------------------------------
        signature = mlflow.models.infer_signature(
            X_train.head(20), pipeline.predict(X_train.head(20))
        )
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=MODEL_ARTIFACT_PATH,
            signature=signature,
            input_example=X_train.head(3),
            # MLflow 3 serializa con skops por defecto y exige autorizar tipos
            # de árboles de decisión. Como el artefacto lo generamos nosotros,
            # usamos cloudpickle para mantener el pipeline portable.
            serialization_format="cloudpickle",
        )

        print(f"Modelo: {model_name}")
        print(f"  accuracy: {metrics['accuracy']:.4f}")
        print(f"  f1_weighted: {metrics['f1_weighted']:.4f}")
        print(f"  run_id: {run.info.run_id}")

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,
            **{key: round(float(value), 6) for key, value in metrics.items()},
        }


def train_models() -> pd.DataFrame:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_segmented_data()

    labels_path = METRICS_DIR / "segment_labels.json"
    segment_labels = {}
    if labels_path.exists():
        segment_labels = json.loads(labels_path.read_text(encoding="utf-8"))["segments"]

    data_version = {
        "n_rows": int(len(df)),
        "n_clusters": int(df[TARGET_COLUMN].nunique()),
        "segment_labels": {
            str(key): value["label"] for key, value in segment_labels.items()
        },
    }

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=500, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = []
    for model_name, model in models.items():
        results.append(
            train_and_log_model(
                model_name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                data_version,
            )
        )

    comparison = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
    comparison_path = METRICS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\nComparación de modelos (ordenada por F1 weighted):")
    print(comparison.to_string(index=False))
    print(f"\nTabla guardada en: {comparison_path}")

    return comparison


if __name__ == "__main__":
    train_models()
