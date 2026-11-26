"""Detección de data drift con Population Stability Index (PSI).

El PSI compara la distribución de los datos de producción (log de
predicciones de la API) contra la distribución de referencia (dataset de
entrenamiento). Se calcula por feature y se expone como métrica de
Prometheus y como reporte JSON.

Reglas de interpretación habituales:
    PSI < 0.10  -> sin drift relevante
    0.10 - 0.25 -> drift moderado
    PSI > 0.25  -> drift significativo (reentrenar)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    MIN_DRIFT_SAMPLES,
    PREDICTIONS_LOG_PATH,
    REFERENCE_DISTRIBUTIONS_PATH,
)
from src.features import FEATURE_COLUMNS

EPSILON = 1e-6


def build_reference_distributions(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """Construye bins y proporciones de referencia para cada feature."""
    features: dict = {}

    for column in FEATURE_COLUMNS:
        values = df[column].astype(float).dropna().to_numpy()

        if column == "gender":
            values_int = values.astype(int)
            counts = np.array(
                [(values_int == value).sum() for value in (0, 1)], dtype=float
            )
            features[column] = {
                "type": "categorical",
                "values": [0, 1],
                "proportions": (counts / counts.sum()).round(6).tolist(),
            }
            continue

        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(values, quantiles))
        if len(edges) < 2:
            edges = np.array([values.min(), values.max() + EPSILON])
        edges[0] = -np.inf
        edges[-1] = np.inf

        counts, _ = np.histogram(values, bins=edges)
        counts = np.maximum(counts, 0).astype(float)
        features[column] = {
            "type": "numeric",
            "bins": [float(edge) for edge in edges],
            "proportions": (counts / counts.sum()).round(6).tolist(),
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_rows": int(len(df)),
        "features": features,
    }


def save_reference_distributions(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or REFERENCE_DISTRIBUTIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = build_reference_distributions(df)
    path.write_text(
        json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def load_reference_distributions(path: Path | None = None) -> dict:
    path = path or REFERENCE_DISTRIBUTIONS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero: python -m src.clustering"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def population_stability_index(
    expected_proportions: list[float], actual_proportions: list[float]
) -> float:
    """PSI entre dos distribuciones de proporciones alineadas por bin."""
    expected = np.clip(np.asarray(expected_proportions, dtype=float), EPSILON, None)
    actual = np.clip(np.asarray(actual_proportions, dtype=float), EPSILON, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _actual_proportions(series: pd.Series, reference_feature: dict) -> list[float]:
    values = series.astype(float).dropna().to_numpy()
    if len(values) == 0:
        return [0.0] * len(reference_feature["proportions"])

    if reference_feature["type"] == "categorical":
        categories = reference_feature["values"]
        counts = np.array(
            [(values.astype(int) == category).sum() for category in categories],
            dtype=float,
        )
    else:
        counts, _ = np.histogram(values, bins=reference_feature["bins"])
        counts = counts.astype(float)

    return (counts / max(counts.sum(), 1)).tolist()


def compute_drift(reference: dict, current_df: pd.DataFrame) -> dict:
    """Calcula PSI por feature y clasifica la severidad."""
    results: dict = {}

    for column in FEATURE_COLUMNS:
        reference_feature = reference["features"][column]
        actual = _actual_proportions(current_df[column], reference_feature)
        psi_value = population_stability_index(reference_feature["proportions"], actual)
        if psi_value < 0.10:
            status = "stable"
        elif psi_value < 0.25:
            status = "moderate"
        else:
            status = "significant"
        results[column] = {
            "psi": round(psi_value, 6),
            "status": status,
        }

    return results


def load_predictions_log(path: Path | None = None) -> pd.DataFrame:
    path = path or PREDICTIONS_LOG_PATH
    if not path.exists():
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, "timestamp"])
    return pd.read_csv(path)


def prediction_log_size() -> int:
    """Cantidad de predicciones registradas (0 si no hay log)."""
    try:
        return int(len(load_predictions_log()))
    except Exception:
        return 0


def compute_drift_for_monitoring() -> dict:
    """PSI de los datos de producción actuales (tolerante a fallos)."""
    try:
        log = load_predictions_log()
        if len(log) < MIN_DRIFT_SAMPLES:
            return {}
        reference = load_reference_distributions()
        return compute_drift(reference, log)
    except Exception:
        return {}


def main() -> None:
    reference = load_reference_distributions()
    log = load_predictions_log()

    if log.empty:
        print("Todavía no hay predicciones registradas. Nada que comparar.")
        return

    results = compute_drift(reference, log)

    print(f"Predicciones analizadas: {len(log)}")
    if len(log) < MIN_DRIFT_SAMPLES:
        print(
            f"Advertencia: se recomiendan al menos {MIN_DRIFT_SAMPLES} "
            "predicciones para que el PSI sea estable."
        )
    print(f"{'feature':<20}{'PSI':>10}  estado")
    for feature, values in results.items():
        print(f"{feature:<20}{values['psi']:>10.4f}  {values['status']}")

    output_path = REFERENCE_DISTRIBUTIONS_PATH.parent / "drift_report.json"
    output_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "n_predictions": int(len(log)),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nReporte guardado en: {output_path}")


if __name__ == "__main__":
    main()
