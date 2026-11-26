"""Ingesta y limpieza del dataset Mall Customers.

El dataset original no trae variable objetivo: en ``src.clustering`` se
construyen los segmentos con KMeans y luego se entrenan clasificadores
supervisados sobre esa etiqueta.

Uso:
    python -m src.data            # descarga (si falta), limpia y guarda
    python -m src.data --no-download
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import (
    DATASET_URL,
    DATASET_URL_FALLBACK,
    METRICS_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)

EXPECTED_COLUMNS = ["gender", "age", "annual_income_k", "spending_score"]

# Rango válido del negocio (se usa como validación de calidad de datos).
COLUMN_RANGES = {
    "age": (18, 100),
    "annual_income_k": (0, 200),
    "spending_score": (1, 100),
}
GENDER_VALUES = {"Male", "Female"}


def download_dataset(force: bool = False) -> Path:
    """Descarga el CSV público si no existe localmente."""
    if RAW_DATA_PATH.exists() and not force:
        return RAW_DATA_PATH

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    for url in (DATASET_URL, DATASET_URL_FALLBACK):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
            RAW_DATA_PATH.write_bytes(content)
            print(f"Dataset descargado desde {url}")
            return RAW_DATA_PATH
        except Exception as exc:  # pragma: no cover - depende de la red
            print(f"No se pudo descargar desde {url}: {exc}")

    raise RuntimeError(
        "No fue posible descargar el dataset. Coloca Mall_Customers.csv "
        f"manualmente en {RAW_DATA_PATH}"
    )


def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {RAW_DATA_PATH}. Ejecuta: python -m src.data"
        )
    return pd.read_csv(RAW_DATA_PATH)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza encabezados a snake_case sin caracteres especiales."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("$", "", regex=False)
    )
    # Nombres explícitos para que el contrato del modelo sea estable.
    df = df.rename(
        columns={
            "annual_income_k": "annual_income_k",
            "annual_income_k_": "annual_income_k",
            "spending_score_1_100": "spending_score",
        }
    )
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Valida el contrato del dataset y devuelve un resumen de calidad."""
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Faltan columnas esperadas: {missing_columns}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    if df[EXPECTED_COLUMNS].isnull().any().any():
        nulls = df[EXPECTED_COLUMNS].isnull().sum().to_dict()
        raise ValueError(f"El dataset contiene valores nulos: {nulls}")

    unknown_gender = set(df["gender"].astype(str).unique()) - GENDER_VALUES
    if unknown_gender:
        raise ValueError(
            f"La columna gender tiene valores no soportados: {unknown_gender}"
        )

    for column, (low, high) in COLUMN_RANGES.items():
        out_of_range = df[~df[column].between(low, high)]
        if not out_of_range.empty:
            raise ValueError(
                f"La columna {column} tiene {len(out_of_range)} valores "
                f"fuera del rango [{low}, {high}]"
            )

    gender_counts = df["gender"].astype(str).value_counts(normalize=True)
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "nulls": int(df[EXPECTED_COLUMNS].isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "gender_distribution": {
            str(key): round(float(value), 4) for key, value in gender_counts.items()
        },
        "ranges": {
            column: {
                "min": float(df[column].min()),
                "max": float(df[column].max()),
                "mean": round(float(df[column].mean()), 3),
            }
            for column in COLUMN_RANGES
        },
    }
    return summary


def preprocess_data(download: bool = True) -> pd.DataFrame:
    """Descarga, limpia, valida y persiste el dataset procesado."""
    if download:
        download_dataset()

    df = clean_column_names(load_raw_data())

    if "customerid" in df.columns:
        df = df.drop(columns=["customerid"])

    df = df.drop_duplicates().reset_index(drop=True)

    df["gender"] = df["gender"].astype(str).str.strip().str.title()

    summary = validate_data(df)

    gender_map = {"Male": 1, "Female": 0}
    df = df[EXPECTED_COLUMNS].copy()
    df["gender"] = df["gender"].map(gender_map).astype("int64")
    for column in EXPECTED_COLUMNS[1:]:
        df[column] = df[column].astype("int64")

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "data_quality.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta y limpieza del dataset")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="No descargar el CSV; usar el que ya está en data/raw",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Volver a descargar el CSV aunque ya exista",
    )
    args = parser.parse_args()

    if args.force_download:
        download_dataset(force=True)

    processed = preprocess_data(download=not args.no_download)
    print(processed.head())
    print(processed.info())
    print(f"\nDataset procesado guardado en: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
