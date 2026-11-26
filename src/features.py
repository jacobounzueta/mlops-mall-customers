"""Definición del contrato de features del modelo."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "gender",
    "age",
    "annual_income_k",
    "spending_score",
]

TARGET_COLUMN = "customer_segment"


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo las columnas usadas por el modelo, en orden fijo."""
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de features: {missing}")
    return df[FEATURE_COLUMNS].copy()


def scale_features(X: pd.DataFrame):
    """Escala features con StandardScaler (el scaler se reutiliza en el pipeline)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
