"""Tests del contrato de features y de la segmentación."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clustering import build_segment_labels
from src.features import FEATURE_COLUMNS, get_features, scale_features


def test_feature_columns_order_is_stable(processed_df: pd.DataFrame):
    features = get_features(processed_df)
    assert list(features.columns) == FEATURE_COLUMNS
    assert len(features) == len(processed_df)


def test_get_features_raises_on_missing_column(processed_df: pd.DataFrame):
    import pytest

    with pytest.raises(ValueError, match="Faltan columnas"):
        get_features(processed_df.drop(columns=["age"]))


def test_scaler_centers_the_data(processed_df: pd.DataFrame):
    scaled, scaler = scale_features(get_features(processed_df))
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-6)
    assert hasattr(scaler, "transform")


def test_segment_labels_describe_behavior(processed_df: pd.DataFrame):
    profiles = pd.DataFrame(
        {
            "customer_segment": [0, 1, 2, 3],
            "size": [50, 50, 50, 50],
            "age": [30, 45, 40, 50],
            "annual_income_k": [25.0, 90.0, 50.0, 88.0],
            "spending_score": [20.0, 85.0, 52.0, 18.0],
            "gender_ratio_male": [0.5, 0.5, 0.5, 0.5],
        }
    )
    labels = build_segment_labels(profiles, processed_df)

    assert labels[0]["label"] == "Bajo valor"
    assert labels[1]["label"] == "Premium"
    assert labels[2]["label"] == "Estándar"
    assert labels[3]["label"] == "Conservador"


def test_segment_labels_are_unique(processed_df: pd.DataFrame):
    profiles = pd.DataFrame(
        {
            "customer_segment": [0, 1],
            "size": [50, 50],
            "age": [25, 55],
            "annual_income_k": [20.0, 20.0],
            "spending_score": [80.0, 80.0],
            "gender_ratio_male": [0.5, 0.5],
        }
    )
    labels = build_segment_labels(profiles, processed_df)
    names = [value["label"] for value in labels.values()]
    assert len(names) == len(set(names))
