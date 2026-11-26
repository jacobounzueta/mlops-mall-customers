"""Tests de ingesta, limpieza y validación del dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import (
    COLUMN_RANGES,
    EXPECTED_COLUMNS,
    clean_column_names,
    validate_data,
)


def test_dataset_shape_and_columns(processed_df: pd.DataFrame):
    assert len(processed_df) == 200
    assert list(processed_df.columns) == EXPECTED_COLUMNS


def test_dataset_has_no_nulls(processed_df: pd.DataFrame):
    assert processed_df.isnull().sum().sum() == 0


def test_gender_is_binary(processed_df: pd.DataFrame):
    assert set(processed_df["gender"].unique()) == {0, 1}


def test_numeric_ranges(processed_df: pd.DataFrame):
    for column, (low, high) in COLUMN_RANGES.items():
        assert processed_df[column].between(low, high).all()


def test_clean_column_names_normalizes_headers():
    raw = pd.DataFrame(
        {
            "CustomerID": [1],
            "Gender": ["Male"],
            "Age": [30],
            "Annual Income (k$)": [60],
            "Spending Score (1-100)": [50],
        }
    )
    cleaned = clean_column_names(raw)
    assert "customerid" in cleaned.columns
    assert "annual_income_k" in cleaned.columns
    assert "spending_score" in cleaned.columns


def test_validate_data_detects_missing_columns():
    incomplete = pd.DataFrame({"gender": ["Male"], "age": [30]})
    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_data(incomplete)


def test_validate_data_detects_out_of_range():
    invalid = pd.DataFrame(
        {
            "gender": ["Male"],
            "age": [30],
            "annual_income_k": [999],
            "spending_score": [50],
        }
    )
    with pytest.raises(ValueError, match="fuera del rango"):
        validate_data(invalid)


def test_validate_data_detects_invalid_gender():
    invalid = pd.DataFrame(
        {
            "gender": ["Otro"],
            "age": [30],
            "annual_income_k": [60],
            "spending_score": [50],
        }
    )
    with pytest.raises(ValueError, match="gender"):
        validate_data(invalid)
