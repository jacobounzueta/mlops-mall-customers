"""Tests del cálculo de drift (PSI)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.drift import (
    build_reference_distributions,
    compute_drift,
    population_stability_index,
)


def test_psi_is_zero_for_identical_distributions():
    assert population_stability_index([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)


def test_psi_is_symmetric_and_positive_when_distributions_differ():
    psi = population_stability_index([0.9, 0.1], [0.1, 0.9])
    assert psi > 0.25


def test_reference_distributions_cover_all_features(processed_df: pd.DataFrame):
    reference = build_reference_distributions(processed_df)
    assert set(reference["features"]) == {
        "gender",
        "age",
        "annual_income_k",
        "spending_score",
    }
    for feature in reference["features"].values():
        assert sum(feature["proportions"]) == pytest.approx(1.0, abs=1e-4)


def test_drift_is_stable_against_training_data(processed_df: pd.DataFrame):
    reference = build_reference_distributions(processed_df)
    results = compute_drift(reference, processed_df)
    assert all(values["status"] == "stable" for values in results.values())


def test_drift_detects_shifted_population(processed_df: pd.DataFrame):
    reference = build_reference_distributions(processed_df)
    shifted = processed_df.copy()
    shifted["annual_income_k"] = 190
    results = compute_drift(reference, shifted)
    assert results["annual_income_k"]["status"] == "significant"
