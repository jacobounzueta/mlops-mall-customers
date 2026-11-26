"""Tests de la API con un modelo simulado (no requiere MLflow)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as main


class DummyModel:
    """Modelo mínimo con la misma interfaz que un Pipeline de sklearn."""

    classes_ = np.array([0, 1, 2, 3])

    def predict(self, X):  # noqa: N803 - firma de sklearn
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):  # noqa: N803 - firma de sklearn
        return np.tile(np.array([0.7, 0.1, 0.1, 0.1]), (len(X), 1))


@pytest.fixture()
def client():
    main.state.model = DummyModel()
    main.state.version = "test-version"
    main.state.error = None
    with TestClient(main.app) as test_client:
        yield test_client
    main.state.model = None
    main.state.version = None


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "Mall Customers" in response.json()["message"]


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_degraded_without_model():
    main.state.model = None
    with TestClient(main.app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_predict_returns_segment_and_probabilities(client: TestClient):
    response = client.post(
        "/predict",
        json={
            "gender": "Female",
            "age": 24,
            "annual_income_k": 90,
            "spending_score": 85,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["customer_segment"] == 0
    assert 0 <= body["latency_ms"] < 5000
    assert set(body["probabilities"]) == {"0", "1", "2", "3"}
    assert body["probabilities"]["0"] == pytest.approx(0.7)


def test_predict_rejects_invalid_gender(client: TestClient):
    response = client.post(
        "/predict",
        json={
            "gender": "Otro",
            "age": 24,
            "annual_income_k": 90,
            "spending_score": 85,
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Datos de entrada inválidos"


def test_predict_rejects_out_of_range_age(client: TestClient):
    response = client.post(
        "/predict",
        json={
            "gender": "Male",
            "age": 5,
            "annual_income_k": 90,
            "spending_score": 85,
        },
    )
    assert response.status_code == 422


def test_predict_returns_503_without_model():
    main.state.model = None
    with TestClient(main.app) as test_client:
        response = test_client.post(
            "/predict",
            json={
                "gender": "Male",
                "age": 30,
                "annual_income_k": 50,
                "spending_score": 50,
            },
        )
    assert response.status_code == 503


def test_batch_prediction(client: TestClient):
    payload = {
        "customers": [
            {
                "gender": "Male",
                "age": 30,
                "annual_income_k": 50,
                "spending_score": 50,
            },
            {
                "gender": "Female",
                "age": 40,
                "annual_income_k": 90,
                "spending_score": 80,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_recent_predictions(client: TestClient):
    client.post(
        "/predict",
        json={
            "gender": "Male",
            "age": 33,
            "annual_income_k": 70,
            "spending_score": 60,
        },
    )
    response = client.get("/predictions/recent?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] >= 1
    assert "annual_income_k" in body["predictions"][0]


def test_metrics_exposes_prometheus_counters(client: TestClient):
    client.post(
        "/predict",
        json={
            "gender": "Male",
            "age": 33,
            "annual_income_k": 70,
            "spending_score": 60,
        },
    )
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "mall_customer_predictions_total" in response.text
    assert "mall_customer_prediction_latency_seconds_bucket" in response.text
    assert "mall_customer_model_loaded 1.0" in response.text
