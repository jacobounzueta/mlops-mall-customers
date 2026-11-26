"""Smoke test de extremo a extremo contra la API desplegada.

Valida salud, predicción individual, lote, historial y métricas, y guarda un
reporte JSON como evidencia para el informe.

Uso:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT_DIR / "reports" / "metrics" / "smoke_test.json"

CASES = [
    {"gender": "Female", "age": 24, "annual_income_k": 90, "spending_score": 85},
    {"gender": "Male", "age": 55, "annual_income_k": 25, "spending_score": 15},
    {"gender": "Male", "age": 30, "annual_income_k": 50, "spending_score": 50},
]


def check(name: str, response: requests.Response, expected: int = 200) -> dict:
    ok = response.status_code == expected
    print(f"[{'OK ' if ok else 'FAIL'}] {name:<22} HTTP {response.status_code}")
    return {
        "name": name,
        "status_code": response.status_code,
        "ok": ok,
        "detail": response.text[:300] if not ok else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test de la API")
    parser.add_argument(
        "--url",
        default=os.getenv("API_URL", "http://127.0.0.1:8000"),
        help="URL base de la API",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    results: list[dict] = []
    predictions: list[dict] = []

    health = requests.get(f"{base_url}/health", timeout=10)
    results.append(check("GET /health", health))
    if health.status_code == 200:
        payload = health.json()
        print(f"       status={payload['status']} version={payload['model_version']}")

    for index, case in enumerate(CASES, start=1):
        response = requests.post(f"{base_url}/predict", json=case, timeout=10)
        results.append(check(f"POST /predict #{index}", response))
        if response.status_code == 200:
            body = response.json()
            predictions.append(body)
            print(
                f"       segmento={body['customer_segment']} "
                f"etiqueta={body['segment_label']} "
                f"latencia={body['latency_ms']:.2f}ms"
            )

    batch = requests.post(
        f"{base_url}/predict/batch",
        json={"customers": CASES},
        timeout=15,
    )
    results.append(check("POST /predict/batch", batch))

    recent = requests.get(f"{base_url}/predictions/recent?limit=5", timeout=10)
    results.append(check("GET /predictions/recent", recent))

    metrics = requests.get(f"{base_url}/metrics", timeout=10)
    results.append(check("GET /metrics", metrics))
    metrics_ok = metrics.status_code == 200 and (
        "mall_customer_predictions_total" in metrics.text
    )
    results[-1]["ok"] = results[-1]["ok"] and metrics_ok

    invalid = requests.post(
        f"{base_url}/predict",
        json={"gender": "Otro", "age": 10, "annual_income_k": 999, "spending_score": 0},
        timeout=10,
    )
    results.append(check("POST /predict inválido", invalid, expected=422))

    ok = all(result["ok"] for result in results)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "api_url": base_url,
        "ok": ok,
        "checks": results,
        "sample_predictions": predictions,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nResultado: {'OK' if ok else 'FALLÓ'} · evidencia en {REPORT_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
