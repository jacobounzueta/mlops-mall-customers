"""Genera tráfico de demostración contra la API.

Toma clientes del dataset real y los envía por lotes, para poblar el dashboard
de Grafana y habilitar el cálculo de PSI (que requiere >= 30 predicciones).

Uso:
    python scripts/generate_traffic.py --n 120
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "mall_customers_processed.csv"
RAW_PATH = ROOT_DIR / "data" / "raw" / "Mall_Customers.csv"


def load_customers() -> pd.DataFrame:
    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH)
    if not RAW_PATH.exists():
        raise FileNotFoundError("No hay dataset. Ejecuta primero: python -m src.data")

    df = pd.read_csv(RAW_PATH)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("$", "", regex=False)
    )
    df = df.rename(columns={"spending_score_1_100": "spending_score"})
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    return df[["gender", "age", "annual_income_k", "spending_score"]]


def to_payload(row: pd.Series) -> dict:
    return {
        "gender": "Male" if int(row["gender"]) == 1 else "Female",
        "age": int(row["age"]),
        "annual_income_k": int(row["annual_income_k"]),
        "spending_score": int(row["spending_score"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tráfico de demostración")
    parser.add_argument("--n", type=int, default=120, help="Número de clientes")
    parser.add_argument(
        "--url",
        default=os.getenv("API_URL", "http://127.0.0.1:8000"),
        help="URL base de la API",
    )
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    customers = load_customers()
    sample = customers.sample(
        n=args.n, replace=args.n > len(customers), random_state=42
    )

    labels: Counter = Counter()
    sent = 0
    for start in range(0, len(sample), args.batch_size):
        chunk = sample.iloc[start : start + args.batch_size]
        payload = {"customers": [to_payload(row) for _, row in chunk.iterrows()]}
        response = requests.post(f"{base_url}/predict/batch", json=payload, timeout=30)
        response.raise_for_status()
        body = response.json()
        sent += body["count"]
        labels.update(prediction["segment_label"] for prediction in body["predictions"])
        print(f"Enviados {sent}/{len(sample)}")

    print("\nDistribución de segmentos predichos:")
    for label, count in labels.most_common():
        print(f"  {label:<16} {count}")

    metrics = requests.get(f"{base_url}/metrics", timeout=10).text
    samples = [
        line
        for line in metrics.splitlines()
        if line.startswith("mall_customer_drift_samples")
    ]
    print("\n" + (samples[0] if samples else "sin métrica de muestras"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
