"""Prueba local del modelo ``champion`` registrado en MLflow.

Usa el flavor pyfunc, que es el artefacto portable: cualquier consumidor
puede cargarlo sin conocer el estimador de scikit-learn subyacente.

Uso:
    python -m src.predict
    python -m src.predict --gender Female --age 24 --income 90 --spending 85
"""

from __future__ import annotations

import argparse
import json

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.config import (
    METRICS_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
)
from src.features import FEATURE_COLUMNS


def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    return mlflow.pyfunc.load_model(model_uri)


def load_segment_labels() -> dict:
    path = METRICS_DIR / "segment_labels.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["segments"]


def predict_customer(
    gender: str = "Male",
    age: int = 35,
    annual_income_k: int = 120,
    spending_score: int = 50,
) -> dict:
    model = load_model()

    gender_value = 1 if gender.strip().lower() == "male" else 0
    input_df = pd.DataFrame(
        [
            {
                "gender": gender_value,
                "age": age,
                "annual_income_k": annual_income_k,
                "spending_score": spending_score,
            }
        ]
    )[FEATURE_COLUMNS]

    prediction = int(model.predict(input_df)[0])
    labels = load_segment_labels()
    label = labels.get(str(prediction), {}).get("label", "desconocido")

    return {
        "customer_segment": prediction,
        "segment_label": label,
        "features": input_df.to_dict(orient="records")[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicción local con MLflow")
    parser.add_argument("--gender", default="Male", choices=["Male", "Female"])
    parser.add_argument("--age", type=int, default=35)
    parser.add_argument("--income", type=int, default=120)
    parser.add_argument("--spending", type=int, default=50)
    args = parser.parse_args()

    result = predict_customer(
        gender=args.gender,
        age=args.age,
        annual_income_k=args.income,
        spending_score=args.spending,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
