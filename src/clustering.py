"""Segmentación de clientes con KMeans.

El dataset no tiene etiqueta objetivo, así que se construyen segmentos con
KMeans y luego se nombran a partir de los centroides (ingreso y gasto),
evitando etiquetas hardcodeadas que no correspondan con los datos reales.

Uso:
    python -m src.clustering
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config import (
    CLUSTERING_FEATURES,
    CLUSTERING_METRICS_PATH,
    FIGURES_DIR,
    K_RANGE,
    METRICS_DIR,
    N_CLUSTERS,
    RANDOM_STATE,
    SEGMENT_LABELS_PATH,
    SEGMENTED_DATA_PATH,
)
from src.data import preprocess_data
from src.drift import save_reference_distributions
from src.features import FEATURE_COLUMNS, scale_features

sns.set_theme(style="whitegrid", palette="Set2")


def evaluate_k(X_scaled) -> tuple[list[int], list[float], list[float]]:
    cluster_range = list(K_RANGE)
    inertias: list[float] = []
    silhouette_scores: list[float] = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(float(kmeans.inertia_))
        silhouette_scores.append(float(silhouette_score(X_scaled, labels)))

    return cluster_range, inertias, silhouette_scores


def plot_elbow(cluster_range, inertias) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, inertias, marker="o", linewidth=2)
    plt.axvline(N_CLUSTERS, color="crimson", linestyle="--", label=f"k={N_CLUSTERS}")
    plt.title("Método del codo para seleccionar el número de clusters")
    plt.xlabel("Número de clusters")
    plt.ylabel("Inercia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "elbow_method.png", dpi=140)
    plt.close()


def plot_silhouette(cluster_range, silhouette_scores) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, silhouette_scores, marker="o", linewidth=2, color="#2a9d8f")
    plt.axvline(N_CLUSTERS, color="crimson", linestyle="--", label=f"k={N_CLUSTERS}")
    plt.title("Silhouette score por número de clusters")
    plt.xlabel("Número de clusters")
    plt.ylabel("Silhouette score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "silhouette_scores.png", dpi=140)
    plt.close()


def build_segment_labels(profiles: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Nombra los segmentos comparando sus centroides con la media global.

    Un segmento es "alto" si su centroide supera media + 0.5 desviaciones
    estándar, "bajo" si está por debajo de media - 0.5 y "medio" en el resto.
    Así el nombre siempre describe el comportamiento real del cluster.
    """
    income_mean = float(df["annual_income_k"].mean())
    income_std = float(df["annual_income_k"].std(ddof=0)) or 1.0
    spending_mean = float(df["spending_score"].mean())
    spending_std = float(df["spending_score"].std(ddof=0)) or 1.0
    age_median = float(df["age"].median())

    def level(value: float, mean: float, std: float) -> str:
        z_score = (value - mean) / std
        if z_score >= 0.5:
            return "high"
        if z_score <= -0.5:
            return "low"
        return "mid"

    names = {
        ("high", "high"): "Premium",
        ("high", "low"): "Conservador",
        ("high", "mid"): "Alto potencial",
        ("mid", "high"): "Impulsivo",
        ("mid", "low"): "Ahorrador",
        ("mid", "mid"): "Estándar",
        ("low", "high"): "Impulsivo",
        ("low", "low"): "Bajo valor",
        ("low", "mid"): "Gasto moderado",
    }
    descriptions = {
        "Premium": "Ingreso alto y gasto alto. El segmento de mayor valor.",
        "Conservador": "Ingreso alto y gasto bajo. Potencial sin explotar.",
        "Alto potencial": "Ingreso alto con gasto medio. Candidato a fidelizar.",
        "Impulsivo": "Gasto alto con ingreso bajo o medio. Compra por impulso.",
        "Ahorrador": "Ingreso medio y gasto bajo. Sensible al precio.",
        "Estándar": "Ingreso y gasto cercanos a la media del mall.",
        "Bajo valor": "Ingreso y gasto bajos. Menor prioridad comercial.",
        "Gasto moderado": "Ingreso bajo con gasto medio. Consumo contenido.",
    }

    labels: dict = {}
    for _, row in profiles.iterrows():
        income_level = level(row["annual_income_k"], income_mean, income_std)
        spending_level = level(row["spending_score"], spending_mean, spending_std)
        name = names[(income_level, spending_level)]

        # Desambiguación: si dos clusters comparten arquetipo, se diferencia
        # por edad para no repetir nombres en la API ni en Grafana.
        if name in {value["label"] for value in labels.values()}:
            suffix = "joven" if row["age"] <= age_median else "senior"
            base_name = name
            name = f"{name} {suffix}"
            descriptions.setdefault(
                name,
                f"{descriptions[base_name]} Diferenciado por edad ({suffix}).",
            )

        labels[int(row["customer_segment"])] = {
            "label": name,
            "description": descriptions[name],
            "income_level": income_level,
            "spending_level": spending_level,
        }

    return labels


def create_customer_segments() -> pd.DataFrame:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = preprocess_data()
    X_cluster = df[CLUSTERING_FEATURES]
    X_scaled, _ = scale_features(X_cluster)

    cluster_range, inertias, silhouette_scores = evaluate_k(X_scaled)
    plot_elbow(cluster_range, inertias)
    plot_silhouette(cluster_range, silhouette_scores)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    df["customer_segment"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_scaled)
    df["pca_1"] = components[:, 0]
    df["pca_2"] = components[:, 1]

    profiles = (
        df.groupby("customer_segment")
        .agg(
            size=("customer_segment", "size"),
            age=("age", "mean"),
            annual_income_k=("annual_income_k", "mean"),
            spending_score=("spending_score", "mean"),
            gender_ratio_male=("gender", "mean"),
        )
        .round(3)
        .reset_index()
    )

    segment_labels = build_segment_labels(profiles, df)
    profiles["segment_label"] = profiles["customer_segment"].map(
        lambda segment: segment_labels[int(segment)]["label"]
    )

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="pca_1",
        y="pca_2",
        hue="customer_segment",
        palette="Set2",
        s=60,
        alpha=0.85,
    )
    for _segment, row in profiles.iterrows():
        subset = df[df["customer_segment"] == row["customer_segment"]]
        plt.annotate(
            segment_labels[int(row["customer_segment"])]["label"],
            (subset["pca_1"].mean(), subset["pca_2"].mean()),
            fontsize=9,
            fontweight="bold",
            ha="center",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.8},
        )
    plt.title("Segmentos de clientes visualizados con PCA")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "customer_segments_pca.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    heatmap_data = profiles.set_index("segment_label")[
        ["age", "annual_income_k", "spending_score"]
    ]
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Perfil promedio por segmento")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "segment_profiles.png", dpi=140)
    plt.close()

    profiles.to_csv(METRICS_DIR / "segment_summary.csv", index=False)

    global_means = {
        column: round(float(df[column].mean()), 3) for column in FEATURE_COLUMNS
    }
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_clusters": N_CLUSTERS,
        "global_means": global_means,
        "feature_columns": FEATURE_COLUMNS,
        "segments": {
            str(segment): {
                **values,
                "size": int(
                    profiles.loc[profiles["customer_segment"] == segment, "size"].iloc[
                        0
                    ]
                ),
            }
            for segment, values in segment_labels.items()
        },
    }
    SEGMENT_LABELS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    silhouette_chosen = silhouette_scores[cluster_range.index(N_CLUSTERS)]
    CLUSTERING_METRICS_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "n_rows": int(len(df)),
                "n_clusters": N_CLUSTERS,
                "silhouette_by_k": dict(
                    zip(cluster_range, silhouette_scores, strict=True)
                ),
                "inertia_by_k": dict(zip(cluster_range, inertias, strict=True)),
                "silhouette_chosen_k": round(float(silhouette_chosen), 6),
                "pca_explained_variance": [
                    round(float(value), 6) for value in pca.explained_variance_ratio_
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    save_reference_distributions(df)

    SEGMENTED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SEGMENTED_DATA_PATH, index=False)

    print("Segmentos creados correctamente.")
    print(f"Dataset segmentado: {SEGMENTED_DATA_PATH}")
    print(f"Figuras: {FIGURES_DIR}")
    print(f"Etiquetas: {SEGMENT_LABELS_PATH}")
    print(f"Silhouette (k={N_CLUSTERS}): {silhouette_chosen:.4f}\n")
    print(profiles.to_string(index=False))

    return df


if __name__ == "__main__":
    create_customer_segments()
