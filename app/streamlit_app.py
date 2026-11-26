"""Frontend Streamlit para probar el modelo de segmentación.

Consume la API de FastAPI (no carga el modelo directamente), de modo que la
UI y el serving quedan desacoplados.

Uso:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

SEGMENT_COLORS = {
    0: "#264653",
    1: "#2a9d8f",
    2: "#e9c46a",
    3: "#e76f51",
    4: "#8ecae6",
}

st.set_page_config(
    page_title="Clasificador de Segmentos de Clientes",
    page_icon="🛍️",
    layout="wide",
)


@st.cache_data(ttl=10, show_spinner=False)
def api_get(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=5)
    response.raise_for_status()
    return response.json()


def api_health() -> dict | None:
    try:
        return api_get("/health")
    except Exception:
        return None


def api_model_info() -> dict | None:
    try:
        return api_get("/model/info")
    except Exception:
        return None


def api_recent(limit: int = 15):
    try:
        return api_get(f"/predictions/recent?limit={limit}")["predictions"]
    except Exception:
        return []


st.title("🛍️ Clasificador de Segmentos de Clientes")
st.caption(
    "Mall Customers · KMeans + clasificador supervisado · MLflow + FastAPI + "
    "Prometheus + Grafana"
)

health = api_health()
model_info = api_model_info()

with st.sidebar:
    st.header("Estado del servicio")
    if health is None:
        st.error("No hay conexión con la API")
        st.code(f"API_URL = {API_URL}")
        st.stop()
    elif health["status"] != "ok":
        st.warning("API arriba, pero sin modelo cargado")
        if health.get("error"):
            st.code(health["error"])
    else:
        st.success("API y modelo operativos")

    st.metric("Modelo", health["model"])
    st.metric("Alias", health["alias"])
    st.metric("Versión", health["model_version"] or "n/d")
    st.metric("Uptime", f"{health['uptime_seconds'] / 60:.1f} min")

    if model_info:
        st.divider()
        st.subheader("Segmentos")
        for segment, info in sorted(
            model_info.get("segments", {}).items(), key=lambda item: int(item[0])
        ):
            st.markdown(f"**{segment} · {info['label']}**  \\n{info['description']}")

col_form, col_result = st.columns([1, 1.3], gap="large")

with col_form:
    st.subheader("Datos del cliente")
    with st.form("customer_form"):
        gender = st.selectbox("Género", ["Male", "Female"])
        age = st.slider("Edad", min_value=18, max_value=100, value=30)
        annual_income_k = st.slider(
            "Ingreso anual (miles de USD)", min_value=0, max_value=200, value=60
        )
        spending_score = st.slider(
            "Spending score (1-100)", min_value=1, max_value=100, value=50
        )
        submitted = st.form_submit_button("Predecir segmento", use_container_width=True)

if submitted:
    payload = {
        "gender": gender,
        "age": age,
        "annual_income_k": annual_income_k,
        "spending_score": spending_score,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 422:
            st.error("Datos inválidos: revisa los rangos del formulario.")
        elif response.status_code == 503:
            st.error("El modelo todavía no está cargado en la API.")
        else:
            response.raise_for_status()
            st.session_state["last_prediction"] = response.json()
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con la API. ¿Está corriendo en el puerto 8000?")
    except requests.exceptions.HTTPError as exc:
        st.error(f"La API respondió con error: {exc}")

with col_result:
    st.subheader("Resultado")

    result = st.session_state.get("last_prediction")
    if result:
        label = result["segment_label"]
        st.markdown(
            f"<div style='padding:1.2rem;border-radius:0.8rem;"
            f"background:{SEGMENT_COLORS.get(result['customer_segment'], '#264653')};"
            f"color:white;text-align:center'>"
            f"<div style='font-size:0.9rem;opacity:0.8'>Segmento "
            f"{result['customer_segment']}</div>"
            f"<div style='font-size:2rem;font-weight:700'>{label}</div></div>",
            unsafe_allow_html=True,
        )
        st.write(result.get("segment_description", ""))

        if result.get("probabilities"):
            proba_df = (
                pd.DataFrame(
                    {
                        "segmento": list(result["probabilities"].keys()),
                        "probabilidad": list(result["probabilities"].values()),
                    }
                )
                .sort_values("probabilidad", ascending=False)
                .set_index("segmento")
            )
            st.bar_chart(proba_df, height=220)

        c1, c2, c3 = st.columns(3)
        c1.metric("Latencia", f"{result['latency_ms']:.1f} ms")
        c2.metric("Versión", result.get("model_version") or "n/d")
        c3.metric(
            "Confianza",
            f"{max(result.get('probabilities', {0: 0}).values()) * 100:.1f}%",
        )

        if model_info:
            means = model_info.get("global_means", {})
            comparison = pd.DataFrame(
                {
                    "cliente": {
                        "age": age,
                        "annual_income_k": annual_income_k,
                        "spending_score": spending_score,
                    },
                    "promedio_dataset": {
                        "age": means.get("age"),
                        "annual_income_k": means.get("annual_income_k"),
                        "spending_score": means.get("spending_score"),
                    },
                }
            )
            st.caption("Comparación con el promedio del dataset de entrenamiento")
            st.dataframe(comparison, use_container_width=True)
    else:
        st.info(
            "Completa el formulario y pulsa **Predecir segmento**. "
            "El resultado también queda registrado para el análisis de drift."
        )

st.divider()
st.subheader("Últimas predicciones registradas")
recent = api_recent()
if recent:
    st.dataframe(pd.DataFrame(recent), use_container_width=True, height=260)
else:
    st.caption("Todavía no hay predicciones registradas.")
