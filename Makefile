# Makefile — atajos del laboratorio MLOps
# En Windows requiere GNU Make (viene con Git Bash / Chocolatey).

PY := .venv/Scripts/python.exe
PIP := .venv/Scripts/pip.exe
COMPOSE := docker compose

export PYTHONUTF8 := 1

.PHONY: help setup data cluster mlflow-up train register predict api app up down logs test lint format drift smoke traffic clean

help:
	@echo "Targets disponibles:"
	@echo "  setup      Crear .venv e instalar dependencias"
	@echo "  data       Descargar, limpiar y validar el dataset"
	@echo "  cluster    Construir segmentos con KMeans"
	@echo "  mlflow-up  Levantar solo MLflow (Docker)"
	@echo "  train      Entrenar los 3 modelos y registrar en MLflow"
	@echo "  register   Registrar el mejor modelo y asignar alias champion"
	@echo "  predict    Predicción local con el modelo champion"
	@echo "  api        Servir la API en local (uvicorn)"
	@echo "  app        Servir Streamlit en local"
	@echo "  up         Levantar el stack completo (Docker Compose)"
	@echo "  down       Apagar el stack"
	@echo "  logs       Ver logs del stack"
	@echo "  test       Ejecutar pytest"
	@echo "  lint       Ejecutar ruff"
	@echo "  drift      Calcular PSI con las predicciones registradas"
	@echo "  smoke      Probar la API desplegada de punta a punta"
	@echo "  traffic    Generar tráfico de demostración contra la API"
	@echo "  clean      Borrar artefactos generados"

setup:
	uv venv .venv --python 3.12
	uv pip install --python $(PY) -r requirements.txt

data:
	$(PY) -m src.data

cluster: data
	$(PY) -m src.clustering

mlflow-up:
	$(COMPOSE) up -d mlflow

train:
	$(PY) -m src.train

register:
	$(PY) -m src.register_model

predict:
	$(PY) -m src.predict

api:
	$(PY) -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

app:
	$(PY) -m streamlit run app/streamlit_app.py

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src api app tests scripts

format:
	$(PY) -m ruff format src api app tests scripts

drift:
	$(PY) -m src.drift

smoke:
	$(PY) scripts/smoke_test.py

traffic:
	$(PY) scripts/generate_traffic.py --n 120

clean:
	rm -rf reports/figures/*.png reports/metrics/*.csv reports/metrics/*.json
	rm -rf data/processed/*.csv data/monitoring/*.csv
