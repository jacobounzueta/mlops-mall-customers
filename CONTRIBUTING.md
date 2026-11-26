# Contribuir

Gracias por querer mejorar este laboratorio. El flujo está pensado para que
cualquier cambio llegue a `main` con tests, lint y documentación al día.

## 1. Preparar el entorno

```bash
git clone https://github.com/jacobounzueta/mlops-mall-customers.git
cd mlops-mall-customers
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

En Linux/macOS usa `.venv/bin/python`.

## 2. Levantar MLflow

```bash
docker compose up -d mlflow
```

## 3. Flujo de trabajo

1. Crea una rama descriptiva: `feat/drift-alertas`, `fix/api-timeout`.
2. Haz commits pequeños con [Conventional Commits](https://www.conventionalcommits.org/):
   `feat: agrega endpoint de scoring por lote`.
3. Verifica antes de abrir el PR:

```bash
ruff check src api app tests
pytest
```

4. Abre el PR usando la plantilla. Describe el cambio, cómo probarlo y adjunta
   evidencias (salida de tests, captura de Grafana, métricas de MLflow).
5. Se requiere al menos una revisión y el CI en verde.

## 4. Reglas de calidad

- **Reproducibilidad**: fija versiones en `requirements.txt`; no uses `latest`
  en imágenes de Docker.
- **Sin secretos**: nunca subas tokens, `.env` ni credenciales.
- **Contrato estable**: si cambias el request/response de la API, actualiza
  `tests/test_api.py`, el README y la versión de la API.
- **Datos**: `data/raw/Mall_Customers.csv` se versiona a propósito para que el
  CI pueda ejecutar el pipeline; los artefactos derivados no.
- **Documentación**: todo cambio funcional debe reflejarse en `README.md` y,
  si afecta la metodología, en `docs/INFORME.md`.

## 5. Co-autoría

Si trabajas en pareja, agrega el trailer al commit:

```text
Co-authored-by: Rommel Prieto <rommel.prieto@uao.edu.co>
```

## 6. Reportar problemas

Usa los templates de issue. Incluye pasos exactos, logs y entorno; sin eso es
muy difícil reproducir el fallo.
