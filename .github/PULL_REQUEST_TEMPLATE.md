## Descripción

<!-- Explica qué cambia y por qué. Si resuelve un issue, usa "Closes #123". -->

Closes #

## Tipo de cambio

- [ ] `feat`: nueva funcionalidad
- [ ] `fix`: corrección de un bug
- [ ] `docs`: cambios en documentación
- [ ] `refactor`: cambio interno sin alterar comportamiento
- [ ] `test`: pruebas nuevas o ajustadas
- [ ] `ci` / `build`: cambios en pipeline o contenedores
- [ ] `chore`: mantenimiento

## Cambios principales

<!-- Lista corta y verificable. -->

-
-
-

## Cómo probarlo

<!-- Comandos exactos para reproducir la validación. -->

```bash
pytest
ruff check src api app tests
```

## Evidencias

<!-- Capturas, salida de tests, métricas de MLflow o queries de Grafana. -->

## Checklist

- [ ] `pytest` pasa en local
- [ ] `ruff check src api app tests` sin errores
- [ ] Actualicé la documentación afectada (`README.md`, `docs/INFORME.md`)
- [ ] No incluí secretos, credenciales ni datos sensibles
- [ ] Si cambié el contrato de la API, actualicé `tests/test_api.py` y el README
- [ ] Si cambié el pipeline, regeneré los artefactos de `reports/`

## Co-autores

<!-- Ejemplo: Co-authored-by: Nombre <correo@uao.edu.co> -->
