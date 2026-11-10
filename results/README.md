# Resultados

Una corrida guarda resultados separados por fuente en `current/`. Abre `current/index.html` para alternar entre WALS y Grambank, y entre el mapa y el explorador posterior.

Cada carpeta de conjunto (`wals/` o `gbi_statistical/`) contiene:

- `map.html`: capas globales de evaluación y de imputación, una por rasgo.
- `posterior.html`: probabilidades por lengua, priors, posteriores, métricas y cobertura.
- `report.md`: resumen de métricas, covariaciones y diagnósticos.
- `predictions.csv`: estados observados, predicciones e intervalos por lengua y rasgo.
- `run.json`: fuente, configuración, cobertura y diagnósticos completos.
- `evaluation.npz` e `imputation.npz`: draws posteriores, probabilidades, máscaras y extras del muestreador.

Las tablas de preparación y auditoría se guardan en `data/processed/<conjunto>/`. La fuente descargada está en `data/source/`. Los archivos `.npz` respaldan las visualizaciones: no se regeneran desde el HTML y se conservan para que el reporte pueda reconstruirse.
