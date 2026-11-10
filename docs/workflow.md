# Flujo de trabajo

## 1. Seleccionar una fuente

El análisis usa adaptadores para que fuentes distintas entreguen una misma tabla: una fila por lengua, coordenadas, glottocódigo, metadata y una columna binaria por rasgo.

- `data_sources/wals.py` lee WALS CLDF y aplica las codificaciones declaradas en `config/model.json`.
- `data_sources/gbi.py` lee Grambank Independent, la curación estadística, y sus metadata CLDF. El panel que se ajusta está en `config/model_gbi.json`.
- `data_sources/base.py` define el contrato que deben cumplir otras fuentes.
- `data.py` armoniza las tablas, genera auditorías y las alinea con el árbol.

La curación estadística de GBI tiene 2.467 lenguas, 196 columnas y 187 rasgos con valores observados 0/1. El programa guarda el catálogo completo de rasgos binarios. La corrida piloto usa GB035, GB480m, GB132 y GB408; GB408 permite estudiar su asociación con GB480m.

## 2. Preparar las lenguas y covarianzas

Las fuentes usan glottocódigos para conectar cada lengua con la geografía y el MCC. Para formar parte del ajuste, una fila requiere coordenadas válidas y presencia en el árbol. Las filas restantes quedan en `data/processed/<conjunto>/languages.csv` con un motivo de exclusión.

El proveedor `MCCProvider` calcula la covarianza Browniana a partir del tiempo de ramas compartidas. La matriz, el orden de sus glottocódigos y su auditoría se guardan dentro de la carpeta de cada conjunto. Así, WALS y GBI nunca usan por accidente una matriz con otro orden de hojas.

Los armónicos esféricos de `design.py` generan la estructura espacial a partir de las coordenadas. El grado máximo `L` y el suavizado `λ` se infieren durante el ajuste.

## 3. Evaluar sin entrenar con el test

`pipeline.py` obtiene las respuestas en una matriz lengua × rasgo. Para cada rasgo, `StratifiedGroupKFold` reserva una de cinco particiones: aproximadamente 20 %, manteniendo la proporción de 0 y 1 y agrupando filas por glottocódigo.

El split se hace por rasgo. Por eso una lengua puede estar en test para una respuesta y aportar otra respuesta observada al ajuste conjunto. La métrica de cada rasgo se calcula solo con sus celdas de test. Este diseño mide predicción de celdas faltantes con otra información de la misma lengua disponible; no mide el caso más estricto de ocultar todas las respuestas de lenguas completas.

El ajuste de evaluación usa únicamente las respuestas marcadas como train. Sus probabilidades de test se comparan con las observaciones reservadas para calcular exactitud, exactitud balanceada, precisión, recall, F1, MCC, AUC, Brier y log-loss.

## 4. Ajustar de nuevo e imputar

Después de evaluar, el pipeline descarta ese ajuste y ejecuta NUTS otra vez con todas las respuestas observadas. Este segundo ajuste produce la distribución posterior de cada celda faltante elegible. Los parámetros de evaluación no se reutilizan; el nuevo ajuste vuelve a estimarlos usando más datos.

Para cada lengua, rasgo y draw posterior se obtiene una probabilidad de la clase positiva. La media resume la predicción y los percentiles 2,5 y 97,5 forman el intervalo creíble puntual del 95 %. Los draws conservan la incertidumbre sobre los parámetros y los efectos latentes.

## 5. Ver el resultado

Las salidas se organizan por fuente dentro de `results/current/`:

```text
results/current/
├── index.html                 selector WALS / Grambank
├── wals/
│   ├── map.html
│   ├── posterior.html
│   ├── report.md
│   └── predictions.csv
└── gbi_statistical/
    ├── map.html
    ├── posterior.html
    ├── report.md
    └── predictions.csv
```

Cada carpeta de fuente también contiene `run.json`, `evaluation.npz` e `imputation.npz`. El portal sirve para cambiar de fuente y de tipo de vista. El mapa muestra capas por rasgo para evaluación e imputación. El explorador presenta distribuciones por lengua, priors, posteriores, métricas, cobertura y diagnósticos.

## 6. Comandos

```powershell
uv sync
uv run lingfeat prepare --model-config config/model.json
uv run lingfeat prepare --model-config config/model_gbi.json
uv run lingfeat audit --dataset wals
uv run lingfeat audit --dataset gbi_statistical
uv run lingfeat clean-results
uv run lingfeat run --all-datasets
uv run lingfeat site
```

`run --all-datasets` prepara y ejecuta ambos paneles. Para ejecutar uno solo, pasa `--model-config config/model.json` o `--model-config config/model_gbi.json`. `map` y `explore` aceptan `--dataset wals` o `--dataset gbi_statistical`.

NUTS está configurado en `config/inference.json`: dos cadenas, 500 pasos de calentamiento, 200 draws por cadena, aceptación objetivo 0,85 y profundidad máxima 7. El split usa la semilla en el archivo del modelo; el muestreo usa la semilla de inferencia.
