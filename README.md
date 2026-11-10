# Imputación bayesiana multirrespuesta de rasgos lingüísticos

Este proyecto estima probabilidades para rasgos lingüísticos binarios usando dependencias espaciales y filogenéticas. Evalúa las predicciones con datos reservados y vuelve a ajustar el modelo con todos los valores conocidos para proponer imputaciones.

## Explorar los resultados

Después de ejecutar el análisis, abre `results/current/index.html`. Desde esa página puedes alternar entre las fuentes y abrir el mapa o el explorador de distribuciones, métricas y cobertura.

Cada fuente guarda sus resultados en una carpeta propia:

```text
results/current/
├── index.html
├── wals/
└── gbi_statistical/
```

El explorador muestra probabilidades e intervalos posteriores por lengua, priors, parámetros estimados, diagnósticos de NUTS y métricas del conjunto de test. El mapa separa la evaluación train/test de la imputación final.

## Fuentes y rasgos incluidos

### WALS

El panel WALS reúne tres rasgos binarios:

| Rasgo | Codificación |
|---|---|
| Orden de objeto y verbo, 83A | OV = 0; VO = 1. «Sin dominante» queda fuera del ajuste. |
| Nasal velar, 9A | Presente = 1; ausente = 0. Las dos categorías de presencia se agrupan. |
| Consonantes uvulares, 6A | Ausentes = 0; cualquier categoría con uvulares = 1. |

La presencia de uvulares añade un rasgo fonológico conocido por su distribución geográfica concentrada en varias regiones, útil para contrastar la señal espacial y genealógica con los otros rasgos.

### Grambank, curación estadística

El proyecto integra la release v1.0.0 de GBI: 2.467 lenguas y 196 columnas de rasgos. La auditoría identifica 187 columnas con estados estrictamente 0/1. El panel inicial modela cuatro rasgos binarios:

- GB035: tres o más contrastes de distancia en demostrativos;
- GB480m: presencia de caso morfológico;
- GB132: orden verbo-medial en cláusulas transitivas;
- GB408: presencia de alineamiento acusativo en la marcación.

GB408 permite contrastar la asociación entre alineamiento acusativo y caso morfológico (GB480m). Entre las lenguas elegibles con ambos valores observados, su correlación binaria descriptiva es aproximadamente 0,35 (1.518 lenguas). Es una asociación observada antes de separar las contribuciones geográfica y filogenética; no es una estimación causal ni el resultado posterior del modelo.

Los 187 rasgos binarios están disponibles en la matriz curada y quedan listados en `data/processed/gbi_statistical/feature_catalog.csv`. El ajuste NUTS usa el panel de cuatro rasgos indicado en `config/model_gbi.json`; no se presenta como un ajuste simultáneo de las 187 columnas. La curación estadística reduce algunas dependencias fuertes adicionales a las explicables por genealogía y área, pero no garantiza independencia absoluta.

## Preparar y ejecutar

Se necesita Python 3.12 o posterior y [`uv`](https://docs.astral.sh/uv/).

```powershell
uv sync --locked

# Revisar la cobertura de cada fuente
uv run lingfeat prepare --model-config config/model.json
uv run lingfeat audit --dataset wals
uv run lingfeat prepare --model-config config/model_gbi.json
uv run lingfeat audit --dataset gbi_statistical

# Ejecutar evaluación e imputación para las dos fuentes
uv run lingfeat clean-results
uv run lingfeat run --all-datasets

# Abrir el portal con el selector de fuente
uv run lingfeat site
```

Para repetir solo una corrida:

```powershell
uv run lingfeat run --model-config config/model.json
uv run lingfeat run --model-config config/model_gbi.json
```

También puedes abrir cada salida por separado con `uv run lingfeat map --dataset gbi_statistical` o `uv run lingfeat explore --dataset wals`.

## Qué ocurre en cada corrida

1. El adaptador de la fuente carga las lenguas, sus rasgos y la metadata.
2. Se alinean los glottocódigos con coordenadas y con las hojas del árbol MCC. Los casos sin coordenadas o fuera del árbol quedan auditados y no se ajustan.
3. Para cada rasgo, un 20 % de los valores observados se reserva con un split estratificado y agrupado por glottocódigo.
4. El modelo multirrespuesta se ajusta con train y predice test. Las métricas se calculan únicamente sobre esas observaciones reservadas.
5. Se vuelve a ajustar con todos los valores observados y se predicen las celdas faltantes elegibles.
6. Se guardan draws posteriores, predicciones, auditorías, reportes y visualizaciones.

El predictor de cada lengua y rasgo suma una tendencia global, un efecto espacial y uno filogenético:

```text
ηᵢₖ = αₖ + gᵢₖ + hᵢₖ
P(yᵢₖ = 1) = Φ(ηᵢₖ)
yᵢₖ ~ Bernoulli(P(yᵢₖ = 1))
```

Las amplitudes y las correlaciones entre rasgos se estiman por separado para cada componente. Las matrices de correlación espacial y filogenética entre lenguas proceden, respectivamente, de armónicos esféricos y de tiempos de ramas compartidas del MCC. La formulación matemática completa, incluidas las covarianzas entre lenguas y rasgos y el producto de Kronecker, está en [Modelo multirrespuesta y matrices de covarianza](docs/model_and_covariance.md).

## Ajustes y extensiones

- [`config/model.json`](config/model.json): panel WALS, codificaciones, split y semilla de evaluación.
- [`config/model_gbi.json`](config/model_gbi.json): panel GBI y columnas fuente.
- [`config/inference.json`](config/inference.json): priors y configuración de NUTS.
- [`docs/workflow.md`](docs/workflow.md): pasos desde la fuente hasta el mapa y el reporte.
- [`docs/data_sources.md`](docs/data_sources.md): alcance, selección y procedencia de WALS y GBI.

Para añadir un rasgo, agrega su definición a la configuración correspondiente. En GBI, `source_column` debe señalar una columna cuyo conjunto de valores observados sea exactamente `{0, 1}`. El cargador valida esta condición antes del ajuste. Añadir rasgos aumenta el tamaño del modelo y el tiempo de NUTS.

## Inferencia y alcance

La configuración actual usa dos cadenas, 500 pasos de calentamiento, 200 draws por cadena, aceptación objetivo 0,85 y profundidad máxima 7. Esto deja hasta 400 draws posteriores para las predicciones. Revisa siempre los diagnósticos antes de interpretar posteriores o métricas; si aparecen avisos de mezcla, el resultado es exploratorio y habrá que ajustar el muestreo.

El split es aleatorio, estratificado por clase y agrupado por glottocódigo; no sustituye una validación que deje familias o regiones completas fuera. El análisis usa el árbol MCC, no integra la incertidumbre entre árboles. Los paneles son selecciones piloto configurables, no un análisis exhaustivo de todos los rasgos de cada base.

## Fuentes y citas

- WALS CLDF, [dataset y código](https://github.com/cldf-datasets/wals).
- Grambank Independent (GBI), Graff et al. (2025), [artículo](https://doi.org/10.1038/s41597-024-04319-4), [release y scripts](https://github.com/annagraff/crossling-curated), [datos v1.0.0](https://doi.org/10.5281/zenodo.14606326). Datos bajo licencia CC BY 4.0.
- Global Language Tree Pipeline, [release v2.0.0](https://github.com/rbouckaert/global-language-tree-pipeline/releases/tag/v2.0.0).

