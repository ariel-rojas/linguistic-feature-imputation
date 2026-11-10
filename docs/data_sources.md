# Fuentes de rasgos y cobertura

El mismo modelo puede recibir tablas de fuentes distintas. Cada adaptador transforma su tabla original a filas de lenguas y columnas binarias 0/1; el resto del pipeline usa esa estructura común.

## WALS

El adaptador `data_sources/wals.py` descarga la revisión fijada de WALS CLDF. Toma `languages.csv`, `values.csv` y `codes.csv`, elimina registros taxonómicos de géneros y familias, y verifica que cada código configurado exista. Después recodifica únicamente los valores indicados en `config/model.json`.

En 83A, `83A-1` se codifica 0 y `83A-2` se codifica 1; `83A-3` («Sin dominante») queda fuera del ajuste binario. En 9A, las dos categorías que registran presencia de nasal velar se agrupan como 1. En 6A, «No uvulares» es 0 y las tres categorías con uvulares se agrupan como 1.

La integración con LOW compara los códigos ISO de WALS con los de Languages of the World. Los conflictos quedan excluidos y contabilizados en la auditoría.

## Grambank Independent (GBI)

El adaptador `data_sources/gbi.py` carga la tabla `statisticalGBI.csv` y sus ficheros CLDF de metadata. Los glottocódigos sirven como identificadores para unir la tabla de valores con nombres, coordenadas y definiciones de rasgos.

La versión estadística contiene 2.467 lenguas y 196 columnas de rasgos. El código convierte cada columna a números y detecta cuáles tienen exactamente dos estados observados: 0 y 1. En esta release, 187 columnas cumplen ese criterio. Los valores `?`, celdas vacías y otras entradas no numéricas se conservan como faltantes; no se convierten artificialmente a 0.

El panel de esta corrida ajusta GB035 (contrastes de distancia en demostrativos), GB480m (caso morfológico), GB132 (orden verbo-medial) y GB408 (alineamiento acusativo en la marcación). GB408 se incluye como rasgo relacionado con GB480m: entre 1.518 lenguas elegibles con ambos valores, la correlación binaria descriptiva es aproximadamente 0,35. Esta asociación marginal todavía mezcla covariación espacial, filogenética y otras fuentes; el modelo estima por separado la covariación residual de sus campos. El conjunto completo y su catálogo se conservan; el panel que entra a NUTS está listado en `config/model_gbi.json` y en el informe de esa corrida.

La curación estadística del artículo intenta reducir ciertas dependencias fuertes entre rasgos que no se atribuyen a historia compartida o área. No transforma la tabla en datos estadísticamente independientes en sentido absoluto ni elimina toda asociación residual. Esta distinción importa: la estimación de `Σ` describe asociación entre rasgos dentro del efecto espacial o del efecto filogenético del modelo.

## Alcance común

Para que una lengua forme parte del ajuste necesita coordenadas válidas y un glottocódigo presente en el árbol MCC. Las demás filas se mantienen en la tabla de preparación para que la cobertura sea visible, pero no entran a NUTS ni reciben una imputación del modelo actual.

| Conjunto | Filas originales | Con coordenadas válidas | En el árbol | Rasgos que se modelan |
|---|---:|---:|---:|---|
| WALS | 2.694 | 2.662 | 2.321 | 83A, 9A, 6A |
| GBI estadístico | 2.467 | 2.463 | 2.205 | GB035, GB480m, GB132, GB408 |

Los conteos de esta tabla describen la preparación actual. Consulta `data/processed/<conjunto>/audit.json` para la auditoría de una ejecución concreta. WALS contiene alias con el mismo glottocódigo; el split los agrupa para evitar que el mismo taxón aparezca en train y test. GBI tiene una fila por glottocódigo en esta versión.

## Procedencia

Los ficheros GBI usados corresponden a la release v1.0.0 (commit `77a8cb9`) del repositorio del proyecto, que distribuye los datos bajo CC BY 4.0. Las sumas SHA-256 se guardan en la auditoría para identificar exactamente la entrada. Consulta [`data/source/gbi/README.md`](../data/source/gbi/README.md) para la cita completa.
