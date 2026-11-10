# Modelo multirrespuesta y matrices de covarianza

Este documento reúne la formulación del modelo y la explicación de sus matrices. El orden va de la pregunta estadística a la construcción de las covarianzas y, finalmente, al muestreo y al código.

## 1. Qué intenta aprender el modelo

Tenemos N lenguas y K rasgos binarios. Para cada pareja lengua–rasgo, la respuesta observada es 0 o 1. Algunos valores faltan. El objetivo es usar las respuestas conocidas junto con la geografía y la genealogía para estimar, en cada celda faltante, una probabilidad de que el rasgo valga 1.

El modelo no supone que todos los rasgos sean iguales ni que todas las lenguas estén relacionadas de la misma manera. Separa tres preguntas:

1. ¿Cuál es la tendencia general de cada rasgo?
2. ¿Qué similitud comparten las lenguas por ubicación o por historia?
3. ¿Qué rasgos tienden a variar juntos dentro de cada tipo de dependencia?

La respuesta se expresa primero en una escala continua llamada predictor latente:

$$
\eta_{ik} = \alpha_k + g_{ik} + h_{ik}.
$$

- i identifica una lengua; k, un rasgo.
- αₖ es la tendencia global del rasgo k, antes de añadir la información espacial y filogenética.
- gᵢₖ es la desviación asociada al campo espacial.
- hᵢₖ es la desviación asociada al campo filogenético.

La escala de η es probit: no es una probabilidad ni una medición observada. Se convierte en probabilidad mediante la función de distribución acumulada de una normal estándar:

$$
p_{ik} = P(y_{ik}=1) = \Phi(\eta_{ik}).
$$

Φ(x) devuelve el área acumulada a la izquierda de x bajo una normal con media 0 y desviación estándar 1. Por eso siempre produce un número entre 0 y 1.

Una manera equivalente de entender la regla es imaginar una propensión continua no observada:

$$
y^{*}_{ik} = \eta_{ik} + \epsilon_{ik},
\qquad \epsilon_{ik} \sim \mathcal{N}(0,1),
\qquad y_{ik} =
\begin{cases}
1 & \text{si } y^{*}_{ik}>0,\\
0 & \text{si } y^{*}_{ik}\leq 0.
\end{cases}
$$

La varianza del ruido se fija en 1 para definir la escala del probit. Sin esa referencia, se podrían multiplicar simultáneamente η y el desvío del ruido por una constante sin cambiar las probabilidades; la escala quedaría sin identificar.

Una celda observada sí contribuye a la verosimilitud; una celda faltante queda enmascarada y no se trata como cero. Para una celda observada:

$$
\log P(y_{ik}\mid\eta_{ik}) =
\begin{cases}
\log \Phi(\eta_{ik}) & \text{si } y_{ik}=1,\\
\log \Phi(-\eta_{ik}) & \text{si } y_{ik}=0.
\end{cases}
$$

El programa suma estas contribuciones solo para las celdas observadas que pertenecen al ajuste.

## 2. Dos ejes distintos: lenguas y rasgos

Cada componente aleatoria es una matriz con lenguas en las filas y rasgos en las columnas:

$$
G \in \mathbb{R}^{N\times K}.
$$

Para describir la covarianza de G hacen falta dos matrices, porque contestan preguntas distintas:

| Matriz | Eje que relaciona | Pregunta |
|---|---|---|
| R (N × N) | Lenguas | ¿Qué tan relacionados están los efectos de dos lenguas? |
| Σ (K × K) | Rasgos | ¿Qué tan relacionados están los efectos de dos rasgos? |

El modelo tiene una pareja para cada componente:

- R espacial y Σ espacial;
- R filogenética y Σ filogenética.

R no es una matriz de rasgos, y Σ no es una matriz de lenguas. No son duplicados: describen los dos ejes de la matriz lengua × rasgo.

### Qué significa una entrada

Para un componente con covarianzas R y Σ, la covarianza entre la celda del rasgo k en la lengua i y la celda del rasgo l en la lengua j es:

$$
\mathrm{Cov}(G_{ik},G_{jl}) = R_{ij}\,\Sigma_{kl}.
$$

Por ejemplo, si Rᵢⱼ = 0,8 y Σₖₗ = 0,6, la covarianza entre esos dos efectos es 0,48. El primer número describe la relación entre las lenguas; el segundo, la relación entre los rasgos. La regla de multiplicación dice que la asociación entre dos celdas depende de ambas relaciones a la vez.

Si se trata del mismo rasgo (k=l), Σₖₖ aporta su varianza y Rᵢⱼ determina cuánto se asocia ese rasgo entre lenguas. Si se trata de dos rasgos de una misma lengua (i=j), Rᵢᵢ aporta la varianza de esa lengua y Σₖₗ describe su covariación entre rasgos.

Estas ecuaciones describen los campos latentes g y h, no una correlación observada directa entre las respuestas binarias. Las respuestas se relacionan a través de esos campos y de la transformación probit.

## 3. Por qué aparece el producto de Kronecker

Una matriz normal es una distribución para matrices que conserva explícitamente los dos ejes. Se escribe:

$$
G \sim \mathcal{MN}(0,R,\Sigma).
$$

Con la convención de apilar las columnas de G en el vector vec(G), esto equivale a:

$$
\mathrm{vec}(G) \sim
\mathcal{N}\left(0,\ \Sigma\otimes R\right).
$$

⊗ es el producto de Kronecker. En este caso, Σ ⊗ R es una matriz de tamaño (N K) × (N K). Cada bloque correspondiente al par de rasgos k,l es Σₖₗ R. Es otra forma de escribir, de una sola vez, la regla:

$$
\mathrm{Cov}(G_{ik},G_{jl}) = R_{ij}\Sigma_{kl}.
$$

El orden de los factores depende de cómo se aplana la matriz: al apilar filas, la expresión equivalente intercambia el orden. La relación entre celdas no cambia.

### Por qué se multiplican las dos covarianzas

La multiplicación no afirma que genealogía y geografía sean el mismo fenómeno. Expresa una estructura separable dentro de cada campo: la relación entre dos celdas se obtiene combinando la semejanza entre sus lenguas con la covariación entre sus rasgos. Esta es una decisión de modelado parsimoniosa. Una covarianza completamente libre entre N K celdas tendría (N K)² entradas y sería inviable con estos datos.

El código no construye la enorme matriz Σ ⊗ R. Usa factores de menor tamaño y obtiene la misma distribución. Si R = C Cᵀ y Σ = T Tᵀ, se puede generar:

$$
G = C Z T^{\mathsf{T}},
\qquad Z_{ab}\ \text{independientes con}\ Z_{ab}\sim\mathcal{N}(0,1).
$$

En efecto, la multiplicación por C induce la covarianza entre lenguas y la multiplicación por T induce la covarianza entre rasgos. Al combinar ambas, la covarianza de las celdas es el producto anterior. Esta construcción se llama parametrización no centrada y evita factorizar una matriz densa de dimensión N K.

## 4. La matriz de lenguas espaciales R espacial

El modelo representa el globo con armónicos esféricos reales. La matriz B evalúa cada patrón espacial en cada lengua:

$$
B_{im} = \text{valor del patrón esférico }m\text{ en la ubicación de la lengua }i.
$$

Los patrones de grado bajo describen cambios amplios; los de grado alto permiten variaciones más finas. El grado máximo efectivo L y el parámetro de suavizado λ determinan cuánto se atenúan esos patrones finos. Para cada patrón de grado ℓ, el peso es:

$$
w_\ell =
\frac{\mathrm{logistic}((L+1/2-\ell)/\tau)}
{1+\lambda[\ell(\ell+1)]^{2}},
\qquad
D_{mm} =
\sqrt{\frac{w_{\ell(m)}}{\sum_q w_{\ell(q)}}}.
$$

D es diagonal: no mezcla patrones, solo cambia su amplitud. En la suma del denominador cada grado aparece tantas veces como patrones de ese grado.

Primero se genera un campo espacial estándar y luego se calcula su covarianza:

$$
Z_{\mathrm{esp}}\sim\mathcal{N}(0,I),
\qquad
G_{\mathrm{esp}} =
B D Z_{\mathrm{esp}} T_{\mathrm{esp}}^{\mathsf{T}},
$$

$$
R_{\mathrm{espacial}} = B D^{2} B^{\mathsf{T}}.
$$

B D es un factor de la covarianza espacial, pues (B D)(B D)ᵀ = B D² Bᵀ. Los pesos normalizados hacen que la diagonal de R espacial sea uno, salvo redondeo numérico. Así, la amplitud de cada rasgo queda en Σ espacial y R espacial se concentra en describir la forma de la dependencia entre ubicaciones.

En la configuración actual, L se infiere entre 4 y 16 y λ se infiere con un prior lognormal. τ, el ancho de la transición suave alrededor de L, se mantiene fijo. La base se calcula hasta el máximo configurado para que sus dimensiones no cambien mientras NUTS explora L.

Esta covarianza previa es isotrópica sobre la esfera: pares de puntos a la misma distancia angular reciben la misma correlación previa. La superficie posterior puede tener formas regionales distintas porque las observaciones se distribuyen de forma desigual, pero el kernel no aprende una longitud de alcance distinta para cada región.

## 5. La matriz de lenguas filogenéticas R filogenética

El punto de partida es el árbol MCC de Global Language Tree Pipeline. El árbol incorpora una reconstrucción geográfica además de información genealógica; por eso R fil puede retener señal espacial. El modelo conserva, aun así, un término espacial separado para distinguir las dos fuentes estadísticas de estructura. Para las hojas i y j, la covarianza browniana sin normalizar suma la longitud de las ramas compartidas desde la raíz hasta su ancestro común:

$$
C^{\mathrm{árbol}}_{ij} = \sum_{e\in P_i\cap P_j} t_e.
$$

Dos lenguas que comparten una historia reciente acumulan más ramas compartidas. La diagonal $C^{\mathrm{árbol}}_{ii}$ es la longitud total de la raíz a la hoja $i$.

El código transforma esa covarianza en correlación:

$$
R_{ij} =
\frac{C^{\mathrm{árbol}}_{ij}}{\sqrt{C^{\mathrm{árbol}}_{ii}C^{\mathrm{árbol}}_{jj}}}.
$$

La normalización deja diagonal uno y hace comparables las relaciones de las hojas aunque las distancias raíz–punta difieran. En la configuración actual, después de normalizar, las entradas entre familias distintas se fijan en cero. Dentro de una familia se conservan las relaciones relativas del árbol. Esa regla es una decisión explícita del análisis, no una propiedad universal de la historia lingüística.

Luego se obtiene un factor A fil tal que:

$$
R_{\mathrm{filogenética}} =
A_{\mathrm{fil}}A_{\mathrm{fil}}^{\mathsf{T}},
\qquad
G_{\mathrm{fil}} =
A_{\mathrm{fil}} Z_{\mathrm{fil}} T_{\mathrm{fil}}^{\mathsf{T}}.
$$

A fil es una raíz matricial de R fil; no es la matriz original de tiempos compartidos. Tampoco se invierte R fil. La factorización permite generar efectos correlacionados usando normales independientes. Para predecir lenguas objetivo, el código condiciona sus efectos filogenéticos a las lenguas de entrenamiento y agrega la incertidumbre residual correspondiente.

El modelo usa un solo árbol MCC y una matriz fijada a partir de él. No promedia los árboles alternativos ni aprende la estructura filogenética de cero.

## 6. La covarianza entre rasgos Σ

Cada componente tiene su propia matriz de covarianza de rasgos. Se construye separando la escala de la correlación:

$$
\Sigma = D_\sigma\,\Omega\,D_\sigma.
$$

- Dσ es diagonal y contiene una amplitud σₖ para cada rasgo.
- Ω es una matriz de correlación entre rasgos, con diagonal uno.
- Σₖₗ es la covarianza entre los efectos de los rasgos k y l para una lengua con Rᵢᵢ=1.

El prior se aplica a una factorización de Cholesky de Ω. Si Ω = LΩ LΩᵀ, definimos T = Dσ LΩ; entonces Σ = T Tᵀ. Esta forma garantiza que Σ sea semidefinida positiva. No se intenta estimar libremente cada entrada de Σ, lo que podría producir una matriz inválida.

El modelo estima Σ espacial y Σ filogenética por separado. Esto permite, por ejemplo, que dos rasgos covaríen en sus patrones geográficos pero no en sus patrones filogenéticos, o al revés. Los valores de R aportan las estructuras de las lenguas; las Σ describen la asociación entre rasgos dentro de cada estructura.

El prior LKJ(2) regulariza las correlaciones hacia valores moderados y evita que unas pocas observaciones fuercen correlaciones extremas. La matriz de rasgos no se fija a partir de una correlación descriptiva previa: se estima durante el ajuste y su incertidumbre se resume con muestras posteriores.

## 7. Cómo se combinan geografía y genealogía

Los dos campos se suman en el predictor:

$$
\eta_{ik}=\alpha_k+g_{ik}+h_{ik}.
$$

En el prior, el campo espacial y el filogenético son independientes entre sí, condicionados a sus parámetros. Por eso sus covarianzas se suman. Con columnas apiladas en vec(G), la covarianza de los dos campos estructurados es:

$$
\mathrm{Cov}(\mathrm{vec}(G_{\mathrm{esp}}+G_{\mathrm{fil}}))
=
\Sigma_{\mathrm{espacial}}\otimes R_{\mathrm{espacial}}
+
\Sigma_{\mathrm{filogenética}}\otimes R_{\mathrm{filogenética}}.
$$

Estas expresiones describen la covarianza previa de los campos, condicionada a los hiperparámetros. Tras observar los datos, la incertidumbre posterior también refleja la estimación de esos parámetros y las dependencias que induce la verosimilitud.

En términos de celdas:

$$
\mathrm{Cov}(\eta_{ik},\eta_{jl})
=
R_{\mathrm{espacial},ij}\Sigma_{\mathrm{espacial},kl}
+
R_{\mathrm{filogenética},ij}\Sigma_{\mathrm{filogenética},kl}
+
\mathrm{Cov}(\alpha_k,\alpha_l).
$$

El último término aparece porque el intercepto de cada rasgo es compartido por todas las lenguas. Con los priors independientes actuales, es cero entre rasgos distintos e igual a intercept_scale² para el mismo rasgo. El ruido probit independiente aparece al pasar del predictor η a la propensión $y^{*}$; su varianza se fija en 1 y no forma parte de la covarianza estructurada de η.

Esta suma explica por qué no hay una sola matriz que represente todo: el componente espacial y el filogenético tienen distintas matrices de lenguas y, además, pueden tener distintas covarianzas entre rasgos.

Las amplitudes σ espacial y σ filogenética no son porcentajes causales. Las dos estructuras pueden parecerse en una región y competir por explicar parte del mismo patrón. Para interpretarlas se revisan sus posteriores, diagnósticos, predicciones y evaluaciones diseñadas para distinguir espacio de genealogía. Una fracción basada en varianzas marginales puede resumir escala, pero no demuestra qué componente es causal ni cuál mejora más la predicción.

## 8. Priors, ajuste bayesiano y predicción

Los principales priors están declarados en config/inference.json:

| Cantidad | Prior | Qué regulariza o permite |
|---|---|---|
| αₖ | Normal(0, escala 1,5) | Nivel global en escala probit. |
| Amplitudes σ | Half-Normal(1) | Variación no negativa por rasgo y componente. |
| Correlaciones Ω | LKJ(2) mediante Cholesky | Dependencia entre rasgos sin salir del espacio de matrices válidas. |
| L | Uniforme(4, 16) | Grado efectivo de detalle espacial. |
| λ | Lognormal | Atenuación de los patrones espaciales de frecuencia alta. |
| Coeficientes Z | Normal estándar independiente | Variables base que, al transformarse, generan campos correlacionados. |
| Ruido probit | Normal estándar, varianza fija en 1 | Define la escala de la propensión latente. |

Para una muestra posterior se construyen Σ, g, h y η; se calcula la probabilidad Φ(η) y se evalúan las respuestas observadas. NUTS explora conjuntamente parámetros globales, correlaciones entre rasgos, amplitudes y campos latentes. Los datos actualizan los priors mediante la verosimilitud; el resultado es una distribución posterior, no un único conjunto definitivo de parámetros.

El pipeline hace dos ajustes:

1. **Evaluación:** reserva aproximadamente el 20 % de las celdas observadas por rasgo, mantiene la proporción de clases y agrupa por glottocódigo. El ajuste usa solo las celdas de entrenamiento; sus probabilidades de test se comparan con las etiquetas guardadas.
2. **Imputación:** vuelve a ajustar desde cero usando todas las celdas conocidas y calcula predicciones para los valores faltantes elegibles. El segundo ajuste reestima los parámetros; no copia los del ajuste de evaluación.

Cada muestra posterior s da una probabilidad pᵢₖ⁽ˢ⁾ para una celda. La media de esas probabilidades es la predicción posterior:

$$
\bar p_{ik}=\frac{1}{S}\sum_{s=1}^{S}p_{ik}^{(s)}.
$$

Los percentiles 2,5 y 97,5 resumen un intervalo creíble puntual del 95 %. Los draws conservan incertidumbre sobre los parámetros, las matrices entre rasgos y los campos latentes. Un intervalo amplio indica que la probabilidad de la celda cambia apreciablemente entre escenarios posteriores.

## 9. Qué se estima y qué se mantiene fijo

**Se estima durante el ajuste:** los interceptos; las amplitudes por rasgo de cada componente; las correlaciones entre rasgos espaciales y filogenéticas; L; λ; y los coeficientes latentes que forman los campos.

**Se mantiene fijo:** las coordenadas; la familia lingüística asignada; el árbol MCC; la estructura base de R filogenética; la definición de armónicos y la regla de covarianza cero entre familias. R espacial cambia entre muestras al variar L y λ, pero su familia de kernels permanece fijada por la base esférica elegida.

La matriz R no se invierte para generar los efectos. Se factoriza o se representa con una base para poder multiplicar normales independientes por una raíz de covarianza. Las resoluciones lineales que aparecen al condicionar un GP en las lenguas de entrenamiento sirven para obtener medias y covarianzas predictivas; no significan que la covarianza entre rasgos sea una inversa.

## 10. Dónde está cada paso en el código

| Archivo | Responsabilidad |
|---|---|
| config/model.json, config/model_gbi.json | Fuentes, rasgos, codificaciones y reglas del panel. |
| config/inference.json | Priors, hiperparámetros y controles de NUTS. |
| data_sources/ y data.py | Carga y armonización de fuentes, respuestas observadas y celdas faltantes. |
| phylogeny.py | Obtiene el MCC, calcula covarianza de ramas compartidas, normaliza R fil y aplica la regla entre familias. |
| design.py | Construye los armónicos esféricos y los factores para las predicciones filogenéticas. |
| model.py | Define priors, Σ, los campos, η y la verosimilitud probit enmascarada. |
| inference.py | Ejecuta NUTS, revisa diagnósticos y convierte draws en probabilidades. |
| pipeline.py | Coordina preparación, split, evaluación, re-ajuste e imputación. |
| report.py, visualization.py, site.py | Guardan métricas, vistas por lengua, mapas y portal. |

Los resultados y diagnósticos actuales se consultan desde el portal o en results/current/<conjunto>/run.json y report.md. Una corrida con advertencia de convergencia puede producir archivos y mapas, pero sus resúmenes posteriores no deben interpretarse como estabilizados.

## Lecturas

- Halliwell, Holland y Yates (2025), [Multi-response phylogenetic mixed models](https://doi.org/10.1111/brv.70001).
- Guzmán Naranjo y Becker (2022), [Statistical control of areal and genealogical biases](https://doi.org/10.1515/lingty-2021-0002).
- Release MCC v2.0.0 del [Global Language Tree Pipeline](https://github.com/rbouckaert/global-language-tree-pipeline/releases/tag/v2.0.0).
- Graff et al. (2025), [Curating global datasets of structural linguistic features for independence](https://doi.org/10.1038/s41597-024-04319-4).



