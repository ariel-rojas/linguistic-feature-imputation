/* Explorador ligero: presenta histogramas resumidos, no vuelve a ajustar el modelo. */
const data = JSON.parse(document.getElementById("data").textContent);
const $ = (id) => document.getElementById(id);
const colors = { probability: "#15768a", parameter: "#6a3d9a" };
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[char]);
const percent = (value) => `${(100 * value).toFixed(1)} %`;
const number = (value) => Number(value).toFixed(3);

function histogram(summary, { probability = false, color = colors.parameter } = {}) {
  const width = 640, height = 230, left = 55, right = 16, top = 24, bottom = 42;
  const edges = summary.edges;
  const masses = summary.mass;
  const min = edges[0], max = edges[edges.length - 1];
  const peak = Math.max(...masses, 0.01);
  const x = (value) => left + ((value - min) / (max - min || 1)) * (width - left - right);
  const y = (value) => height - bottom - (value / peak) * (height - top - bottom);
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Histograma de muestras posteriores">`;
  svg += `<rect x="${x(summary.interval[0])}" y="${top}" width="${Math.max(0, x(summary.interval[1]) - x(summary.interval[0]))}" height="${height - top - bottom}" fill="${color}" opacity="0.11"/>`;
  masses.forEach((mass, index) => {
    const x0 = x(edges[index]), x1 = x(edges[index + 1]);
    svg += `<rect x="${x0 + 1}" y="${y(mass)}" width="${Math.max(0, x1 - x0 - 2)}" height="${height - bottom - y(mass)}" fill="${color}"><title>${format(edges[index], probability)}–${format(edges[index + 1], probability)}: ${percent(mass)} de las muestras</title></rect>`;
  });
  svg += `<line x1="${x(summary.mean)}" x2="${x(summary.mean)}" y1="${top}" y2="${height - bottom}" stroke="#152f40" stroke-width="2.5"/>`;
  svg += `<line x1="${left}" x2="${width - right}" y1="${height - bottom}" y2="${height - bottom}" stroke="#7e909d"/>`;
  for (const value of [min, (min + max) / 2, max]) {
    const anchor = value === min ? "start" : value === max ? "end" : "middle";
    svg += `<text x="${x(value)}" y="${height - 15}" text-anchor="${anchor}">${format(value, probability)}</text>`;
  }
  svg += `<text x="${left - 7}" y="${top + 3}" text-anchor="end">${percent(peak)}</text>`;
  svg += `<text x="${left - 7}" y="${height - bottom + 4}" text-anchor="end">0</text>`;
  svg += `<text x="${width / 2}" y="${height - 1}" text-anchor="middle">${probability ? "Probabilidad de la clase positiva" : "Valor del parámetro"}</text>`;
  return svg + "</svg>";
}
function format(value, probability) {
  return probability ? `${Math.round(value * 100)} %` : Number(value).toPrecision(3);
}

function setTab(view) {
  $("predictionView").hidden = view !== "predictions";
  $("parameterView").hidden = view !== "parameters";
  $("metricsView").hidden = view !== "metrics";
  $("predictionsTab").setAttribute("aria-pressed", String(view === "predictions"));
  $("parametersTab").setAttribute("aria-pressed", String(view === "parameters"));
  $("metricsTab").setAttribute("aria-pressed", String(view === "metrics"));
  if (view === "parameters") renderParameters();
  if (view === "metrics") renderMetrics();
}

function casesForSelection() {
  const stage = $("stage").value;
  const feature = $("feature").value;
  const query = $("search").value.trim().toLocaleLowerCase();
  return data.cases.filter((item) => item.stage === stage && item.feature === feature)
    .filter((item) => `${item.name} ${item.id} ${item.family}`.toLocaleLowerCase().includes(query))
    .sort((a, b) => a.name.localeCompare(b.name, "es"));
}

function renderSummary() {
  const stage = $("stage").value;
  const feature = $("feature").value;
  const split = data.split.by_feature[feature];
  const metrics = data.metrics[feature].metrics;
  const missing = data.n_imputed[feature];
  const selectedCases = casesForSelection().length;
  if (stage === "evaluation") {
    $("stageExplanation").textContent = "El modelo se ajusta con train; el histograma corresponde a las predicciones de las lenguas reservadas en test.";
    $("stageSummary").innerHTML = [
      ["Train", split.train], ["Test reservado", split.test],
      ["Aciertos en test", `${metrics.correct} / ${metrics.n_test}`],
      ["Exactitud", percent(metrics.accuracy)],
    ].map(([label, value]) => statCard(label, value)).join("");
  } else {
    $("stageExplanation").textContent = "El modelo se reajusta con todas las respuestas observadas; las distribuciones corresponden solo a celdas faltantes imputables.";
    $("stageSummary").innerHTML = [
      ["Datos observados usados", split.train + split.test],
      ["Valores faltantes imputados", missing],
      ["Casos en esta lista", selectedCases],
      ["Ajuste", "Todos los observados"],
    ].map(([label, value]) => statCard(label, value)).join("");
  }
  const positive = data.features.find((item) => item.slug === feature).positive_label;
  $("probabilityDescription").textContent = `Las barras muestran la distribución posterior de P(${positive}) para esta lengua. La línea vertical marca la media; la franja sombreada contiene el intervalo creíble del 95 %.`;
}

function statCard(label, value) {
  return `<div class="stat-card"><span>${escapeHTML(label)}</span><strong>${escapeHTML(value)}</strong></div>`;
}

function renderCase(preferredId = $("caseList").value) {
  const options = casesForSelection();
  $("caseList").replaceChildren(...options.map((item) => new Option(`${item.name} · ${item.id} · ${item.family}`, item.id)));
  if (options.some((item) => item.id === preferredId)) $("caseList").value = preferredId;
  $("caseCount").textContent = `${options.length} casos disponibles para esta etapa y rasgo.`;
  renderSummary();

  const selected = options.find((item) => item.id === $("caseList").value);
  if (!selected) {
    $("languageName").textContent = "No hay casos que coincidan";
    $("languageMeta").textContent = "Cambia la búsqueda, la etapa o el rasgo.";
    $("caseOutcome").replaceChildren();
    $("probabilityChart").replaceChildren();
    $("drawCount").textContent = "";
    return;
  }

  const feature = data.features.find((item) => item.slug === selected.feature);
  const posterior = selected.probability;
  $("caseStage").textContent = selected.stage === "evaluation" ? "Predicción fuera de muestra · test" : "Imputación final · faltante no observado";
  $("languageName").textContent = selected.name;
  $("languageMeta").textContent = `Identificador: ${selected.id} · Glottocódigo: ${selected.glottocode} · Familia: ${selected.family}`;
  const status = selected.stage === "evaluation"
    ? `<span class="pill ${selected.correct ? "correct" : "incorrect"}">${selected.correct ? "Acierto" : "Error"} en test</span>`
    : '<span class="pill imputed">Valor imputado</span>';
  $("caseOutcome").innerHTML = `${status}<div class="outcome"><div class="outcome-card"><span>Valor observado</span><strong>${escapeHTML(selected.observed)}</strong></div><div class="outcome-card"><span>Clase predicha · umbral 0,5</span><strong>${escapeHTML(selected.predicted)}</strong></div><div class="outcome-card"><span>P(${escapeHTML(feature.positive_label)})</span><strong>${percent(posterior.mean)}</strong></div><div class="outcome-card"><span>Intervalo creíble 95 %</span><strong>${percent(posterior.interval[0])}–${percent(posterior.interval[1])}</strong></div></div>`;
  $("probabilityChart").innerHTML = histogram(posterior, { probability: true, color: colors.probability });
  $("drawCount").textContent = `${posterior.n} draws predictivos resumidos. La probabilidad integrada de ${feature.positive_label} es ${percent(posterior.mean)}; la de ${feature.negative_label}, ${percent(1 - posterior.mean)}.`;
  updateAddress(selected);
}

function parameterFor(result, key) {
  return result.parameters.find((parameter) => parameter.key === key);
}

function pairParameter(result, component, firstSlug, secondSlug) {
  return result.parameters.find((parameter) =>
    parameter.component === component
    && parameter.scope === "pair"
    && parameter.pair_slugs.includes(firstSlug)
    && parameter.pair_slugs.includes(secondSlug)
  );
}

function formatParameter(value) {
  const magnitude = Math.abs(value);
  if (magnitude >= 100) return Number(value).toFixed(0);
  if (magnitude >= 10) return Number(value).toFixed(1);
  if (magnitude >= 1) return Number(value).toFixed(2);
  return Number(value).toPrecision(2);
}

function comparisonChart(parameter) {
  const width = 700, height = 250, left = 62, right = 18, top = 24, bottom = 44;
  const edges = parameter.edges;
  const centers = edges.slice(0, -1).map((edge, index) => (edge + edges[index + 1]) / 2);
  const posterior = parameter.posterior_density;
  const prior = parameter.prior.density;
  const peak = Math.max(...posterior, ...prior, 0.001);
  const min = edges[0], max = edges[edges.length - 1];
  const x = (value) => left + ((value - min) / (max - min || 1)) * (width - left - right);
  const y = (value) => height - bottom - (value / peak) * (height - top - bottom);
  const chartId = "chart-" + parameter.key.replace(/[^a-zA-Z0-9-]/g, "-");
  let svg = '<svg id="' + chartId + '" viewBox="0 0 ' + width + ' ' + height +
    '" role="img" aria-label="Comparación de distribución previa y posterior para ' +
    escapeHTML(parameter.title) + '">';
  svg += '<title>Prior y posterior: ' + escapeHTML(parameter.title) + '</title>';
  [0.25, 0.5, 0.75, 1].forEach((fraction) => {
    const gridY = y(peak * fraction);
    svg += '<line x1="' + left + '" x2="' + (width - right) + '" y1="' + gridY +
      '" y2="' + gridY + '" stroke="#dfe5e9" stroke-width="1"/>';
  });
  const lower = Math.max(min, parameter.interval[0]);
  const upper = Math.min(max, parameter.interval[1]);
  svg += '<rect x="' + x(lower) + '" y="' + top + '" width="' +
    Math.max(0, x(upper) - x(lower)) + '" height="' + (height - top - bottom) +
    '" fill="#2864a6" opacity="0.09"/>';
  posterior.forEach((density, index) => {
    const x0 = x(edges[index]), x1 = x(edges[index + 1]);
    svg += '<rect x="' + (x0 + 1) + '" y="' + y(density) + '" width="' +
      Math.max(0, x1 - x0 - 2) + '" height="' + (height - bottom - y(density)) +
      '" fill="#2864a6" opacity="0.60"><title>Posterior ' +
      formatParameter(edges[index]) + ' a ' + formatParameter(edges[index + 1]) +
      ': densidad ' + number(density) + '</title></rect>';
  });
  const path = centers.map((center, index) =>
    (index === 0 ? "M " : "L ") + x(center) + " " + y(prior[index])
  ).join(" ");
  svg += '<path d="' + path +
    '" fill="none" stroke="#c56a24" stroke-width="3" stroke-dasharray="7 4" pointer-events="none"/>';
  svg += '<line x1="' + x(parameter.mean) + '" x2="' + x(parameter.mean) +
    '" y1="' + top + '" y2="' + (height - bottom) +
    '" stroke="#152f40" stroke-width="2"/>';
  svg += '<line x1="' + left + '" x2="' + (width - right) +
    '" y1="' + (height - bottom) + '" y2="' + (height - bottom) +
    '" stroke="#687985" stroke-width="1.2"/>';
  [min, (min + max) / 2, max].forEach((value, index) => {
    const anchor = index === 0 ? "start" : index === 2 ? "end" : "middle";
    svg += '<text x="' + x(value) + '" y="' + (height - 16) +
      '" text-anchor="' + anchor + '">' + formatParameter(value) + '</text>';
  });
  svg += '<text x="' + (width / 2) + '" y="' + (height - 1) +
    '" text-anchor="middle">Valor del parámetro</text></svg>';
  return svg;
}

function comparisonPanel(parameter, heading) {
  if (!parameter) return '<p class="empty-state">No hay draws para este parámetro en el ajuste elegido.</p>';
  const title = heading || parameter.title;
  return '<article class="distribution-card">' +
    '<h4>' + escapeHTML(title) + '</h4>' +
    '<p class="distribution-description">' + escapeHTML(parameter.description) + '</p>' +
    '<div class="chart-legend" aria-label="Leyenda">' +
      '<span><i class="legend-posterior"></i> Posterior</span>' +
      '<span><i class="legend-prior"></i> Prior</span>' +
      '<span><i class="legend-mean"></i> Media posterior</span></div>' +
    '<div class="comparison-chart">' + comparisonChart(parameter) + '</div>' +
    '<div class="distribution-summary">' +
      '<p><span>Prior · intervalo central 95 %</span><strong>' +
        number(parameter.prior.interval[0]) + ' a ' +
        number(parameter.prior.interval[1]) + '</strong><small>' +
        escapeHTML(parameter.prior.label) + '</small></p>' +
      '<p><span>Posterior · media e intervalo creíble 95 %</span><strong>' +
        number(parameter.mean) + ' · ' + number(parameter.interval[0]) + ' a ' +
        number(parameter.interval[1]) + '</strong><small>' +
        parameter.n + ' draws</small></p></div></article>';
}

function renderParameters() {
  const featureSlug = $("parameterFeature").value;
  const feature = data.features.find((item) => item.slug === featureSlug);
  const host = $("parameterCards");
  if (!feature) {
    host.innerHTML = '<p class="empty-state">Selecciona un rasgo para abrir sus distribuciones y componentes.</p>';
    $("diagnostics").innerHTML = "";
    return;
  }

  const previousPeer = document.getElementById("relatedFeature")
    ? $("relatedFeature").value : "";
  const peers = data.features.filter((item) => item.slug !== feature.slug);
  const result = data.stages[$("parameterStage").value];
  const slug = feature.slug;
  const intercept = parameterFor(result, "posterior_intercept_" + slug);
  const spatialSigma = parameterFor(result, "posterior_sigma_spatial_" + slug);
  const phylogeneticSigma = parameterFor(result, "posterior_sigma_phylogenetic_" + slug);
  const varianceShare = parameterFor(result, "derived_spatial_variance_share_" + slug);
  const shareExplanation = data.phylogeny_normalized
    ? "Con ambas matrices R de las lenguas normalizadas a diagonal uno, 0,8 significa que el campo espacial representa el 80 % de la varianza latente marginal total de estos dos campos para este rasgo. No significa 80 % de probabilidad ni es una fracción específica por lengua. La línea y el intervalo previos muestran el punto de partida de las dos priors HalfNormal independientes e iguales."
    : "La matriz filogenética no está normalizada a diagonal uno; esta razón compara las amplitudes al cuadrado, pero no equivale directamente a la fracción de varianza marginal. No significa una probabilidad ni es un coeficiente ajustado aparte."
  const maxDegree = parameterFor(result, "posterior_max_degree");
  const smoothing = parameterFor(result, "posterior_smoothing");

  host.innerHTML =
    '<section class="selected-feature"><p class="eyebrow">Rasgo seleccionado</p>' +
      '<h2>' + escapeHTML(feature.id) + ' · ' + escapeHTML(feature.name) + '</h2>' +
      '<p>Clase positiva: <strong>' + escapeHTML(feature.positive_label) +
      '</strong> · Clase negativa: <strong>' + escapeHTML(feature.negative_label) + '</strong></p></section>' +
    '<section class="component-section baseline-section">' +
      '<div class="component-heading"><div><p class="component-symbol">αₖ</p><h3>Tendencia global</h3></div>' +
      '<p>Nivel base del rasgo antes de sumar sus desviaciones espaciales y filogenéticas.</p></div>' +
      '<div class="parameter-grid">' + comparisonPanel(intercept) + '</div></section>' +
    '<section class="component-grid">' +
      '<article class="component-section spatial-section">' +
        '<div class="component-heading"><div><p class="component-symbol">gᵢₖ</p><h3>Componente espacial</h3></div>' +
        '<p>R espacial relaciona lenguas por ubicación; Σ espacial relaciona rasgos dentro de este campo.</p></div>' +
        '<div class="component-equation">G espacial ∼ MN(0, R espacial, Σ espacial)</div>' +
        '<div class="parameter-grid">' + comparisonPanel(spatialSigma) + '</div>' +
        '<details class="global-parameters"><summary>Forma y suavizado · compartidos por todos los rasgos</summary>' +
        '<p>L y λ modifican R espacial. λ atenúa patrones finos; no mezcla geografía con filogenia.</p>' +
        '<div class="parameter-grid">' + comparisonPanel(maxDegree) + comparisonPanel(smoothing) + '</div></details>' +
      '</article>' +
      '<article class="component-section phylogenetic-section">' +
        '<div class="component-heading"><div><p class="component-symbol">hᵢₖ</p><h3>Componente filogenético</h3></div>' +
        '<p>R filogenética relaciona lenguas por el árbol MCC; Σ filogenética relaciona rasgos dentro del campo.</p></div>' +
        '<div class="component-equation">G filogenético ∼ MN(0, R filogenética, Σ filogenética)</div>' +
        '<div class="parameter-grid">' + comparisonPanel(phylogeneticSigma) + '</div>' +
        '<p class="fixed-structure"><strong>Estructura fija:</strong> árbol MCC y regla entre familias fijos; amplitudes y correlaciones entre rasgos inferidas.</p>' +
      '</article></section>' +
    '<section class="weight-comparison-section">' +
      '<div class="component-heading"><div><p class="component-symbol">Resumen derivado</p><h3>¿Qué amplitud relativa tienen los campos?</h3></div>' +
      '<p>La página no estima un coeficiente de mezcla. Este resumen compara la varianza marginal de los dos efectos para el rasgo elegido.</p></div>' +
      '<p class="weight-equation">w espacial = σ espacial² / (σ espacial² + σ filogenética²) · w filogenética = 1 − w espacial</p>' +
      '<p class="scale-note">' + shareExplanation + '</p>' +
      '<div class="parameter-grid">' + comparisonPanel(varianceShare) + '</div>' +
    '</section>' +
    '<section class="cross-trait-section">' +
      '<div class="component-heading"><div><p class="component-symbol">Σₖₗ</p><h3>Asociación con otro rasgo</h3></div>' +
      '<p>R describe semejanza entre lenguas; Σ describe covariación entre rasgos. Selecciona otro rasgo para comparar cada componente.</p></div>' +
      '<label for="relatedFeature">Comparar ' + escapeHTML(feature.id) + ' con</label>' +
      '<select id="relatedFeature"></select><div id="correlationResults" class="correlation-grid"></div>' +
      '<p class="scale-note">Las correlaciones corresponden a los campos latentes; no son asociaciones causales entre las respuestas binarias.</p>' +
    '</section>' +
    '<section class="interpretation-note"><h3>Cómo leer las amplitudes</h3>' +
      '<p>σ espacial y σ filogenética son amplitudes por rasgo en la misma escala probit. Como R tiene diagonal uno, cada σ resume la desviación estándar marginal de su componente. El modelo no tiene un único coeficiente de mezcla geografía/filogenia: suma los dos efectos. Sus amplitudes no son porcentajes de importancia.</p></section>';

  const related = $("relatedFeature");
  related.replaceChildren(...peers.map((item) =>
    new Option(item.id + " · " + item.name, item.slug)
  ));
  if (peers.some((item) => item.slug === previousPeer)) related.value = previousPeer;
  related.disabled = peers.length === 0;
  function renderCorrelations() {
    const peer = peers.find((item) => item.slug === related.value);
    const results = $("correlationResults");
    if (!peer) {
      results.innerHTML = '<p class="empty-state">Este panel contiene un solo rasgo; no hay correlaciones entre rasgos que estimar.</p>';
      return;
    }
    const spatial = pairParameter(result, "spatial", slug, peer.slug);
    const phylogenetic = pairParameter(result, "phylogenetic", slug, peer.slug);
    results.innerHTML =
      comparisonPanel(spatial, "Correlación entre rasgos · espacial") +
      comparisonPanel(phylogenetic, "Correlación entre rasgos · filogenética");
  }
  related.onchange = renderCorrelations;
  renderCorrelations();

  const diagnostic = result.diagnostics;
  $("diagnostics").innerHTML =
    '<p class="diagnostic-status ' + (diagnostic.passed ? "ok" : "warning") + '">' +
    (diagnostic.passed
      ? "Diagnósticos dentro de los criterios configurados."
      : "Advertencia: la mezcla no cumple todos los criterios; interpreta estas distribuciones como exploratorias.") +
    '</p><div class="diagnostic-grid">' + [
      ["R-hat máximo", number(diagnostic.max_rhat)],
      ["ESS bulk mínimo", Math.round(diagnostic.min_ess_bulk)],
      ["ESS tail mínimo", Math.round(diagnostic.min_ess_tail)],
      ["Divergencias", diagnostic.divergences],
      ["Topes de profundidad", diagnostic.tree_depth_hits],
    ].map(([label, value]) => statCard(label, value)).join("") + '</div>';
}
function renderMetrics() {
  const features = data.features;
  const scored = [
    ["Exactitud", "accuracy"],
    ["Exactitud balanceada", "balanced_accuracy"],
    ["AUC", "roc_auc"],
    ["Precisión positiva", "precision_positive"],
    ["Recall positivo", "recall_positive"],
    ["F1 positivo", "f1_positive"],
  ];
  $("metricCards").innerHTML = features.map((feature) => {
    const entry = data.metrics[feature.slug];
    const metric = entry.metrics;
    const bars = scored.map(([label, key]) => `
      <div class="metric-row"><span>${escapeHTML(label)}</span>
        <div class="metric-track" role="img" aria-label="${escapeHTML(label)}: ${percent(metric[key])}">
          <div class="metric-fill" style="width:${Math.max(0, Math.min(100, 100 * metric[key]))}%"></div>
        </div><strong class="metric-value">${percent(metric[key])}</strong>
      </div>`).join("");
    return `<article class="metric-card">
      <h3>${escapeHTML(feature.id)} · ${escapeHTML(feature.name)}</h3>
      <p class="metric-subtitle">Clase positiva: ${escapeHTML(feature.positive_label)} · ${metric.n_test} lenguas en test</p>
      <div class="metric-highlight">
        ${statCard("Aciertos", `${metric.correct} / ${metric.n_test}`)}
        ${statCard("Exactitud", percent(metric.accuracy))}
        ${statCard("MCC", number(metric.mcc))}
      </div>
      ${bars}
    </article>`;
  }).join("");

  const columns = [
    ["Rasgo", null], ["Test", "n_test"], ["Exactitud", "accuracy"],
    ["Balanceada", "balanced_accuracy"], ["AUC", "roc_auc"],
    ["MCC", "mcc"], ["Brier ↓", "brier"], ["Log-loss ↓", "log_loss"],
  ];
  const rows = features.map((feature) => {
    const metric = data.metrics[feature.slug].metrics;
    return `<tr>${columns.map(([, key], index) => {
      const value = index === 0 ? `${feature.id} · ${feature.name}`
        : key === "n_test" ? metric[key] : number(metric[key]);
      return `<td>${escapeHTML(value)}</td>`;
    }).join("")}</tr>`;
  }).join("");
  $("lossTable").innerHTML = `<div class="table-wrap"><table><thead><tr>${columns.map(([label]) => `<th>${escapeHTML(label)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;

  const coverageRows = features.map((feature) => {
    const values = data.coverage[feature.slug];
    const observed = values.train + values.test;
    const cells = [
      `${feature.id} · ${feature.name}`, values.source_languages, values.eligible,
      `${observed} (${values.train} train)`, values.test, values.imputed,
      values.out_of_scope, values.excluded,
    ];
    return `<tr>${cells.map((value) => `<td>${escapeHTML(value)}</td>`).join("")}</tr>`;
  }).join("");
  $("coverageTable").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Rasgo</th><th>Lenguas en la fuente</th><th>Aptas para modelo</th><th>Observadas</th><th>Test</th><th>Imputadas</th><th>Fuera de alcance</th><th>Excluidas</th></tr></thead><tbody>${coverageRows}</tbody></table></div>`;

  const diagnostics = [data.stages.evaluation.diagnostics, data.stages.imputation.diagnostics];
  const hasWarning = diagnostics.some((item) => !item.passed);
  $("metricDiagnostics").innerHTML = `<p class="diagnostic-status ${hasWarning ? "warning" : "ok"}">${hasWarning ? "Advertencia de interpretación: al menos uno de los ajustes tiene mezcla insuficiente según sus diagnósticos NUTS." : "Ambos ajustes cumplen los criterios NUTS configurados."}</p>`;
}

function updateAddress(item) {
  try {
    const url = new URL(location.href);
    url.searchParams.set("stage", item.stage);
    url.searchParams.set("feature", item.feature);
    url.searchParams.set("language", item.id);
    history.replaceState(null, "", url);
  } catch (_) { /* El navegador puede restringir history con file://. */ }
}

function initialize() {
  $("feature").replaceChildren(...data.features.map((item) => new Option(`${item.id} · ${item.name}`, item.slug)));
  const query = new URLSearchParams(location.search);
  if (["evaluation", "imputation"].includes(query.get("stage"))) $("stage").value = query.get("stage");
  if (data.features.some((item) => item.slug === query.get("feature"))) $("feature").value = query.get("feature");
  const runWarning = $("runWarning");
  runWarning.className = data.status === "complete" ? "ok" : "warning";
  runWarning.textContent = data.status === "complete"
    ? "Ajustes completados con diagnósticos satisfactorios."
    : "La ejecución terminó con advertencias de diagnóstico. Consulta los parámetros del ajuste antes de interpretar las distribuciones.";

  $("predictionsTab").onclick = () => setTab("predictions");
  $("parametersTab").onclick = () => setTab("parameters");
  $("metricsTab").onclick = () => setTab("metrics");
  $("metricsFooter").onclick = () => setTab("metrics");
  $("stage").onchange = () => renderCase();
  $("feature").onchange = () => renderCase();
  $("search").oninput = () => renderCase();
  $("caseList").onchange = () => renderCase();
  $("parameterStage").onchange = renderParameters;
  renderCase(query.get("language"));
  renderParameters();
  if (location.hash === "#metrics") setTab("metrics");
}

initialize();
function initializeFeatureControls() {
  $("parameterFeature").replaceChildren(
    new Option("Selecciona un rasgo para ver sus componentes", ""),
    ...data.features.map((item) => new Option(item.id + " · " + item.name, item.slug))
  );
  const query = new URLSearchParams(location.search);
  if (["evaluation", "imputation"].includes(query.get("stage"))) {
    $("parameterStage").value = query.get("stage");
    $("stage").value = query.get("stage");
  }
  if (data.features.some((item) => item.slug === query.get("feature"))) {
    $("parameterFeature").value = query.get("feature");
  }

  $("parameterFeature").onchange = () => {
    if ($("parameterFeature").value) $("feature").value = $("parameterFeature").value;
    renderParameters();
    renderCase();
  };
  $("parameterStage").onchange = () => {
    $("stage").value = $("parameterStage").value;
    renderParameters();
    renderCase();
  };
  $("feature").onchange = () => {
    $("parameterFeature").value = $("feature").value;
    renderCase();
    renderParameters();
  };
  $("stage").onchange = () => {
    $("parameterStage").value = $("stage").value;
    renderCase();
    renderParameters();
  };

  renderParameters();
  if (location.hash === "#metrics") setTab("metrics");
  else if (location.hash === "#predictions") setTab("predictions");
  else setTab("parameters");
}

initializeFeatureControls();

