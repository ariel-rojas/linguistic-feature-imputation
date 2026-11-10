"""Portal estático para alternar entre resultados de distintas fuentes."""

import html
import json
from pathlib import Path


def build_dataset_site(results_root: Path):
    """Crea ``results/current/index.html`` con selector de fuente y visualización."""
    datasets = []
    order = {"wals": 0, "gbi_statistical": 1}
    folders = sorted(
        (path for path in results_root.iterdir() if path.is_dir()),
        key=lambda path: (order.get(path.name, 10), path.name),
    )
    for folder in folders:
        run_path = folder / "run.json"
        if not run_path.exists() or not (folder / "map.html").exists():
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        datasets.append({
            "id": folder.name,
            "label": run["configuration"].get("dataset_label", folder.name),
            "languages": run["coverage"]["source_languages"],
            "eligible": run["coverage"]["eligible_languages"],
            "source_features": run["coverage"]["binary_features_in_source"],
            "fit_features": run["coverage"]["features_fit"],
            "status": run["status"],
            "source_url": run["dataset"]["source"].get("url", ""),
            "citation": run["dataset"]["source"].get("citation", ""),
            "features": [
                f"{feature['id']} · {feature['name']}"
                for feature in run["configuration"]["features"]
            ],
        })
    if not datasets:
        return None

    payload = json.dumps(datasets, ensure_ascii=True).replace("</", "<\\/")
    options = "\n".join(
        f'<option value="{html.escape(item["id"])}">'
        f'{html.escape(item["label"])}</option>'
        for item in datasets
    )
    default_id = html.escape(datasets[0]["id"])
    page = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Imputación de rasgos lingüísticos</title>
  <style>
    :root {{ color-scheme: light; font-family: Segoe UI, Arial, sans-serif; color: #172b3a; background: #f3f6f8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    header {{ background: #15354b; color: white; padding: 22px clamp(16px, 4vw, 52px); }}
    h1 {{ margin: 0 0 7px; font-size: clamp(22px, 3vw, 30px); }}
    header p {{ margin: 0; max-width: 900px; line-height: 1.5; color: #dbe7ec; }}
    main {{ padding: 18px clamp(12px, 3vw, 36px) 28px; }}
    .toolbar {{ display: flex; align-items: end; flex-wrap: wrap; gap: 12px; background: white; padding: 14px 16px; border-radius: 12px 12px 0 0; border: 1px solid #d7e0e5; }}
    label {{ display: grid; gap: 5px; font-size: 13px; font-weight: 650; }}
    select, button {{ min-height: 40px; border: 1px solid #9eafb9; border-radius: 7px; background: white; color: #18394d; font: inherit; padding: 7px 11px; }}
    select {{ min-width: min(360px, 80vw); }}
    button {{ cursor: pointer; font-weight: 650; }}
    button[aria-pressed="true"] {{ background: #176b87; color: white; border-color: #176b87; }}
    .details {{ display: flex; flex-wrap: wrap; gap: 9px 22px; align-items: center; padding: 10px 16px; background: #e8f0f4; border-inline: 1px solid #d7e0e5; font-size: 13px; }}
    .details strong {{ color: #123d55; }}
    .features {{ width: 100%; color: #435965; }}
    .frame {{ height: min(78vh, 900px); min-height: 620px; border: 1px solid #d7e0e5; border-top: 0; border-radius: 0 0 12px 12px; overflow: hidden; background: white; }}
    iframe {{ display: block; width: 100%; height: 100%; border: 0; }}
    .view-buttons {{ display: flex; gap: 8px; }}
    .note {{ margin: 12px 2px 0; max-width: 1100px; color: #50636c; font-size: 12px; line-height: 1.5; }}
    @media (max-width: 650px) {{ .frame {{ min-height: 72vh; height: 72vh; }} .toolbar {{ align-items: stretch; }} .view-buttons {{ width: 100%; }} .view-buttons button {{ flex: 1; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Imputación bayesiana de rasgos lingüísticos</h1>
    <p>Explora mapas, distribuciones posteriores, métricas y cobertura. Cambia de fuente para comparar el panel de WALS con el panel binario seleccionado de Grambank.</p>
  </header>
  <main>
    <div class="toolbar">
      <label>Conjunto de datos<select id="datasetSelect">{options}</select></label>
      <div class="view-buttons" aria-label="Tipo de visualización">
        <button id="mapButton" type="button" aria-pressed="true">Mapa</button>
        <button id="posteriorButton" type="button" aria-pressed="false">Distribuciones y métricas</button>
      </div>
    </div>
    <div id="datasetDetails" class="details" aria-live="polite"></div>
    <div class="frame"><iframe id="datasetView" title="Visualización del conjunto seleccionado" src="{default_id}/map.html"></iframe></div>
    <p class="note">El selector muestra los resultados que existen en esta carpeta. En el conjunto de Grambank se importan y auditan todos los rasgos binarios disponibles, mientras el ajuste inicial utiliza el pequeño panel indicado arriba por coste computacional del muestreo conjunto.</p>
  </main>
  <script>
    const datasets = {payload};
    const datasetSelect = document.getElementById('datasetSelect');
    const frame = document.getElementById('datasetView');
    const mapButton = document.getElementById('mapButton');
    const posteriorButton = document.getElementById('posteriorButton');
    let currentView = 'map';
    function renderDataset() {{
      const item = datasets.find(value => value.id === datasetSelect.value);
      if (!item) return;
      frame.src = `${{item.id}}/${{currentView === 'map' ? 'map.html' : 'posterior.html'}}`;
      const status = item.status === 'complete' ? 'Diagnósticos: adecuados' : 'Diagnósticos: revisar';
      const sourceLink = item.source_url ? `<a href="${{item.source_url}}" target="_blank" rel="noopener">Consultar fuente</a>` : '';
      document.getElementById('datasetDetails').innerHTML = `<strong>${{item.label}}</strong><span>${{item.languages.toLocaleString('es')}} lenguas en fuente</span><span>${{item.eligible.toLocaleString('es')}} con coordenadas y árbol</span><span>${{item.source_features.toLocaleString('es')}} rasgos binarios disponibles</span><span>${{item.fit_features}} rasgos incluidos en este ajuste</span><span>${{status}}</span>${{sourceLink}}<div class="features"><b>Panel ajustado:</b> ${{item.features.join(' · ')}}</div>`;
    }}
    function setView(view) {{
      currentView = view;
      mapButton.setAttribute('aria-pressed', String(view === 'map'));
      posteriorButton.setAttribute('aria-pressed', String(view === 'posterior'));
      renderDataset();
    }}
    datasetSelect.addEventListener('change', renderDataset);
    mapButton.addEventListener('click', () => setView('map'));
    posteriorButton.addEventListener('click', () => setView('posterior'));
    renderDataset();
  </script>
</body>
</html>
"""
    output = results_root / "index.html"
    output.write_text(page, encoding="utf-8")
    return output
