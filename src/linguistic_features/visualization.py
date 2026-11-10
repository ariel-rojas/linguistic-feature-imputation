"""Mapas separados para la evaluación train/test y la imputación final."""

from html import escape

import folium
from folium.plugins import Fullscreen


def build_map(predictions, output_path, features, dataset_label=""):
    """Crea capas por rasgo y etapa para comparar evaluación e imputación.

    La capa de evaluación muestra las etiquetas observadas de train y las
    predicciones fuera de muestra para test. La capa de imputación final muestra
    las etiquetas conocidas y las probabilidades para los datos faltantes.
    """
    world = folium.Map(
        location=[15, 0], zoom_start=2, min_zoom=1,
        tiles="OpenStreetMap", control_scale=True,
    )
    Fullscreen(position="topleft").add_to(world)
    rows = predictions.loc[predictions.coordinates_valid]

    for feature_index, feature in enumerate(features):
        slug = feature["slug"]
        evaluation = folium.FeatureGroup(
            name=f"{feature['id']} · {feature['name']} · Split train/test",
            show=feature_index == 0,
        )
        imputation = folium.FeatureGroup(
            name=f"{feature['id']} · {feature['name']} · Imputación completa",
            show=False,
        )

        for row in rows.itertuples():
            role = getattr(row, f"role_{slug}")
            observed = getattr(row, f"observed_{slug}")
            predicted = getattr(row, f"predicted_{slug}")
            probability = getattr(row, f"p_{slug}")
            lower = getattr(row, f"p_{slug}_lower")
            upper = getattr(row, f"p_{slug}_upper")
            status = getattr(row, f"status_{slug}")
            name = escape(str(row.Name))
            identifier = escape(str(row.ID))
            family = escape(str(row.Family))
            observed_text = escape(str(observed))
            predicted_text = escape(str(predicted))
            positive = feature["positive_label"]
            negative = feature["negative_label"]

            if role in {"train", "test"}:
                if role == "train":
                    color = class_color(observed, positive, negative)
                    detail = (
                        f"Entrenamiento · dato observado: {observed_text}. "
                        "La categoría se muestra directamente; no es una predicción de test."
                    )
                    border, border_weight = "#263238", 1
                    link = ""
                else:
                    color = probability_color(float(probability))
                    is_correct = status == "Test: correcto"
                    border = "#147d3f" if is_correct else "#c62828"
                    border_weight = 3
                    detail = (
                        f"Test reservado · observado: {observed_text} · "
                        f"predicho: {predicted_text} · "
                        f"P({escape(positive)})={float(probability):.3f} "
                        f"[IC 95 % {float(lower):.3f}, {float(upper):.3f}] · "
                        f"{escape(status)}"
                    )
                    link = posterior_link("evaluation", slug, identifier)
                popup = language_popup(
                    name, feature, identifier, family, detail, link
                )
                folium.CircleMarker(
                    location=[float(row.Latitude), float(row.Longitude)],
                    radius=4 if role == "train" else 6,
                    color=border,
                    weight=border_weight,
                    fill=True,
                    fill_color=color,
                    fill_opacity=1.0,
                    tooltip=f"{name} · {escape(role)} · {observed_text}",
                    popup=folium.Popup(popup, max_width=390),
                ).add_to(evaluation)

            # Vista del modelo reajustado con todas las etiquetas observadas.
            if role in {"train", "test"}:
                color = class_color(observed, positive, negative)
                border, border_weight = "#263238", 1
                detail = (
                    f"Ajuste final · dato observado: {observed_text}. "
                    "El modelo usó todos los datos observados de este rasgo."
                )
                link = ""
            elif role == "imputed":
                color = probability_color(float(probability))
                border, border_weight = "#6a3d9a", 3
                detail = (
                    f"Imputación · predicción: {predicted_text} · "
                    f"P({escape(positive)})={float(probability):.3f} "
                    f"[IC 95 % {float(lower):.3f}, {float(upper):.3f}]"
                )
                link = posterior_link("imputation", slug, identifier)
            else:
                # Incluye categorías binarias excluidas y lenguas sin datos
                # suficientes para entrar al modelo, cuando tienen coordenadas.
                color, border, border_weight = "#a0a0a0", "#555555", 1
                detail = "Sin predicción: " + escape(str(status))
                link = ""

            popup = language_popup(name, feature, identifier, family, detail, link)
            folium.CircleMarker(
                location=[float(row.Latitude), float(row.Longitude)],
                radius=5 if role == "imputed" else 4,
                color=border,
                weight=border_weight,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
                tooltip=f"{name} · {escape(str(status))}",
                popup=folium.Popup(popup, max_width=390),
            ).add_to(imputation)

        evaluation.add_to(world)
        imputation.add_to(world)

    navigation = f"""
    <div style="position:fixed;left:64px;top:12px;z-index:9999;background:white;
      padding:8px 12px;border-radius:8px;box-shadow:0 1px 8px #777;
      display:flex;align-items:center;gap:14px;flex-wrap:wrap;font:13px Arial;line-height:1.5">
      <b>{escape(dataset_label or 'Explorar resultados')}</b>
      <a href="../index.html">Cambiar conjunto</a>
      <a href="posterior.html">Distribuciones y casos</a>
      <a href="posterior.html#metrics">Métricas y cobertura</a>
    </div>
    """
    legend = """
    <div style="position:fixed;left:18px;bottom:24px;z-index:9999;background:white;
      padding:12px 14px;border-radius:8px;box-shadow:0 1px 8px #777;
      font:13px Arial;max-width:330px;line-height:1.5">
      <b>Imputación bayesiana de rasgos lingüísticos</b><br>
      En el selector de capas, elige un rasgo y una etapa.<br>
      <b>Split train/test:</b> train muestra la clase observada; test muestra
      la probabilidad predicha y su acierto/error.<br>
      <b>Imputación completa:</b> incorpora todos los datos observados y las
      predicciones de valores faltantes.<br>
      <div style="height:12px;margin:6px 0;background:linear-gradient(90deg,#2166ac,#f7f7f7,#d95f02)"></div>
      Azul = probabilidad/clase 0 · naranja = probabilidad/clase 1.<br>
      Bordes verde/rojo = acierto/error en test; violeta = dato imputado;
      gris = excluido o sin predicción.<br>
      Para evitar puntos duplicados, deja visible una sola etapa por rasgo.
    </div>
    """
    world.get_root().html.add_child(folium.Element(navigation))
    world.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(world)
    world.save(str(output_path))


def language_popup(name, feature, identifier, family, detail, link):
    """Arma un popup HTML escapado y, si aplica, enlazado al caso posterior."""
    link_html = (
        f'<br><a href="{link}" target="_blank">Abrir distribución posterior</a>'
        if link else ""
    )
    return (
        f"<b>{name}</b><br>{escape(feature['id'])} · "
        f"{escape(feature['name'])}<br>Identificador: {identifier}<br>"
        f"Familia: {family}<br>{detail}{link_html}"
    )


def posterior_link(stage, feature_slug, identifier):
    """Devuelve una URL relativa para abrir el caso en el explorador posterior."""
    return (
        f"posterior.html?stage={escape(stage)}&feature={escape(feature_slug)}"
        f"&language={identifier}"
    )


def class_color(observed, positive_label, negative_label):
    """Color fijo de la etiqueta observada 0 o 1."""
    if observed == positive_label:
        return "#d95f02"
    if observed == negative_label:
        return "#2166ac"
    return "#a0a0a0"


def probability_color(probability):
    """Interpola azul, blanco y naranja para una probabilidad entre 0 y 1."""
    anchors = [(33, 102, 172), (247, 247, 247), (217, 95, 2)]
    value = min(1.0, max(0.0, probability))
    segment = 0 if value < 0.5 else 1
    fraction = value * 2 if segment == 0 else (value - 0.5) * 2
    rgb = tuple(
        round(anchors[segment][channel] * (1 - fraction)
              + anchors[segment + 1][channel] * fraction)
        for channel in range(3)
    )
    return "#" + "".join(f"{value:02x}" for value in rgb)
