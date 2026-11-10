"""Construye una vista didáctica de predicciones y distribuciones posteriores."""

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from scipy.stats import halfnorm, lognorm, norm, uniform


def summarize_draws(values, edges=None, bins=20):
    """Resume muestras como histograma de masa, media e intervalo creíble."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("No hay muestras finitas para resumir")
    if edges is None:
        lower, upper = float(values.min()), float(values.max())
        if lower == upper:
            padding = max(abs(lower) * 0.05, 0.05)
            lower, upper = lower - padding, upper + padding
        edges = np.linspace(lower, upper, bins + 1)
    else:
        edges = np.asarray(edges, dtype=float)
    counts, edges = np.histogram(values, bins=edges)
    if counts.sum() != len(values):
        raise ValueError("Los intervalos no incluyen todas las muestras")
    return {
        "edges": edges.tolist(),
        "mass": (counts / len(values)).tolist(),
        "mean": float(values.mean()),
        "interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "n": int(values.size),
    }


def build_posterior_explorer(folder, run, predictions):
    """Genera posterior.html a partir de draws ya guardados, sin reajustar NUTS.

    Los draws se convierten en resúmenes compactos. La vista de modelo muestra
    solo el rasgo elegido y compara su posterior con la densidad teórica del
    prior correspondiente.
    """
    features = run["configuration"]["features"]
    predictions = predictions.copy()
    predictions["ID"] = predictions.ID.fillna("nan").astype(str)
    by_id = predictions.set_index("ID", drop=False)
    payload = {
        "dataset_label": run["configuration"].get("dataset_label", "Fuente de datos"),
        "features": features,
        "cases": [],
        "stages": {},
        "metrics": run["metrics_test"],
        "split": run["split"],
        "n_imputed": run["n_imputed"],
        "coverage": _coverage_summary(predictions, features),
        "status": run["status"],
        "inference": run["inference"],
        "phylogeny_normalized": bool(run["configuration"].get("normalize_phylogeny", False)),
    }

    for stage, archive_name, expected_role, diagnostics_key in [
        ("evaluation", "evaluation.npz", "test", "diagnostics_evaluation"),
        ("imputation", "imputation.npz", "imputed", "diagnostics_imputation"),
    ]:
        with np.load(folder / archive_name, allow_pickle=False) as archive:
            ids = archive["target_ids"].astype(str)
            target_mask = archive["target_mask"]
            draws = archive["probability_draws"]
            archive_features = archive["feature_ids"].astype(str).tolist()
            expected_features = [feature["id"] for feature in features]
            if archive_features != expected_features:
                raise ValueError(f"El orden de rasgos no coincide en {archive_name}")
            if draws.shape[1:] != target_mask.shape or target_mask.shape != (
                len(ids), len(features)
            ):
                raise ValueError(f"Dimensiones incompatibles en {archive_name}")

            stage_cases = 0
            for row_index, identifier in enumerate(ids):
                if identifier not in by_id.index:
                    raise ValueError(
                        f"Identificador ausente en predictions.csv: {identifier}"
                    )
                row = by_id.loc[identifier]
                for feature_index, feature in enumerate(features):
                    if not target_mask[row_index, feature_index]:
                        continue
                    slug = feature["slug"]
                    role = str(row[f"role_{slug}"])
                    if role != expected_role:
                        raise ValueError(
                            f"Etapa {stage}: {identifier}/{slug} tiene rol {role}, "
                            f"se esperaba {expected_role}"
                        )
                    probability_draws = np.clip(
                        draws[:, row_index, feature_index].astype(float), 0, 1
                    )
                    distribution = summarize_draws(
                        probability_draws, edges=np.linspace(0, 1, 21)
                    )
                    mean_column = float(row[f"p_{slug}"])
                    if not np.isclose(distribution["mean"], mean_column, atol=2e-5):
                        raise ValueError(
                            f"Media de draws no coincide para {identifier}/{slug}"
                        )
                    observed = str(row[f"observed_{slug}"])
                    predicted = str(row[f"predicted_{slug}"])
                    payload["cases"].append({
                        "id": (str(row["Glottocode"]) if identifier == "nan" else identifier),
                        "name": str(row["Name"]),
                        "family": str(row["Family"] or "Familia no documentada"),
                        "glottocode": str(row["Glottocode"]),
                        "stage": stage,
                        "feature": slug,
                        "observed": observed,
                        "predicted": predicted,
                        "status": str(row[f"status_{slug}"]),
                        "probability": distribution,
                        "correct": stage == "evaluation" and observed == predicted,
                    })
                    stage_cases += 1

            diagnostics = run[diagnostics_key]
            payload["stages"][stage] = {
                "diagnostics": diagnostics,
                "parameters": _parameter_distributions(
                    archive, features, run["inference"],
                    bool(run["configuration"].get("normalize_phylogeny", False))
                ),
                "n_cases": stage_cases,
            }

    output_path = folder / "posterior.html"
    _write_html(output_path, payload)
    return output_path


def _coverage_summary(predictions, features):
    """Cuenta la cobertura por rasgo y conserva visible cada estado del pipeline."""
    eligible = predictions["eligible"].astype(str).str.lower().eq("true")
    output = {}
    for feature in features:
        roles = predictions[f"role_{feature['slug']}"].astype(str)
        output[feature["slug"]] = {
            "source_languages": int(len(predictions)),
            "eligible": int(eligible.sum()),
            "train": int(roles.eq("train").sum()),
            "test": int(roles.eq("test").sum()),
            "imputed": int(roles.eq("imputed").sum()),
            "out_of_scope": int(roles.eq("out_of_scope").sum()),
            "excluded": int(roles.eq("excluded").sum()),
        }
    return output


def _parameter_distributions(archive, features, settings, phylogeny_normalized):
    """Resume parámetros globales, por rasgo y por pareja de rasgos."""
    parameters = []
    parameters.append(_parameter(
        archive["posterior_max_degree"],
        key="posterior_max_degree",
        title="Grado espacial máximo L",
        description=(
            "Parámetro global del ajuste. Determina cuánto detalle de la base "
            "esférica conserva el campo espacial."
        ),
        component="spatial",
        scope="global",
        prior={
            "kind": "uniform",
            "lower": settings["degree_min"],
            "upper": settings["degree_max"],
            "label": (
                f"Uniforme({settings['degree_min']:g}, "
                f"{settings['degree_max']:g})"
            ),
        },
    ))
    parameters.append(_parameter(
        archive["posterior_smoothing"],
        key="posterior_smoothing",
        title="Suavizado espacial λ",
        description=(
            "Parámetro global que atenúa patrones espaciales de frecuencia "
            "alta. No es un peso geografía-versus-filogenia."
        ),
        component="spatial",
        scope="global",
        prior={
            "kind": "lognormal",
            "median": settings["smoothing_median"],
            "log_sd": settings["smoothing_log_sd"],
            "label": (
                f"LogNormal(log({settings['smoothing_median']:g}), "
                f"{settings['smoothing_log_sd']:g})"
            ),
        },
    ))

    n_traits = len(features)
    concentration = settings["trait_correlation_concentration"]
    for index, feature in enumerate(features):
        slug = feature["slug"]
        parameters.append(_parameter(
            archive["posterior_intercept"][..., index],
            key=f"posterior_intercept_{slug}",
            title=f"Intercepto global α · {feature['id']}",
            description=(
                "Tendencia base de este rasgo en la escala probit, antes de "
                "sumar los efectos espaciales y filogenéticos."
            ),
            component="baseline",
            scope="feature",
            feature_slug=slug,
            prior={
                "kind": "normal",
                "location": 0.0,
                "scale": settings["intercept_scale"],
                "label": f"Normal(0, {settings['intercept_scale']:g})",
            },
        ))
        parameters.append(_parameter(
            archive["posterior_sigma_spatial"][..., index],
            key=f"posterior_sigma_spatial_{slug}",
            title=f"Amplitud espacial σ · {feature['id']}",
            description=(
                "Desviación estándar marginal del efecto espacial de este "
                "rasgo, en escala probit. Es una amplitud, no un porcentaje."
            ),
            component="spatial",
            scope="feature",
            feature_slug=slug,
            prior={
                "kind": "halfnormal",
                "scale": settings["amplitude_scale"],
                "label": f"HalfNormal({settings['amplitude_scale']:g})",
            },
        ))
        parameters.append(_parameter(
            archive["posterior_sigma_phylogenetic"][..., index],
            key=f"posterior_sigma_phylogenetic_{slug}",
            title=f"Amplitud filogenética σ · {feature['id']}",
            description=(
                "Desviación estándar marginal del efecto filogenético de este "
                "rasgo, en escala probit. Es una amplitud, no un porcentaje."
            ),
            component="phylogenetic",
            scope="feature",
            feature_slug=slug,
            prior={
                "kind": "halfnormal",
                "scale": settings["amplitude_scale"],
                "label": f"HalfNormal({settings['amplitude_scale']:g})",
            },
        ))

    share_title = (
        "Fracción de varianza latente espacial"
        if phylogeny_normalized else "Fracción cuadrática de escala espacial"
    )
    share_description = (
        "Resumen derivado por rasgo: σ espacial² / "
        "(σ espacial² + σ filogenética²). Como las matrices espaciales y "
        "filogenéticas de lenguas tienen diagonal uno en esta corrida, "
        "compara las varianzas marginales de los dos campos. No es un "
        "coeficiente ajustado aparte ni una medida específica para una lengua."
        if phylogeny_normalized else
        "Resumen derivado por rasgo: σ espacial² / "
        "(σ espacial² + σ filogenética²). La matriz filogenética no está "
        "normalizada a diagonal uno, por lo que esta razón compara escalas "
        "cuadráticas y no representa directamente la fracción de varianza "
        "marginal. No es un coeficiente ajustado aparte."
    )

    # Proporción derivada de las varianzas marginales de los dos campos.
    # En esta configuración ambas matrices de lenguas tienen diagonal uno y
    # los campos se suman independientemente: Var(g)=sigma_espacial^2 y
    # Var(h)=sigma_filogenética^2 en cada lengua.
    for index, feature in enumerate(features):
        spatial_sigma = archive["posterior_sigma_spatial"][..., index]
        phylogenetic_sigma = archive["posterior_sigma_phylogenetic"][..., index]
        denominator = spatial_sigma**2 + phylogenetic_sigma**2
        spatial_share = np.divide(
            spatial_sigma**2,
            denominator,
            out=np.full_like(denominator, 0.5, dtype=float),
            where=denominator > 0,
        )
        parameters.append(_parameter(
            spatial_share,
            key=f"derived_spatial_variance_share_{feature['slug']}",
            title=f"{share_title} · {feature['id']}",
            description=share_description,
            component="derived",
            scope="feature",
            feature_slug=feature["slug"],
            prior={
                "kind": "beta",
                "alpha": 0.5,
                "beta": 0.5,
                "label": "Derivada de dos HalfNormal independientes e iguales: Beta(½, ½)",
            },
        ))

    feature_pairs = [
        (first, second)
        for first in range(n_traits)
        for second in range(first + 1, n_traits)
    ]
    for key, component, values_name in [
        ("posterior_spatial_trait_correlations", "spatial",
         "Correlación entre rasgos · componente espacial"),
        ("posterior_phylogenetic_trait_correlations", "phylogenetic",
         "Correlación entre rasgos · componente filogenético"),
    ]:
        values = archive[key]
        if values.shape[-1] != len(feature_pairs):
            raise ValueError(f"Número inesperado de correlaciones en {key}")
        for pair_index, (first, second) in enumerate(feature_pairs):
            first_feature, second_feature = features[first], features[second]
            parameters.append(_parameter(
                values[..., pair_index],
                key=f"{key}_{first}_{second}",
                title=values_name,
                description=(
                    f"Asociación posterior entre {first_feature['name']} y "
                    f"{second_feature['name']}. Esta es una relación entre "
                    "rasgos dentro del componente indicado."
                ),
                component=component,
                scope="pair",
                pair_slugs=[first_feature["slug"], second_feature["slug"]],
                prior={
                    "kind": "lkj_correlation",
                    "dimension": n_traits,
                    "concentration": concentration,
                    "label": (
                        f"Marginal LKJ(K={n_traits}, "
                        f"concentración={concentration:g})"
                    ),
                },
            ))
    return parameters


def _parameter(values, *, key, title, description, component, scope,
               prior, feature_slug=None, pair_slugs=None):
    """Prepara densidades comparables para el prior teórico y los draws."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError(f"No hay draws finitos para {key}")

    prior_bounds = _prior_plot_bounds(prior)
    lower = min(float(values.min()), prior_bounds[0])
    upper = max(float(values.max()), prior_bounds[1])
    if not np.isfinite(lower + upper) or lower >= upper:
        padding = max(abs(lower) * 0.05, 0.05)
        lower, upper = lower - padding, upper + padding

    edges = np.linspace(lower, upper, 31)
    summary = summarize_draws(values, edges=edges)
    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    prior_interval = _prior_quantile(prior, [0.025, 0.975])
    return {
        "key": key,
        "title": title,
        "description": description,
        "component": component,
        "scope": scope,
        "feature_slug": feature_slug,
        "pair_slugs": pair_slugs,
        **summary,
        "posterior_density": (
            np.asarray(summary["mass"], dtype=float) / widths
        ).tolist(),
        "prior": {
            "label": prior["label"],
            "interval": np.asarray(prior_interval, dtype=float).tolist(),
            "density": _prior_density(prior, centers).tolist(),
        },
    }


def _prior_plot_bounds(prior):
    """Escoge un intervalo que incluya la masa central del prior."""
    kind = prior["kind"]
    if kind == "uniform":
        return float(prior["lower"]), float(prior["upper"])
    if kind == "lkj_correlation":
        return -1.0, 1.0
    if kind == "beta":
        return 0.0, 1.0
    if kind == "halfnormal":
        return 0.0, float(
            halfnorm.ppf(0.998, scale=prior["scale"])
        )
    quantiles = _prior_quantile(prior, [0.002, 0.998])
    return float(quantiles[0]), float(quantiles[1])


def _prior_quantile(prior, probabilities):
    """Calcula cuantiles de la distribución previa, incluida la marginal LKJ."""
    probabilities = np.asarray(probabilities, dtype=float)
    kind = prior["kind"]
    if kind == "normal":
        return norm.ppf(
            probabilities, loc=prior["location"], scale=prior["scale"]
        )
    if kind == "halfnormal":
        return halfnorm.ppf(probabilities, scale=prior["scale"])
    if kind == "uniform":
        return uniform.ppf(
            probabilities,
            loc=prior["lower"],
            scale=prior["upper"] - prior["lower"],
        )
    if kind == "lognormal":
        return lognorm.ppf(
            probabilities,
            s=prior["log_sd"],
            scale=prior["median"],
        )
    if kind == "lkj_correlation":
        shape = prior["concentration"] + (prior["dimension"] - 2) / 2
        return 2 * beta_distribution.ppf(
            probabilities, shape, shape
        ) - 1
    if kind == "beta":
        return beta_distribution.ppf(
            probabilities, prior["alpha"], prior["beta"]
        )
    raise ValueError(f"Distribución previa desconocida: {kind}")


def _prior_density(prior, values):
    """Evalúa la densidad previa exacta en la grilla común del gráfico."""
    values = np.asarray(values, dtype=float)
    kind = prior["kind"]
    if kind == "normal":
        return norm.pdf(
            values, loc=prior["location"], scale=prior["scale"]
        )
    if kind == "halfnormal":
        return halfnorm.pdf(values, scale=prior["scale"])
    if kind == "uniform":
        return uniform.pdf(
            values,
            loc=prior["lower"],
            scale=prior["upper"] - prior["lower"],
        )
    if kind == "lognormal":
        return lognorm.pdf(
            values,
            s=prior["log_sd"],
            scale=prior["median"],
        )
    if kind == "lkj_correlation":
        shape = prior["concentration"] + (prior["dimension"] - 2) / 2
        x = (values + 1) / 2
        density = beta_distribution.pdf(x, shape, shape) / 2
        return np.where(np.abs(values) <= 1, density, 0.0)
    if kind == "beta":
        return beta_distribution.pdf(values, prior["alpha"], prior["beta"])
    raise ValueError(f"Distribución previa desconocida: {kind}")


def _write_html(path: Path, payload):
    """Incrusta CSS, JavaScript y datos para que el HTML sea portable."""
    assets = Path(__file__).parent / "posterior_assets"
    html = (assets / "index.html").read_text(encoding="utf-8")
    html = html.replace("__STYLE__", (assets / "style.css").read_text(encoding="utf-8"))
    html = html.replace("__SCRIPT__", (assets / "app.js").read_text(encoding="utf-8"))
    html = html.replace(
        "__DATASET_LABEL__",
        escape(payload.get("dataset_label", "Fuente de datos")),
    )
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = html.replace("__DATA__", data.replace("<", "\\u003c"))
    path.write_text(html, encoding="utf-8")
