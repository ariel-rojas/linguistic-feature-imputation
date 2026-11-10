"""Evaluación conjunta, ajuste final e imputación de varios rasgos binarios."""

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import PROCESSED, RESULTS_ROOT, dataset_results, load_settings
from .data import load_prepared_data, prepare_data
from .design import phylogenetic_design, spherical_basis
from .inference import (
    flatten_samples, probability_draws, sample_posterior,
    summarize_probability_draws,
)
from .report import binary_metrics, write_report


def run_pipeline(model_path="config/model.json", inference_path="config/inference.json"):
    """Evalúa cada rasgo, reajusta con todos los datos e imputa sus faltantes."""
    model_config, settings = load_settings(model_path, inference_path)
    features = model_config["features"]
    settings["n_traits"] = len(features)
    result_dir = dataset_results(model_config["dataset"])
    prepare_data(model_config)
    data, covariance, tips = load_prepared_data(model_config)
    result_dir.mkdir(parents=True, exist_ok=True)
    tip_indices = dict(zip(tips.Glottocode, tips.tip_index))
    if not np.array_equal(
        data.Glottocode.map(tip_indices).fillna(-1).astype(int), data.tree_tip_index
    ):
        raise ValueError("La tabla y la matriz no siguen el mismo orden de hojas")

    eligible = data.loc[data.eligible].reset_index(drop=True)
    eligible["basis_index"] = np.arange(len(eligible))
    labels = response_matrix(eligible, features)
    observed_mask = np.isfinite(labels)
    if not observed_mask.any(axis=0).all():
        raise ValueError("Cada rasgo necesita observaciones binarias para el ajuste")
    # Se reserva una partición por rasgo. Un glottocódigo queda entero en un
    # mismo lado para ese rasgo, aunque esa lengua pueda aportar otro rasgo.
    train_mask, test_mask = stratified_group_split(
        eligible, labels, features, model_config
    )

    basis, degrees = spherical_basis(
        eligible.Latitude.to_numpy(), eligible.Longitude.to_numpy(),
        int(settings["degree_max"]),
    )
    print("Etapa 1/2: evaluación conjunta con celdas de test reservadas", flush=True)
    evaluation, eval_diagnostics = fit_and_predict(
        eligible, labels, train_mask, test_mask, covariance,
        basis, degrees, settings, settings["seed"],
    )
    metrics = evaluate_features(labels, test_mask, evaluation, features)
    print(json.dumps(metrics, ensure_ascii=True), flush=True)

    imputable_mask = np.column_stack([
        eligible[f"observed_{feature['slug']}"]
        .eq("Faltante").to_numpy(dtype=bool)
        for feature in features
    ])
    # La evaluación y la imputación son dos ajustes distintos: esta segunda
    # corrida vuelve a estimar parámetros con todas las celdas observadas.
    print("Etapa 2/2: ajuste conjunto con todos los valores observados", flush=True)
    imputation, imputation_diagnostics = fit_and_predict(
        eligible, labels, observed_mask, imputable_mask, covariance,
        basis, degrees, settings, settings["seed"] + 100,
        return_fitted=True,
    )

    save_stage(
        result_dir / "evaluation.npz", evaluation, eval_diagnostics,
        eligible.iloc[np.flatnonzero(train_mask.any(axis=1))],
        eligible.iloc[np.flatnonzero(test_mask.any(axis=1))],
        features, settings,
    )
    save_stage(
        result_dir / "imputation.npz", imputation, imputation_diagnostics,
        eligible.iloc[np.flatnonzero(observed_mask.any(axis=1))],
        eligible.iloc[np.flatnonzero(imputable_mask.any(axis=1))],
        features, settings,
    )

    predictions = assemble_predictions(
        data, eligible, labels, observed_mask, train_mask, test_mask,
        evaluation, imputation, features,
    )
    predictions.to_csv(result_dir / "predictions.csv", index=False)
    status_ok = eval_diagnostics["passed"] and imputation_diagnostics["passed"]
    prepared_audit = json.loads(
        (PROCESSED / model_config["dataset"] / "audit.json").read_text(encoding="utf-8")
    )
    run = {
        "status": "complete" if status_ok else "diagnostics_warning",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "configuration": model_config,
        "dataset": prepared_audit["dataset"],
        "coverage": {
            "source_languages": prepared_audit["n_source_languages"],
            "eligible_languages": prepared_audit["eligible_languages"],
            "coordinates_valid": prepared_audit["coordinates_valid"],
            "tree_overlap": prepared_audit["tree_overlap"],
            "features_in_source": prepared_audit["dataset"]["source"].get(
                "feature_columns", len(features)
            ),
            "binary_features_in_source": prepared_audit["dataset"]["source"].get(
                "binary_01_features", len(features)
            ),
            "features_fit": len(features),
        },
        "inference": {key: value for key, value in settings.items() if key != "n_traits"},
        "model": {
            "likelihood": "Bernoulli probit por cada celda observada",
            "linear_predictor": "eta_ik = alpha_k + g_ik + h_ik",
            "probability": "P(y_ik=1) = Phi(eta_ik)",
            "features": [feature_summary(feature) for feature in features],
            "spatial_effect": "Campo de armónicos esféricos con grado máximo L inferido",
            "phylogenetic_effect": "Covarianza Browniana de tiempos compartidos del MCC",
            "trait_covariance": "Matrices LKJ distintas para los efectos espacial y filogenético",
        },
        "source": {
            "dataset": model_config["dataset"],
            "dataset_label": model_config.get("dataset_label", model_config["dataset"]),
            "files": prepared_audit["dataset"]["source"],
            "tree": "Global Language Tree Pipeline v2.0.0, STATE_1 MCC",
            "mcc_matrix_sha256": hashlib.sha256(
                (PROCESSED / model_config["dataset"] / "phylogenetic_covariance.npy").read_bytes()
            ).hexdigest(),
        },
        "versions": {
            name: version(name) for name in ["numpyro", "jax", "arviz", "numpy", "scipy"]
        },
        "split": {
            "method": "StratifiedGroupKFold; 5 folds; 1 fold reservado por rasgo",
            "group": "Glottocode",
            "by_feature": split_counts(train_mask, test_mask, features),
        },
        "metrics_test": metrics,
        "diagnostics_evaluation": eval_diagnostics,
        "diagnostics_imputation": imputation_diagnostics,
        "trait_correlations": named_trait_correlations(
            imputation_diagnostics["trait_correlations"], features
        ),
        "n_imputed": {
            feature["slug"]: int(imputable_mask[:, index].sum())
            for index, feature in enumerate(features)
        },
        "limitations": [
            "La ejecución usa un único árbol MCC y no integra incertidumbre entre árboles.",
            "El test aleatorio por glottocódigo no reemplaza una validación por familia o región.",
            "Las categorías excluidas de una codificación binaria no se imputan.",
            "La covarianza entre rasgos se estima por separado para los efectos espacial y filogenético.",
        ],
    }
    (result_dir / "run.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    from .visualization import build_map
    from .posterior import build_posterior_explorer
    from .site import build_dataset_site

    build_map(
        predictions, result_dir / "map.html", features,
        model_config.get("dataset_label", model_config["dataset"]),
    )
    build_posterior_explorer(result_dir, run, predictions)
    write_report(result_dir, run)
    build_dataset_site(RESULTS_ROOT)
    print(f"Finalizado: {result_dir}; estado={run['status']}", flush=True)


def response_matrix(frame, features):
    """Devuelve la matriz lengua × rasgo, usando NaN para no observado/excluido."""
    return np.column_stack([
        pd.to_numeric(frame[f"y_{feature['slug']}"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        for feature in features
    ])


def stratified_group_split(frame, labels, features, model_config):
    """Reserva celdas por rasgo, manteniendo juntos los alias de cada glottocódigo.

    Cada rasgo recibe su propio fold estratificado. El agrupamiento evita que
    dos filas del mismo taxón aparezcan a la vez en train y test para ese rasgo.
    Otras respuestas observadas de la misma lengua sí pueden informar el ajuste
    multirrespuesta; no es una validación donde se oculta la lengua completa.
    """
    n_folds = round(1 / model_config["test_fraction"])
    train_mask = np.zeros(labels.shape, dtype=bool)
    test_mask = np.zeros(labels.shape, dtype=bool)
    groups = frame.Glottocode.astype(str).to_numpy()
    for trait_index, feature in enumerate(features):
        positions = np.flatnonzero(np.isfinite(labels[:, trait_index]))
        observed = labels[positions, trait_index].astype(int)
        splitter = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True,
            random_state=model_config["seed"] + trait_index,
        )
        train_local, test_local = next(
            splitter.split(np.zeros(len(positions)), observed, groups[positions])
        )
        train_positions, test_positions = positions[train_local], positions[test_local]
        for current in (train_positions, test_positions):
            if set(labels[current, trait_index].astype(int)) != {0, 1}:
                raise ValueError(
                    f"El split de {feature['id']} no contiene ambas clases"
                )
        if set(groups[train_positions]) & set(groups[test_positions]):
            raise ValueError(f"Un glottocódigo de {feature['id']} cruza train/test")
        train_mask[train_positions, trait_index] = True
        test_mask[test_positions, trait_index] = True
    return train_mask, test_mask


def fit_and_predict(
    frame, labels, fit_mask, target_mask, covariance, basis, degrees,
    settings, seed, return_fitted=False,
):
    """Ajusta con celdas seleccionadas y predice las filas objetivo conjuntas."""
    # Se pasan todas las filas con al menos una etiqueta seleccionada; la máscara
    # por celda evita que faltantes y respuestas de test entren al likelihood.
    fit_positions = np.flatnonzero(fit_mask.any(axis=1))
    target_positions = np.flatnonzero(target_mask.any(axis=1))
    fit_rows = frame.iloc[fit_positions]
    target_rows = frame.iloc[target_positions]
    factor, conditional_mean, residual_factor, unique_tips = phylogenetic_design(
        covariance,
        fit_rows.tree_tip_index.to_numpy(dtype=int),
        target_rows.tree_tip_index.to_numpy(dtype=int),
    )
    fit_labels = np.nan_to_num(labels[fit_positions], nan=0).astype(int)
    samples, extras, diagnostics = sample_posterior(
        basis_for_rows(fit_rows, basis), degrees, factor, fit_labels,
        fit_mask[fit_positions], settings, seed,
    )
    # Se conservan draws espaciados de NUTS para propagar incertidumbre a las
    # probabilidades, sin guardar una predicción para cada estado de la cadena.
    selected = flatten_samples(samples, settings["predictive_draws"])
    target_draws = probability_draws(
        selected, basis_for_rows(target_rows, basis), degrees,
        conditional_mean, residual_factor, seed + 1, settings,
    )
    result = {
        **summarize_probability_draws(target_draws),
        "draws": target_draws,
        "samples": samples,
        "extras": extras,
        "fit_positions": fit_positions,
        "target_positions": target_positions,
        "fit_mask": fit_mask[fit_positions],
        "target_mask": target_mask[target_positions],
        "unique_training_tips": unique_tips,
        "diagnostics": diagnostics,
    }
    if return_fitted:
        fitted_draws = probability_draws(
            selected, basis_for_rows(fit_rows, basis), degrees,
            factor, np.empty((len(fit_rows), 0)), seed + 2, settings,
        )
        result["fitted"] = summarize_probability_draws(fitted_draws)
        result["fitted_positions"] = fit_positions
    return result, diagnostics


def basis_for_rows(rows, full_basis):
    """Selecciona filas de la base espacial según el índice WALS preparado."""
    if "basis_index" not in rows:
        raise ValueError("Falta la posición de cada lengua en la base espacial")
    return full_basis[rows.basis_index.to_numpy(dtype=int)]


def evaluate_features(labels, test_mask, evaluation, features):
    """Calcula métricas por rasgo usando solo sus celdas reservadas."""
    target_map = {
        position: local
        for local, position in enumerate(evaluation["target_positions"])
    }
    output = {}
    for trait_index, feature in enumerate(features):
        positions = np.flatnonzero(test_mask[:, trait_index])
        local = np.asarray([target_map[position] for position in positions], dtype=int)
        output[feature["slug"]] = {
            "feature_id": feature["id"],
            "name": feature["name"],
            "metrics": binary_metrics(
                labels[positions, trait_index].astype(int),
                evaluation["mean"][local, trait_index],
            ),
        }
    return output


def assemble_predictions(
    data, eligible, labels, observed_mask, train_mask, test_mask,
    evaluation, imputation, features,
):
    """Une estados y probabilidades de test, ajuste final e imputación por rasgo."""
    output = data.copy()
    eligible_positions = {
        language_id: position for position, language_id in enumerate(eligible.ID)
    }
    eval_local = {position: i for i, position in enumerate(evaluation["target_positions"])}
    imp_local = {position: i for i, position in enumerate(imputation["target_positions"])}
    fit_local = {position: i for i, position in enumerate(imputation["fitted_positions"])}

    for trait_index, feature in enumerate(features):
        slug = feature["slug"]
        observed_col = f"observed_{slug}"
        role_col, status_col = f"role_{slug}", f"status_{slug}"
        p_col = f"p_{slug}"
        lower_col, upper_col = f"{p_col}_lower", f"{p_col}_upper"
        predicted_col = f"predicted_{slug}"
        output[role_col] = "excluded"
        output[status_col] = output.exclusion_reason
        output[p_col] = np.nan
        output[lower_col] = np.nan
        output[upper_col] = np.nan
        output[predicted_col] = ""

        for row_index, row in output.iterrows():
            if not bool(row.eligible):
                continue
            position = eligible_positions[row.ID]
            observed_label = row[observed_col]
            if observed_label != "Faltante" and not observed_mask[position, trait_index]:
                output.at[row_index, role_col] = "out_of_scope"
                output.at[row_index, status_col] = f"{observed_label}: fuera de la codificación binaria"
                continue
            if observed_mask[position, trait_index]:
                output.at[row_index, role_col] = "train"
                output.at[row_index, status_col] = "Observado: ajuste final"
                local = fit_local[position]
                mean = imputation["fitted"]["mean"][local, trait_index]
                lower = imputation["fitted"]["lower"][local, trait_index]
                upper = imputation["fitted"]["upper"][local, trait_index]
            else:
                output.at[row_index, role_col] = "imputed"
                output.at[row_index, status_col] = "Faltante imputado"
                local = imp_local[position]
                mean = imputation["mean"][local, trait_index]
                lower = imputation["lower"][local, trait_index]
                upper = imputation["upper"][local, trait_index]

            if test_mask[position, trait_index]:
                output.at[row_index, role_col] = "test"
                local_eval = eval_local[position]
                mean = evaluation["mean"][local_eval, trait_index]
                lower = evaluation["lower"][local_eval, trait_index]
                upper = evaluation["upper"][local_eval, trait_index]
                actual = feature["positive_label"] if labels[position, trait_index] else feature["negative_label"]
                predicted = feature["positive_label"] if mean >= 0.5 else feature["negative_label"]
                output.at[row_index, status_col] = (
                    "Test: correcto" if predicted == actual else "Test: incorrecto"
                )
            output.at[row_index, p_col] = float(mean)
            output.at[row_index, lower_col] = float(lower)
            output.at[row_index, upper_col] = float(upper)
            output.at[row_index, predicted_col] = (
                feature["positive_label"] if mean >= 0.5 else feature["negative_label"]
            )
    return output


def save_stage(path, result, diagnostics, fit_rows, target_rows, features, settings):
    """Guarda draws, máscaras y glottocódigos alineados para reproducibilidad."""
    arrays = {"probability_draws": result["draws"].astype(np.float32)}
    arrays.update({"posterior_" + key: value for key, value in result["samples"].items()})
    arrays.update({"sampler_" + key: value for key, value in result["extras"].items()})
    arrays["fit_ids"] = fit_rows.ID.to_numpy(dtype=str)
    arrays["target_ids"] = target_rows.ID.to_numpy(dtype=str)
    arrays["fit_mask"] = result["fit_mask"]
    arrays["target_mask"] = result["target_mask"]
    arrays["feature_ids"] = np.asarray([feature["id"] for feature in features], dtype=str)
    arrays["settings_json"] = np.asarray(json.dumps(settings))
    arrays["diagnostics_json"] = np.asarray(json.dumps(diagnostics))
    np.savez_compressed(path, **arrays)


def feature_summary(feature):
    return {
        "id": feature["id"],
        "slug": feature["slug"],
        "name": feature["name"],
        "description": feature.get("description", ""),
        "source_column": feature.get("source_column"),
        "negative_label": feature["negative_label"],
        "positive_label": feature["positive_label"],
        "class_codes": feature["class_codes"],
        "excluded_codes": feature.get("excluded_codes", {}),
    }


def split_counts(train_mask, test_mask, features):
    return {
        feature["slug"]: {
            "train": int(train_mask[:, index].sum()),
            "test": int(test_mask[:, index].sum()),
        }
        for index, feature in enumerate(features)
    }


def named_trait_correlations(correlations, features):
    output = {}
    for effect, pairs in correlations.items():
        output[effect] = {}
        for pair, summary in pairs.items():
            first, second = map(int, pair.split("-"))
            key = f"{features[first]['slug']} × {features[second]['slug']}"
            output[effect][key] = summary
    return output
