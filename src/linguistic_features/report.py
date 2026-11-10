"""Métricas por rasgo y reporte de una ejecución multirrespuesta."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
    log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)


def _count(value):
    """Formatea cantidades con separador de miles español."""
    return f"{int(value):,}".replace(",", ".")


def binary_metrics(observed, probability_positive):
    """Calcula métricas de clase y probabilidad para una respuesta binaria."""
    observed = np.asarray(observed, dtype=int)
    probability_positive = np.clip(
        np.asarray(probability_positive, dtype=float), 1e-8, 1 - 1e-8
    )
    predicted = (probability_positive >= 0.5).astype(int)
    return {
        "n_test": int(len(observed)),
        "correct": int((predicted == observed).sum()),
        "accuracy": float(accuracy_score(observed, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
        "precision_positive": float(precision_score(observed, predicted, zero_division=0)),
        "recall_positive": float(recall_score(observed, predicted, zero_division=0)),
        "f1_positive": float(f1_score(observed, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(observed, predicted)),
        "roc_auc": float(roc_auc_score(observed, probability_positive)),
        "brier": float(brier_score_loss(observed, probability_positive)),
        "log_loss": float(log_loss(
            observed, np.column_stack([1 - probability_positive, probability_positive]),
            labels=[0, 1],
        )),
    }


def write_report(folder, run):
    """Escribe métricas, diagnósticos y cobertura para cada rasgo."""
    evaluation = run["diagnostics_evaluation"]
    imputation = run["diagnostics_imputation"]
    lines = [
        "# Informe de imputación multirrespuesta de rasgos lingüísticos",
        "",
        f"Conjunto: **{run['configuration'].get('dataset_label', run['configuration']['dataset'])}**.",
        "",
        f"Estado de diagnóstico: **{run['status']}**.",
        "",
        "Cada rasgo se codifica como respuesta binaria 0/1. El modelo estima "
        "probabilidades posteriores continuas y covarianzas espaciales y filogenéticas entre rasgos.",
        "",
        "## Rasgos y evaluación reservada",
        "",
        "El 20 % de las observaciones de cada rasgo se reservó mediante un split "
        "estratificado por clase y agrupado por glottocódigo.",
        "",
        "| Rasgo | Test | Aciertos | Exactitud | Balanceada | Precisión + | Recall + | F1 + | MCC | AUC | Brier | Log-loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for slug, entry in run["metrics_test"].items():
        metrics = entry["metrics"]
        lines.append(
            f"| {entry['feature_id']} · {entry['name']} | {metrics['n_test']} | "
            f"{metrics['correct']} | {metrics['accuracy']:.3f} | "
            f"{metrics['balanced_accuracy']:.3f} | "
            f"{metrics['precision_positive']:.3f} | {metrics['recall_positive']:.3f} | "
            f"{metrics['f1_positive']:.3f} | {metrics['mcc']:.3f} | "
            f"{metrics['roc_auc']:.3f} | {metrics['brier']:.3f} | "
            f"{metrics['log_loss']:.3f} |"
        )
    lines.extend([
        "",
        "Las métricas con el signo + corresponden a la clase positiva configurada para cada rasgo. "
        "Las métricas de etiqueta usan umbral 0,5. AUC, Brier y log-loss utilizan las probabilidades; "
        "Brier y log-loss más bajos indican mejores probabilidades.",
        "",
        "## Cobertura de la fuente",
        "",
        f"- Lenguas en la fuente: {_count(run['coverage']['source_languages'])}.",
        f"- Lenguas con coordenadas y presencia en el MCC: {_count(run['coverage']['eligible_languages'])}.",
        f"- Columnas binarias disponibles en la fuente: {_count(run['coverage']['binary_features_in_source'])}.",
        f"- Rasgos incluidos en este ajuste conjunto: {run['coverage']['features_fit']}.",
        "",
        "La matriz original puede incluir más rasgos que el panel ajustado. El panel se declara en la configuración; "
        "el informe no presenta el subconjunto como un ajuste de todas las columnas disponibles.",
        "",
        f"Esta corrida es un piloto computacional: {run['inference']['warmup']} pasos de calentamiento "
        f"y {run['inference']['draws']} draws por cadena. Interpreta las métricas como exploratorias "
        "hasta que la posterior mezcle adecuadamente.",
        "",
        "## Covariación posterior entre rasgos",
        "",
        "Las correlaciones resumen cómo covarían los campos latentes, condicionadas a los datos "
        "y al MCC. El intervalo muestra incertidumbre posterior; no es una asociación causal.",
        "",
        "| Componente | Par de rasgos | Correlación media | Intervalo creíble 95 % |",
        "|---|---|---:|---:|",
    ])
    for effect, pairs in run["trait_correlations"].items():
        for pair, summary in pairs.items():
            lines.append(
                f"| {effect} | {pair} | {summary['mean']:.3f} | "
                f"[{summary['interval_95'][0]:.3f}, {summary['interval_95'][1]:.3f}] |"
            )
    lines.extend([
        "",
        "## Diagnósticos NUTS",
        "",
        "| Ajuste | R-hat máximo | ESS bulk mínimo | ESS tail mínimo | Divergencias | Topes de profundidad | Estado |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| Evaluación | {evaluation['max_rhat']:.3f} | {evaluation['min_ess_bulk']:.1f} | "
        f"{evaluation['min_ess_tail']:.1f} | {evaluation['divergences']} | "
        f"{evaluation['tree_depth_hits']} | {'adecuado' if evaluation['passed'] else 'revisar'} |",
        f"| Imputación | {imputation['max_rhat']:.3f} | {imputation['min_ess_bulk']:.1f} | "
        f"{imputation['min_ess_tail']:.1f} | {imputation['divergences']} | "
        f"{imputation['tree_depth_hits']} | {'adecuado' if imputation['passed'] else 'revisar'} |",
        "",
        f"El grado espacial L del ajuste final tiene mediana "
        f"{imputation['degree_posterior']['median']:.2f} e intervalo creíble del 95 % "
        f"{imputation['degree_posterior']['interval_95'][0]:.2f}–"
        f"{imputation['degree_posterior']['interval_95'][1]:.2f}.",
        "",
        "El flujo marca como adecuado solo un ajuste con R-hat < 1,01, ESS bulk y tail ≥ 400, "
        "sin divergencias ni saturaciones de profundidad.",
        "",
        "## Imputaciones y archivos",
        "",
    ])
    for slug, count in run["n_imputed"].items():
        lines.append(f"- `{slug}`: {count} probabilidades para observaciones faltantes imputables.")
    lines.extend([
        "- `map.html`: mapa con capas de evaluación train/test e imputación final por rasgo.",
        "- `posterior.html`: distribuciones por caso, etapas y parámetros globales.",
        "- `predictions.csv`: etiquetas, roles, probabilidades e intervalos por lengua y rasgo.",
        "- `run.json`: configuración, fuentes, métricas y diagnósticos completos.",
        "- `evaluation.npz` y `imputation.npz`: draws posteriores y probabilidades por rasgo.",
        "",
        "La evaluación ajusta un modelo conjunto sin las celdas de test. El ajuste final vuelve a "
        "usar todas las respuestas observadas y predice los valores faltantes. El split no es una "
        "validación de familias completas y la incertidumbre corresponde a un único árbol MCC.",
    ])
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
