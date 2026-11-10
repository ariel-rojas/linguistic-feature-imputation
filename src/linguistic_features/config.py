"""Configuración, validación y rutas principales del proyecto."""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SOURCE = DATA / "source"
PROCESSED = DATA / "processed"
RESULTS_ROOT = ROOT / "results" / "current"


def dataset_results(dataset: str) -> Path:
    """Carpeta de salida independiente para cada juego de rasgos."""
    return RESULTS_ROOT / dataset


def read_json(path: Path) -> dict:
    """Lee un JSON UTF-8 y exige que contenga un objeto."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {path}")
    return value


def load_settings(model_path: str, inference_path: str) -> tuple[dict, dict]:
    """Carga y valida el catálogo de rasgos y los parámetros de inferencia."""
    model = read_json(resolve(model_path))
    inference = read_json(resolve(inference_path))
    if model.get("dataset") not in {"wals", "gbi_statistical"}:
        raise ValueError("dataset debe ser 'wals' o 'gbi_statistical'")
    features = model.get("features")
    if not isinstance(features, list) or len(features) < 2:
        raise ValueError("features debe contener al menos dos rasgos binarios")
    identifiers, slugs = set(), set()
    for feature in features:
        required_feature_fields = {
            "id", "slug", "name", "positive_label", "negative_label", "class_codes"
        }
        if not required_feature_fields <= set(feature):
            raise ValueError(
                f"Faltan campos para el rasgo: {sorted(required_feature_fields - set(feature))}"
            )
        if feature["id"] in identifiers or feature["slug"] in slugs:
            raise ValueError("Los identificadores y slugs de rasgos deben ser únicos")
        if not re.fullmatch(r"[A-Za-z0-9_]+", feature["slug"]):
            raise ValueError(f"Slug no válido: {feature['slug']}")
        identifiers.add(feature["id"])
        slugs.add(feature["slug"])
        class_codes = feature["class_codes"]
        if not isinstance(class_codes, dict) or set(class_codes.values()) != {0, 1}:
            raise ValueError(f"{feature['id']} debe incluir clases 0 y 1")
        if set(class_codes) & set(feature.get("excluded_codes", {})):
            raise ValueError(f"{feature['id']} repite códigos incluidos y excluidos")
        if model["dataset"] == "gbi_statistical" and not feature.get("source_column"):
            raise ValueError(
                f"El rasgo GBI {feature['id']} necesita indicar source_column"
            )
    if model.get("tree_source") != "mcc":
        raise ValueError("En esta versión solo está implementado tree_source='mcc'")
    if model.get("phylogenetic_structure") not in {"global", "within_family"}:
        raise ValueError("phylogenetic_structure debe ser global o within_family")
    if type(model.get("normalize_phylogeny")) is not bool:
        raise ValueError("normalize_phylogeny debe ser true o false")
    if not 0 < model.get("test_fraction", 0) < 1:
        raise ValueError("test_fraction debe estar entre 0 y 1")
    folds = round(1 / model["test_fraction"])
    if folds < 2 or not math.isclose(1 / folds, model["test_fraction"]):
        raise ValueError("test_fraction debe ser 1/k; 0.2 reserva un fold de cinco")

    required = {
        "chains", "warmup", "draws", "predictive_draws", "target_accept",
        "max_tree_depth", "amplitude_scale", "intercept_scale",
        "smoothing_median", "smoothing_log_sd", "degree_min", "degree_max",
        "degree_taper_width", "trait_correlation_concentration", "seed",
    }
    if set(inference) != required:
        raise ValueError(f"Claves requeridas en inferencia: {sorted(required)}")
    if inference["chains"] < 2 or inference["warmup"] < 1 or inference["draws"] < 4:
        raise ValueError("NUTS requiere al menos 2 cadenas, warmup y 4 draws")
    if not 0 < inference["target_accept"] < 1:
        raise ValueError("target_accept debe estar entre 0 y 1")
    if inference["trait_correlation_concentration"] <= 0:
        raise ValueError("trait_correlation_concentration debe ser positiva")
    if inference["degree_min"] >= inference["degree_max"]:
        raise ValueError("degree_min debe ser menor que degree_max")
    if inference["degree_max"] % 1:
        raise ValueError("degree_max debe ser entero para construir la base")
    if type(inference["seed"]) is not int or inference["seed"] < 0:
        raise ValueError("seed de inferencia debe ser un entero no negativo")
    for key, value in inference.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"{key} debe ser numérico")
    return model, inference


def resolve(filename: str) -> Path:
    """Interpreta rutas relativas desde la raíz del proyecto."""
    path = Path(filename)
    return path if path.is_absolute() else ROOT / path
