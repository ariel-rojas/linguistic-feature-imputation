"""Preparación común y auditoría de las distintas fuentes de rasgos."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED
from .data_sources import source_for
from .data_sources.gbi import GBIStatisticalSource
from .data_sources.wals import WalsSource, iso_conflicts
from .phylogeny import MCCProvider, save_covariance_bundle


def prepare_data(model_config):
    """Normaliza una fuente, la alinea con el MCC y guarda su auditoría."""
    dataset = model_config["dataset"]
    output = PROCESSED / dataset
    output.mkdir(parents=True, exist_ok=True)
    source = source_for(dataset)
    loaded = source.load(model_config)
    frame, source_audit = loaded[:2]
    catalog = loaded[2] if len(loaded) > 2 else None

    for column in ("Latitude", "Longitude"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["coordinates_valid"] = (
        frame.Latitude.between(-90, 90) & frame.Longitude.between(-180, 180)
    )
    frame["iso_conflict"] = (
        iso_conflicts(frame) if isinstance(source, WalsSource)
        else pd.Series(False, index=frame.index, dtype=bool)
    )

    # Un mismo proveedor y release del árbol alimentan ambos conjuntos de rasgos.
    provider = MCCProvider(
        structure=model_config["phylogenetic_structure"],
        normalize=model_config["normalize_phylogeny"],
    )
    tree = next(iter(provider.iter_covariances(frame)))
    tip_index = {tip: index for index, tip in enumerate(tree.tip_ids)}
    family_index = dict(zip(tree.tip_ids, tree.family_ids))
    frame["tree_tip_index"] = (
        frame.Glottocode.map(tip_index).fillna(-1).astype(int)
    )
    frame["family_id"] = frame.Glottocode.map(family_index).fillna("")
    if "Family" not in frame:
        frame["Family"] = frame.family_id
    else:
        frame["Family"] = frame.Family.where(
            frame.Family.fillna("").astype(str).ne(""), frame.family_id
        )
    frame["eligible"] = (
        frame.coordinates_valid & frame.tree_tip_index.ge(0) & ~frame.iso_conflict
    )
    frame["exclusion_reason"] = [
        "; ".join(reason for condition, reason in [
            (not row.coordinates_valid, "sin coordenadas válidas"),
            (row.tree_tip_index < 0, "glottocódigo ausente en el MCC"),
            (row.iso_conflict, "conflicto de ISO entre WALS y LOW"),
        ] if condition)
        for row in frame.itertuples()
    ]

    for feature in model_config["features"]:
        slug = feature["slug"]
        observed = frame[f"observed_{slug}"]
        response = frame[f"y_{slug}"]
        source_audit["features"][feature["id"]]["eligible_by_class"] = (
            observed.loc[frame.eligible & response.notna()].value_counts().to_dict()
        )
        source_audit["features"][feature["id"]]["eligible_missing"] = int(
            (frame.eligible & response.isna() & observed.eq("Faltante")).sum()
        )

    frame.to_csv(output / "languages.csv", index=False)
    if catalog is not None:
        selected_ids = {feature.get("source_column", feature["id"])
                        for feature in model_config["features"]}
        catalog["selected_for_model"] = catalog.feature_id.isin(selected_ids)
        catalog.to_csv(output / "feature_catalog.csv", index=False)
    save_covariance_bundle(tree, output)

    audit = {
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "dataset": source_audit,
        "tree": tree.metadata,
        "n_source_languages": int(len(frame)),
        "eligible_languages": int(frame.eligible.sum()),
        "coordinates_valid": int(frame.coordinates_valid.sum()),
        "tree_overlap": int(frame.tree_tip_index.ge(0).sum()),
        "excluded": frame.loc[~frame.eligible, "exclusion_reason"]
        .value_counts().to_dict(),
        "duplicate_tree_tips": int(
            frame.loc[frame.eligible, "Glottocode"].duplicated().sum()
        ),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=True), flush=True)
    return frame, tree


def load_prepared_data(model_config):
    """Carga tabla, MCC y orden de hojas del conjunto indicado."""
    dataset = model_config["dataset"]
    folder = PROCESSED / dataset
    path = folder / "languages.csv"
    matrix_path = folder / "phylogenetic_covariance.npy"
    tips_path = folder / "phylogenetic_tips.csv"
    if not path.exists() or not matrix_path.exists() or not tips_path.exists():
        raise FileNotFoundError(
            f"Faltan datos preparados para {dataset}: ejecuta "
            f"`uv run lingfeat prepare --model-config config/{'model_gbi.json' if dataset == 'gbi_statistical' else 'model.json'}`"
        )
    frame = pd.read_csv(path, keep_default_na=False)
    for feature in model_config["features"]:
        column = f"y_{feature['slug']}"
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["tree_tip_index"] = pd.to_numeric(frame.tree_tip_index).astype(int)
    for column in ("eligible", "coordinates_valid", "iso_conflict"):
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    for column in ("Latitude", "Longitude"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    covariance = np.load(matrix_path, allow_pickle=False)
    tips = pd.read_csv(tips_path, keep_default_na=False)
    if covariance.shape != (len(tips), len(tips)):
        raise ValueError("Matriz filogenética y tabla de hojas no coinciden")
    return frame, covariance, tips
