"""Adaptador para la curación estadística binaria de Grambank (GBI)."""

import hashlib
from urllib.request import urlopen

import pandas as pd

from ..config import SOURCE
from .base import FeatureSource

GBI_RELEASE = "v1.0.0"
GBI_COMMIT = "77a8cb9"
GBI_FILES = {
    "statisticalGBI.csv": "curated_data/GBI/statisticalGBI/statisticalGBI.csv",
    "languages.csv": "curated_data/GBI/statisticalGBI/cldf/languages.csv",
    "parameters.csv": "curated_data/GBI/statisticalGBI/cldf/parameters.csv",
    "codes.csv": "curated_data/GBI/statisticalGBI/cldf/codes.csv",
    "modifications.csv": "curated_data/GBI/statisticalGBI/cldf/modifications.csv",
}


class GBIStatisticalSource(FeatureSource):
    """Carga respuestas 0/1 seleccionadas de la matriz GBI estadística."""

    name = "Grambank · curación estadística"

    def load(self, model_config):
        folder = ensure_gbi_source()
        matrix = pd.read_csv(folder / "statisticalGBI.csv", keep_default_na=False)
        languages = pd.read_csv(folder / "languages.csv", keep_default_na=False)
        parameters = pd.read_csv(folder / "parameters.csv", keep_default_na=False)
        if "glottocode" not in matrix or "glottocode" not in languages:
            raise ValueError("La matriz GBI necesita glottocode como identificador")
        if matrix.glottocode.duplicated().any() or languages.glottocode.duplicated().any():
            raise ValueError("GBI debe tener una fila única por glottocode")

        catalog = build_feature_catalog(matrix, parameters)
        selected = matrix.merge(
            languages[[
                "glottocode", "name", "isocodes", "latitude", "longitude"
            ]],
            on="glottocode", how="left", validate="one_to_one",
        )
        if selected["name"].isna().any():
            raise ValueError("Hay filas GBI sin metadata lingüística asociada")

        selected = selected.rename(columns={
            "glottocode": "Glottocode",
            "name": "Name",
            "isocodes": "ISO639P3code",
            "latitude": "Latitude",
            "longitude": "Longitude",
        })
        selected["ID"] = selected.Glottocode.astype(str)
        selected["Family"] = ""
        audit_features = {}
        for feature in model_config["features"]:
            feature_id = feature["id"]
            column = feature.get("source_column", feature_id)
            if column not in matrix.columns:
                raise ValueError(f"La columna {column} no está en la matriz GBI")
            values = pd.to_numeric(matrix[column], errors="coerce")
            observed_values = set(values.dropna().unique())
            if observed_values != {0, 1}:
                raise ValueError(
                    f"{column} no es una respuesta binaria 0/1: "
                    f"estados observados={sorted(observed_values)}"
                )
            slug = feature["slug"]
            selected[f"code_{slug}"] = values
            selected[f"y_{slug}"] = values.astype("Float64")
            selected[f"observed_{slug}"] = values.map({
                0: feature["negative_label"],
                1: feature["positive_label"],
            }).fillna("Faltante")
            metadata = catalog.loc[catalog.feature_id.eq(column)]
            description = (
                str(metadata.iloc[0].description)
                if not metadata.empty else feature.get("description", "")
            )
            audit_features[feature_id] = {
                "name": feature["name"],
                "description": description,
                "slug": slug,
                "source_column": column,
                "positive_label": feature["positive_label"],
                "negative_label": feature["negative_label"],
                "class_encoding": {"0": 0, "1": 1},
                "observed_all": selected[f"observed_{slug}"].value_counts().to_dict(),
                "n_observed": int(values.notna().sum()),
            }

        audit = {
            "dataset": "gbi_statistical",
            "label": model_config.get("dataset_label", self.name),
            "features": audit_features,
            "source": {
                "name": self.name,
                "release": GBI_RELEASE,
                "commit": GBI_COMMIT,
                "citation": (
                    "Graff et al. (2025), Scientific Data 12:106, "
                    "doi:10.1038/s41597-024-04319-4"
                ),
                "license": "CC BY 4.0",
                "url": "https://github.com/annagraff/crossling-curated",
                "sha256": {name: _sha256(folder / name) for name in GBI_FILES},
                "language_rows": int(len(matrix)),
                "feature_columns": int(len(catalog)),
                "binary_01_features": int(catalog.binary_01.sum()),
                "curation_note": (
                    "La curación estadística reduce dependencias fuertes entre rasgos "
                    "más allá de las atribuibles a genealogía y área."
                ),
            },
        }
        return selected, audit, catalog


def build_feature_catalog(matrix, parameters):
    """Describe cobertura, estados y dominio de cada columna del conjunto."""
    id_column = "new.name"
    metadata = parameters.set_index(id_column, drop=False)
    rows = []
    ignored = {"Unnamed: 0", "glottocode"}
    for column in [name for name in matrix.columns if name not in ignored]:
        values = pd.to_numeric(matrix[column], errors="coerce")
        states = set(values.dropna().unique())
        row = metadata.loc[column] if column in metadata.index else None
        rows.append({
            "feature_id": column,
            "description": str(row["description"]) if row is not None else "",
            "grouping": str(row["grouping"]) if row is not None else "",
            "n_observed": int(values.notna().sum()),
            "n_zero": int((values == 0).sum()),
            "n_one": int((values == 1).sum()),
            "states": ", ".join(str(value) for value in sorted(states, key=str)),
            "binary_01": states == {0, 1},
        })
    return pd.DataFrame(rows).sort_values("feature_id").reset_index(drop=True)


def ensure_gbi_source():
    """Descarga, si hace falta, solo los ficheros necesarios de la release fijada."""
    folder = SOURCE / "gbi"
    folder.mkdir(parents=True, exist_ok=True)
    for filename, relative_path in GBI_FILES.items():
        path = folder / filename
        if not path.exists():
            url = (
                "https://raw.githubusercontent.com/annagraff/crossling-curated/"
                f"{GBI_COMMIT}/{relative_path}"
            )
            print(f"Descargando GBI: {filename}", flush=True)
            with urlopen(url, timeout=180) as response:
                path.write_bytes(response.read())
    return folder


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
