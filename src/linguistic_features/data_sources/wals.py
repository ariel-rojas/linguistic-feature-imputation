"""Adaptador de WALS CLDF para rasgos binarios seleccionados en model.json."""

import hashlib
from importlib.metadata import version
from urllib.request import urlopen

import low
import pandas as pd

from ..config import SOURCE
from .base import FeatureSource

WALS_REVISION = "81298aa26d01fdc11b769bcc7fa87ad9e43d7ccd"
WALS_FILES = ("languages.csv", "values.csv", "codes.csv")


class WalsSource(FeatureSource):
    """Une códigos WALS con lenguas y los recodifica como respuestas 0/1."""

    name = "WALS CLDF"

    def load(self, model_config):
        features = model_config["features"]
        folder = download_wals()
        languages = pd.read_csv(folder / "languages.csv", keep_default_na=False)
        values = pd.read_csv(folder / "values.csv", keep_default_na=False)
        codes = pd.read_csv(folder / "codes.csv", keep_default_na=False)
        _validate_feature_codes(codes, features)

        # Los registros taxonómicos no representan lenguas observadas.
        languages = languages.loc[
            ~languages.ID.str.startswith(("genus-", "family-"))
        ].copy()
        audit = {}
        for feature in features:
            feature_id, slug = feature["id"], feature["slug"]
            selected = values.loc[
                values.Parameter_ID.eq(feature_id),
                ["Language_ID", "Code_ID", "Source"],
            ].rename(columns={
                "Code_ID": f"code_{slug}",
                "Source": f"source_{slug}",
            })
            if selected.Language_ID.duplicated().any():
                raise ValueError(
                    f"WALS registra más de un valor {feature_id} para alguna lengua"
                )
            languages = languages.merge(
                selected, left_on="ID", right_on="Language_ID", how="left",
                validate="one_to_one",
            ).drop(columns="Language_ID")

            code_column = f"code_{slug}"
            labels = {
                code: feature["positive_label"] if value == 1
                else feature["negative_label"]
                for code, value in feature["class_codes"].items()
            }
            labels.update(feature.get("excluded_codes", {}))
            languages[f"observed_{slug}"] = (
                languages[code_column].map(labels).fillna("Faltante")
            )
            languages[f"y_{slug}"] = pd.to_numeric(
                languages[code_column].map(feature["class_codes"]),
                errors="coerce",
            ).astype("Float64")
            audit[feature_id] = {
                "name": feature["name"],
                "description": feature.get("description", ""),
                "slug": slug,
                "positive_label": feature["positive_label"],
                "negative_label": feature["negative_label"],
                "class_encoding": feature["class_codes"],
                "excluded_codes": feature.get("excluded_codes", {}),
                "observed_all": languages[f"observed_{slug}"].value_counts().to_dict(),
                "source_values": int(
                    languages[f"source_{slug}"].fillna("").ne("").sum()
                ),
            }

        # El adaptador publica nombres comunes para el resto del pipeline.
        languages["ISO639P3code"] = languages["ISO639P3code"].astype(str)
        return languages, {
            "dataset": "wals",
            "label": model_config.get("dataset_label", self.name),
            "features": audit,
            "source": {
                "name": self.name,
                "revision": WALS_REVISION,
                "url": f"https://github.com/cldf-datasets/wals/tree/{WALS_REVISION}/cldf",
                "sha256": {name: _sha256(folder / name) for name in WALS_FILES},
                "low_version": version("languages-of-the-world"),
            },
        }


def download_wals():
    """Descarga una revisión fijada de los tres ficheros CLDF que se necesitan."""
    folder = SOURCE / "wals"
    folder.mkdir(parents=True, exist_ok=True)
    for filename in WALS_FILES:
        path = folder / filename
        if not path.exists():
            url = (
                "https://raw.githubusercontent.com/cldf-datasets/wals/"
                f"{WALS_REVISION}/cldf/{filename}"
            )
            print(f"Descargando WALS: {filename}", flush=True)
            with urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
    return folder


def _validate_feature_codes(codes, features):
    """Asegura que las categorías configuradas existen en los códigos WALS."""
    published = set(codes.ID)
    for feature in features:
        included = set(feature["class_codes"])
        excluded = set(feature.get("excluded_codes", {}))
        configured = included | excluded
        if set(feature["class_codes"].values()) != {0, 1}:
            raise ValueError(f"{feature['id']} necesita clases 0 y 1")
        if included & excluded:
            raise ValueError(f"{feature['id']} repite categorías incluidas y excluidas")
        if not configured <= published:
            raise ValueError(
                f"Códigos WALS no publicados para {feature['id']}: "
                f"{sorted(configured - published)}"
            )
        if not all(code.startswith(feature["id"] + "-") for code in configured):
            raise ValueError(f"Hay códigos asignados al rasgo incorrecto: {feature['id']}")


def iso_conflicts(frame):
    """Marca discrepancias entre el ISO de WALS y el de LOW para un glottocódigo."""
    database = low.LanguagesOfTheWorld()
    by_glottocode = {
        language.glottocode: language
        for language in database.languages if language.glottocode
    }
    return pd.Series([
        bool(
            (language := by_glottocode.get(row.Glottocode))
            and row.ISO639P3code and language.part3
            and str(language.part3) != str(row.ISO639P3code)
        )
        for row in frame.itertuples()
    ], index=frame.index, dtype=bool)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
