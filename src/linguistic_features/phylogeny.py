"""Fuentes de covarianza filogenética y adaptador para el árbol MCC.

La interfaz TreeCovarianceProvider permite sustituir o ampliar la fuente de
árboles sin cambiar el ajuste estadístico. Hoy MCCProvider entrega un solo árbol.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import low
import numpy as np
import pandas as pd
from Bio import Phylo

from .config import PROCESSED, SOURCE

TREE_FILENAME = "global-language-summary-tree-relabelled_cartoonised.tree.gz"
TREE_URL = (
    "https://github.com/rbouckaert/global-language-tree-pipeline/releases/"
    f"download/v2.0.0/{TREE_FILENAME}"
)
TREE_SHA256 = "35578af18d987e58172c88793b33e8ba60626404bdad246610cd94e73146cb46"


@dataclass(frozen=True)
class TreeCovariance:
    """Una matriz de lenguas alineada con una versión concreta del árbol."""

    tree_id: str
    tip_ids: tuple[str, ...]
    covariance: np.ndarray
    family_ids: tuple[str, ...]
    metadata: dict


class TreeCovarianceProvider(ABC):
    """Contrato para proveedores MCC o, más adelante, conjuntos de árboles."""

    @abstractmethod
    def iter_covariances(self, language_rows: pd.DataFrame):
        """Entrega matrices con orden y procedencia explícitos."""


class MCCProvider(TreeCovarianceProvider):
    """Construye la matriz Browniana de tiempos compartidos para el MCC actual."""

    def iter_covariances(self, language_rows: pd.DataFrame):
        tree_path = _obtain_tree()
        tree = read_published_tree(tree_path)
        tree_tips = {tip.name for tip in tree.get_terminals()}
        names = tuple(sorted(set(language_rows.Glottocode) & tree_tips))
        if not names:
            raise ValueError("No hay glottocódigos WALS presentes en el MCC")
        raw = shared_branch_covariance(tree, names)
        family_map = lowest_family_by_glottocode()
        families = tuple(family_map.get(name, f"tip:{name}") for name in names)

        structure = self.structure
        covariance = normalize_correlation(raw) if self.normalize else raw
        if structure == "within_family":
            # La máscara deja covarianzas solo dentro de una familia. El
            # resultado es bloque-diagonal, no una inversión de la matriz.
            labels = np.asarray(families, dtype=str)
            covariance = covariance * (labels[:, None] == labels[None, :])
        _validate_covariance(covariance)
        yield TreeCovariance(
            tree_id="STATE_1_MCC",
            tip_ids=names,
            covariance=covariance,
            family_ids=families,
            metadata={
                "provider": "MCCProvider",
                "release": "v2.0.0",
                "url": TREE_URL,
                "sha256": TREE_SHA256,
                "tree_state": "STATE_1",
                "branch_units": "thousand_years",
                "base_covariance": "Brownian shared branch time",
                "structure": structure,
                "normalized": self.normalize,
                "n_tree_tips": len(tree_tips),
                "n_aligned_tips": len(names),
            },
        )

    def __init__(self, structure="within_family", normalize=True):
        self.structure = structure
        self.normalize = normalize


def _obtain_tree() -> Path:
    """Reutiliza el árbol descargado o baja la release con hash fijado."""
    path = SOURCE / "phylogeny" / TREE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print("Descargando árbol MCC de Global Language Tree Pipeline…", flush=True)
        with urlopen(TREE_URL, timeout=180) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != TREE_SHA256:
        raise ValueError(f"Hash MCC inesperado: {digest}")
    return path


def read_published_tree(path: Path):
    """Lee STATE_1 y verifica las dos copias MCC del NEXUS publicado."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        text = stream.read()
    clean = re.sub(r"\[[^\[\]]*\]", "", text)
    translated = re.search(r"(?is)\btranslate\s+(.*?);", clean)
    trees = dict(re.findall(r"(?i)\btree\s+(\w+)\s*=\s*([^;]+);", clean))
    if translated is None or set(trees) != {"STATE_0", "STATE_1"}:
        raise ValueError("La estructura NEXUS no coincide con la release fijada")
    if trees["STATE_0"] != trees["STATE_1"]:
        raise ValueError("STATE_0 y STATE_1 dejaron de ser copias del MCC")
    translations = dict(re.findall(r"(\d+)\s+([^,\s]+)", translated.group(1)))
    tree = Phylo.read(io.StringIO(trees["STATE_1"] + ";"), "newick")
    tree.name, tree.rooted = "STATE_1_MCC", True
    for tip in tree.get_terminals():
        label = translations.get(tip.name, "")
        if not re.match(r"^[a-z]{4}[0-9]{4}_", label):
            raise ValueError(f"La hoja {tip.name} no tiene glottocode reconocible")
        tip.name = label[:8]
    if len(tree.get_terminals()) != 6636:
        raise ValueError("La release MCC debería tener 6636 hojas")
    return tree


def shared_branch_covariance(tree, names):
    """Covarianza Browniana: suma de ramas compartidas por cada par de hojas."""
    all_tips = [tip.name for tip in tree.get_terminals()]
    if len(set(all_tips)) != len(all_tips) or not set(names) <= set(all_tips):
        raise ValueError("Las hojas del árbol deben ser únicas y contener los datos")
    index = {name: position for position, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=float)
    descendants = {}
    for node in tree.find_clades(order="postorder"):
        if node is tree.root:
            continue
        length = node.branch_length
        if length is None or not np.isfinite(length) or length < 0:
            raise ValueError("El árbol tiene longitudes de rama no válidas")
        if node.is_terminal():
            selected = [index[node.name]] if node.name in index else []
        else:
            selected = [i for child in node.clades for i in descendants.pop(child)]
        descendants[node] = selected
        # La longitud de esta rama se comparte entre todos sus descendientes;
        # por eso se suma a cada par de hojas que pasa por el nodo.
        matrix[np.ix_(selected, selected)] += length
    return matrix


def lowest_family_by_glottocode():
    """Usa la clasificación LOW para obtener IDs de familia estables."""
    database = low.LanguagesOfTheWorld()
    mapping = {}
    for language in database.languages:
        if not language.glottocode:
            continue
        node = language.family
        seen = set()
        ancestors = []
        while node:
            if node.glottocode in seen:
                raise ValueError("Ciclo en la jerarquía de familias LOW")
            seen.add(node.glottocode)
            ancestors.append(node)
            node = node.parent
        if ancestors:
            mapping[language.glottocode] = "family:" + ancestors[-1].glottocode
    return mapping


def normalize_correlation(covariance):
    """Convierte Cij en correlación Cij/sqrt(Cii*Cjj), sin invertir C."""
    scale = np.sqrt(np.diag(covariance))
    if np.any(scale <= 0):
        raise ValueError("La covarianza debe tener diagonal positiva")
    result = covariance / scale[:, None] / scale[None, :]
    np.fill_diagonal(result, 1.0)
    return result


def _validate_covariance(matrix):
    """Comprueba simetría, diagonal y semidefinición positiva numérica."""
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        raise ValueError("La matriz filogenética debe ser simétrica y finita")
    if np.any(np.diag(matrix) <= 0):
        raise ValueError("La matriz filogenética debe tener diagonal positiva")
    smallest = np.linalg.eigvalsh(matrix)[0]
    if smallest < -1e-8:
        raise ValueError(f"La matriz no es semidefinida positiva ({smallest:.3g})")


def save_covariance_bundle(tree: TreeCovariance, output_folder: Path | None = None):
    """Guarda matriz, orden de glottocódigos y auditoría con la misma versión."""
    output_folder = output_folder or PROCESSED
    output_folder.mkdir(parents=True, exist_ok=True)
    np.save(output_folder / "phylogenetic_covariance.npy", tree.covariance)
    pd.DataFrame({"tip_index": range(len(tree.tip_ids)), "Glottocode": tree.tip_ids,
                  "family_id": tree.family_ids}).to_csv(
        output_folder / "phylogenetic_tips.csv", index=False
    )

