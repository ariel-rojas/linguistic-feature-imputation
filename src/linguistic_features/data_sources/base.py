"""Contrato mínimo de una fuente compatible con el flujo de inferencia."""

from abc import ABC, abstractmethod
from typing import TypeAlias

import pandas as pd

LoadedFeatures: TypeAlias = (
    tuple[pd.DataFrame, dict]
    | tuple[pd.DataFrame, dict, pd.DataFrame]
)


class FeatureSource(ABC):
    """Entrega lenguas, columnas 0/1 y metadatos en un formato común."""

    name = "fuente sin nombre"

    @abstractmethod
    def load(self, model_config: dict) -> LoadedFeatures:
        """Carga rasgos y devuelve tabla, auditoría y, si aplica, catálogo."""
