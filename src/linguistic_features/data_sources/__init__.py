"""Adaptadores que convierten fuentes distintas a una tabla común de rasgos."""

from .gbi import GBIStatisticalSource
from .wals import WalsSource


def source_for(dataset: str):
    """Devuelve el adaptador indicado en el archivo de configuración."""
    sources = {
        "wals": WalsSource,
        "gbi_statistical": GBIStatisticalSource,
    }
    try:
        return sources[dataset]()
    except KeyError as error:
        raise ValueError(
            f"Fuente de rasgos desconocida: {dataset}. "
            f"Opciones: {', '.join(sources)}"
        ) from error
