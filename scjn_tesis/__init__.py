"""Scrapers del Buscador Jurídico (bj.scjn.gob.mx) y Semanario Judicial (sjf2.scjn.gob.mx)."""

from scjn_tesis.base import TesisSourceAdapter
from scjn_tesis.buscador_juridico import BuscadorJuridicoAdapter
from scjn_tesis.models import SearchParams, TesisRecord
from scjn_tesis.semanario import SemanarioAdapter

__all__ = [
    "TesisRecord",
    "SearchParams",
    "TesisSourceAdapter",
    "BuscadorJuridicoAdapter",
    "SemanarioAdapter",
]
