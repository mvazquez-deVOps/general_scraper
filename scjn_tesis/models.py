from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


SourceId = Literal["buscador_juridico", "semanario_judicial"]


@dataclass
class SearchParams:
    """Parámetros de búsqueda unificados."""

    texto: str | None = None
    numero_registro: str | None = None
    rubro: str | None = None

    def primary_query(self) -> str:
        if self.numero_registro:
            return self.numero_registro.strip()
        if self.rubro:
            return self.rubro.strip()
        if self.texto:
            return self.texto.strip()
        raise ValueError("Indica texto, numero_registro o rubro")


@dataclass
class TesisRecord:
    """Salida normalizada (ambas fuentes)."""

    source: SourceId
    numero_registro: str
    rubro: str
    texto_tesis: str
    organo_emisor: str
    epoca: str
    url_detalle: str
    url_listado: str = ""
    fuente: str = ""
    indice: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    scraped_at: str = ""

    def __post_init__(self) -> None:
        if not self.scraped_at:
            self.scraped_at = datetime.now(timezone.utc).isoformat()

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def save_json(path: str, records: list[TesisRecord], meta: dict[str, Any] | None = None) -> None:
    import json
    from pathlib import Path

    payload: dict[str, Any] = {
        "meta": meta or {},
        "count": len(records),
        "items": [r.to_json_dict() for r in records],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
