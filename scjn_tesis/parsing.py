"""Extrae campos desde texto libre (listados y detalle)."""

from __future__ import annotations

import re


def parse_organo_epoca_line(line: str) -> tuple[str, str]:
    """
    Ej.: 'SCJN;12a. época;Semanario Judicial...' o 'SCJN;  5a. Epoca;...'
    """
    line = line.strip()
    if not line:
        return "", ""
    parts = [p.strip() for p in line.split(";") if p.strip()]
    organo = parts[0] if parts else ""
    epoca = ""
    for p in parts[1:]:
        if re.search(r"\b[0-9]+a\.?\s*[éeE]poca", p, re.I) or "época" in p.lower() or "epoca" in p.lower():
            epoca = p
            break
    if not epoca and len(parts) > 1:
        epoca = parts[1]
    return organo, epoca


def extract_registro_digital(text: str) -> str | None:
    m = re.search(r"Registro\s+digital:\s*(\d+)", text, re.I)
    return m.group(1) if m else None


def trim_footer(text: str) -> str:
    """Quita bloques típicos de pie de página SCJN."""
    cut = re.split(r"\n\s*UBICACI[ÓO]N\s*\n", text, maxsplit=1, flags=re.I)
    return cut[0].strip() if cut else text.strip()
