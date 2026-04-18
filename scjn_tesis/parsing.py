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


def clean_legal_artifacts(text: str) -> str:
    """
    Normaliza basura de render (SCJN / Angular): NBSP, espacios múltiples, saltos reiterados.
    No altera el significado del fallo; solo mejora legibilidad y JSON.
    """
    if not text:
        return text
    t = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    t = re.sub(r"[\t\u2000-\u200a\ufeff]+", " ", t)
    t = re.sub(r" {2,}", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def trim_footer(text: str) -> str:
    """
    Quita menús y pie de sitio SCJN solo si el bloque aparece al final del documento.
    No hace un split global en la primera 'UBICACIÓN' (evita truncar criterios jurídicos).
    """
    t = text.rstrip()
    if not t:
        return t
    n = len(t)
    if n < 500:
        return t

    out = t
    # Sólo inspeccionar el tramo final (p. ej. menú, dirección, redes)
    start = int(n * 0.72)
    tail = out[start:]

    # Patrón: salto de párrafo real antes de un título de pie. El más arriba en el tramo gana.
    pats = (
        r"(?is)\n{2,}\s*UBICACI[ÓO]N\s*[\n\r]",
        r"(?is)\n{2,}\s*CONT[ÁA]CTANOS?\s*[\n\r]",
        r"(?is)\n{2,}\s*REDES\s+SOCIALES\s*[\n\r]",
    )
    best: int | None = None
    for pat in pats:
        m = re.search(pat, tail)
        if m and (best is None or m.start() < best):
            best = m.start()
    if best is not None:
        out = out[: start + best].rstrip()

    return out.strip()
