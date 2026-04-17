"""Descarga de binarios (PDF) vía Playwright request API (misma sesión/cookies que el navegador)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from playwright.sync_api import APIRequestContext

LogFn = Callable[[str], None]


def _log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def download_file(
    request_context: APIRequestContext,
    url: str,
    dest_path: Path,
    *,
    log: LogFn | None = None,
) -> Path | None:
    """
    GET binario y guarda en ``dest_path``. Crea directorios padre.
    Devuelve la ruta si OK, None si falla.
    """
    log = log or _log
    try:
        r = request_context.get(url, timeout=120_000)
        if not r.ok:
            log(f"HTTP {r.status} al descargar {url!r}")
            return None
        body = r.body()
        if not body:
            log(f"Cuerpo vacío: {url!r}")
            return None
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(body)
        log(f"Guardado {len(body)} bytes → {dest_path}")
        return dest_path
    except Exception as e:
        log(f"Error descarga {url!r}: {e!r}")
        return None


def is_pdf_url(url: str) -> bool:
    p = urlparse(url)
    return p.path.lower().endswith(".pdf") or ".pdf" in p.path.lower()


def safe_pdf_filename(url: str, registro: str, fallback: str = "doc") -> str:
    base = Path(urlparse(url).path).name
    if base and base.lower().endswith(".pdf"):
        return base
    return f"{registro}_{fallback}.pdf"
