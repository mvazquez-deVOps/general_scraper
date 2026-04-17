"""
Buscador Jurídico — https://bj.scjn.gob.mx/

Selectores probados (Angular):
- Tarjetas de resultado: div.card.mb-1 (contiene "Registro digital:")
- Paginación URL: &page=N
- Detalle: /documento/{tipo}/{registro} según índice
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

from playwright.sync_api import Page, sync_playwright

from scjn_tesis.bj_urls import (
    bj_busqueda_url,
    bj_documento_path,
    parse_bj_busqueda_url,
)
from scjn_tesis.browser import launch_browser, new_context, settle_page
from scjn_tesis.downloads import download_file, is_pdf_url, safe_pdf_filename
from scjn_tesis.models import SearchParams, TesisRecord
from scjn_tesis.parsing import extract_registro_digital, parse_organo_epoca_line, trim_footer

LogFn = Callable[[str], None]
ProgressFn = Callable[[str, str, int], None]


def _default_log(msg: str) -> None:
    print(f"[bj] {msg}", flush=True)


def _parse_bj_card_text(card_text: str) -> dict[str, str]:
    reg = extract_registro_digital(card_text) or ""
    lines = [ln.strip() for ln in card_text.splitlines() if ln.strip()]
    rubro = ""
    meta_line = ""
    for i, ln in enumerate(lines):
        if "Registro digital" in ln:
            if i + 1 < len(lines) and not lines[i + 1].lower().startswith("scjn"):
                rubro = lines[i + 1]
            continue
        if ln.lower().startswith("scjn") or "; " in ln and "epoca" in ln.lower():
            meta_line = ln
            break
    if not rubro:
        for ln in lines:
            if ln != f"Registro digital: {reg}" and not ln.lower().startswith("publicación") and len(ln) > 10:
                if "scjn" not in ln.lower()[:8]:
                    rubro = ln
                    break
    org, ep = parse_organo_epoca_line(meta_line) if meta_line else ("", "")
    return {
        "numero_registro": reg,
        "rubro": rubro,
        "organo_emisor": org,
        "epoca": ep,
        "resumen_lista": card_text.strip(),
    }


def _collect_pdf_links(page: Page) -> list[str]:
    hrefs: list[str] = []
    try:
        for el in page.locator('a[href*="pdf"], a[href*=".pdf"]').all():
            h = el.get_attribute("href")
            if h and is_pdf_url(h):
                if h.startswith("/"):
                    h = "https://bj.scjn.gob.mx" + h
                hrefs.append(h)
    except Exception:
        pass
    return list(dict.fromkeys(hrefs))


def _fetch_bj_detail(
    page: Page,
    api_request,
    registro: str,
    indice: str,
    headless: bool,
    log: LogFn,
    pdfs_dir: Path | None,
) -> tuple[str, list[str]]:
    """
    Carga detalle del documento; si hay PDFs enlazados o respuesta PDF, descarga a pdfs_dir.
    """
    path_rel = bj_documento_path(indice, registro)
    url = f"https://bj.scjn.gob.mx{path_rel}"
    log(f"Detalle BJ: {url}")
    saved_pdfs: list[str] = []

    try:
        r0 = api_request.get(url, timeout=120_000)
        if r0.ok:
            ct0 = (r0.headers.get("content-type") or "").lower()
            if "pdf" in ct0 and pdfs_dir and r0.body():
                name = safe_pdf_filename(url, registro, "detalle")
                pth = pdfs_dir / name
                pth.parent.mkdir(parents=True, exist_ok=True)
                pth.write_bytes(r0.body())
                log(f"PDF guardado → {pth}")
                saved_pdfs.append(str(pth))
                return "[PDF descargado como binario; ver almacen/pdfs/]", saved_pdfs
    except Exception as e:
        log(f"GET detalle (probe): {e!r}")

    page.goto(url, wait_until="load")
    settle_page(page, headless=headless)

    body = page.locator("body").inner_text(timeout=60_000)
    body = trim_footer(body)

    if pdfs_dir:
        for href in _collect_pdf_links(page):
            name = safe_pdf_filename(href, registro, "enlace")
            out = download_file(api_request, href, pdfs_dir / name, log=log)
            if out:
                saved_pdfs.append(str(out))

    m = re.search(
        r"(?is)(Instancia:.*?)(?:UBICACI[ÓO]N|CONT[ÁA]CTANOS|REDES SOCIALES|$)",
        body,
    )
    texto = m.group(1).strip() if m else body.strip()
    if saved_pdfs:
        texto = texto + "\n\n[PDFs adjuntos: " + ", ".join(saved_pdfs) + "]"
    return texto, saved_pdfs


def scrape_buscador_juridico(
    params: SearchParams,
    *,
    max_pages: int | None = 20,
    fetch_detail: bool = True,
    headless: bool = True,
    fuente: str | None = "SJF",
    indice: str | None = "tesis",
    pdfs_dir: Path | None = None,
    prefer_direct_pdf: bool = False,
    log: LogFn | None = None,
    on_progress: ProgressFn | None = None,
) -> tuple[list[TesisRecord], str | None]:
    """
    Recorre páginas 1..max_pages. Detiene si no hay tarjetas de resultado (fin natural).

    Devuelve (registros, mensaje_fin | None). ``mensaje_fin`` indica fin por vacío, p.ej.
    ``Fin de resultados alcanzado``.
    """
    log = log or _default_log
    q = params.primary_query()
    if max_pages is None:
        max_pages = 200

    records: list[TesisRecord] = []
    seen: set[str] = set()
    fin_mensaje: str | None = None

    with sync_playwright() as p:
        browser = launch_browser(p, headless=headless)
        context = new_context(browser)
        page = context.new_page()
        request = context.request
        page.set_default_timeout(120_000)
        try:
            for pn in range(1, max_pages + 1):
                fuente_s = fuente or ""
                indice_s = indice or ""
                if on_progress:
                    on_progress(fuente_s, indice_s, pn)
                list_url = bj_busqueda_url(q, page=pn, fuente=fuente, indice=indice)
                params_url = parse_bj_busqueda_url(list_url)
                log(f"Listado p.{pn}: {list_url}")
                page.goto(list_url, wait_until="load")
                settle_page(page, headless=headless)

                if prefer_direct_pdf and pdfs_dir:
                    path_only = page.url.lower().split("?", 1)[0]
                    if path_only.endswith(".pdf"):
                        name = safe_pdf_filename(page.url, f"list_{pn}", "directo")
                        out = download_file(request, page.url, pdfs_dir / name, log=log)
                        if out:
                            rec = TesisRecord(
                                source="buscador_juridico",
                                numero_registro=f"PDF-{pn}",
                                rubro="PDF directo (respuesta de listado)",
                                texto_tesis=f"[PDF descargado: {out}]",
                                organo_emisor="",
                                epoca="",
                                url_detalle=page.url,
                                url_listado=list_url,
                                fuente=fuente_s,
                                indice=indice_s,
                                extra={
                                    "pagina_resultados": pn,
                                    "pdfs_descargados": [str(out)],
                                    "modo": "pdf_listado_directo",
                                },
                            )
                            records.append(rec)
                        fin_mensaje = (
                            "Listado devuelto como archivo PDF; descarga directa aplicada."
                        )
                        log(fin_mensaje)
                        break

                # Blindaje: sin div.card en el DOM → fin (páginas fuera de rango)
                n_any_card = page.locator("div.card").count()
                if n_any_card == 0:
                    fin_mensaje = "Fin de resultados alcanzado (sin selector div.card)."
                    log(fin_mensaje)
                    break

                cards = page.locator("div.card.mb-1").filter(
                    has_text=re.compile(r"Registro\s+digital", re.I)
                )
                n = cards.count()
                if n == 0:
                    fin_mensaje = "Fin de resultados alcanzado (sin tarjetas de resultado)."
                    log(fin_mensaje)
                    break

                for i in range(n):
                    card = cards.nth(i)
                    try:
                        txt = card.inner_text(timeout=15_000)
                    except Exception as e:
                        log(f"Tarjeta {i}: {e!r}")
                        continue
                    parsed = _parse_bj_card_text(txt)
                    reg = parsed["numero_registro"]
                    if not reg or reg in seen:
                        continue
                    seen.add(reg)

                    texto = parsed["resumen_lista"]
                    pdf_paths: list[str] = []
                    if fetch_detail:
                        try:
                            texto, pdf_paths = _fetch_bj_detail(
                                page,
                                request,
                                reg,
                                indice_s or "tesis",
                                headless,
                                log,
                                pdfs_dir,
                            )
                        except Exception as e:
                            log(f"Detalle {reg} falló: {e!r}")
                        page.goto(list_url, wait_until="load")
                        settle_page(page, headless=headless)

                    url_doc = f"https://bj.scjn.gob.mx{bj_documento_path(indice_s or 'tesis', reg)}"
                    extra: dict = {
                        "pagina_resultados": pn,
                        "pdfs_descargados": pdf_paths,
                    }
                    rec = TesisRecord(
                        source="buscador_juridico",
                        numero_registro=reg,
                        rubro=parsed["rubro"],
                        texto_tesis=texto,
                        organo_emisor=parsed["organo_emisor"],
                        epoca=parsed["epoca"],
                        url_detalle=url_doc,
                        url_listado=list_url,
                        fuente=str(params_url.get("fuente") or fuente_s or ""),
                        indice=str(params_url.get("indice") or indice_s or ""),
                        extra=extra,
                    )
                    records.append(rec)
                    time.sleep(0.15)

        finally:
            context.close()
            browser.close()

    return records, fin_mensaje


class BuscadorJuridicoAdapter:
    """Adaptador orientado a objeto (misma lógica que `scrape_buscador_juridico`)."""

    def search(
        self,
        params: SearchParams,
        *,
        max_pages: int | None = 20,
        fetch_detail: bool = True,
        headless: bool = True,
        fuente: str | None = "SJF",
        indice: str | None = "tesis",
        pdfs_dir: Path | None = None,
        prefer_direct_pdf: bool = False,
        log: LogFn | None = None,
        on_progress: ProgressFn | None = None,
    ) -> tuple[list[TesisRecord], str | None]:
        return scrape_buscador_juridico(
            params,
            max_pages=max_pages,
            fetch_detail=fetch_detail,
            headless=headless,
            fuente=fuente,
            indice=indice,
            pdfs_dir=pdfs_dir,
            prefer_direct_pdf=prefer_direct_pdf,
            log=log,
            on_progress=on_progress,
        )
