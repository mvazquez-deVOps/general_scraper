"""
Buscador Jurídico — https://bj.scjn.gob.mx/

Selectores probados (Angular):
- Tarjetas de resultado: div.card.mb-1 (contiene "Registro digital:")
- Paginación URL: &page=N
- Detalle tesis: /documento/tesis/{registro}
"""

from __future__ import annotations

import re
import time
import urllib.parse
from typing import Callable

from playwright.sync_api import Page, sync_playwright

from scjn_tesis.browser import launch_browser, new_context, settle_page
from scjn_tesis.models import SearchParams, TesisRecord
from scjn_tesis.parsing import extract_registro_digital, parse_organo_epoca_line, trim_footer

LogFn = Callable[[str], None]


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


def _fetch_bj_detail(page: Page, registro: str, headless: bool, log: LogFn) -> str:
    url = f"https://bj.scjn.gob.mx/documento/tesis/{registro}"
    log(f"Detalle BJ: {url}")
    page.goto(url, wait_until="load")
    settle_page(page, headless=headless)
    body = page.locator("body").inner_text(timeout=60_000)
    body = trim_footer(body)
    # Contenido principal: desde rubro en mayúsculas típico hasta antes de repetir registro
    m = re.search(
        r"(?is)(Instancia:.*?)(?:UBICACI[ÓO]N|CONT[ÁA]CTANOS|REDES SOCIALES|$)",
        body,
    )
    if m:
        return m.group(1).strip()
    return body.strip()


def scrape_buscador_juridico(
    params: SearchParams,
    *,
    max_pages: int | None = 20,
    fetch_detail: bool = True,
    headless: bool = True,
    log: LogFn | None = None,
) -> list[TesisRecord]:
    """
    Recorre resultados de búsqueda de tesis en el Buscador Jurídico.
    `max_pages`: límite de páginas de resultados (None = hasta 200 por seguridad).
    """
    log = log or _default_log
    q = params.primary_query()
    if max_pages is None:
        max_pages = 200

    records: list[TesisRecord] = []
    seen: set[str] = set()

    with sync_playwright() as p:
        browser = launch_browser(p, headless=headless)
        context = new_context(browser)
        page = context.new_page()
        page.set_default_timeout(120_000)
        try:
            for pn in range(1, max_pages + 1):
                list_url = (
                    f"https://bj.scjn.gob.mx/busqueda?indice=tesis&q={urllib.parse.quote(q)}&page={pn}"
                )
                log(f"Listado p.{pn}: {list_url}")
                page.goto(list_url, wait_until="load")
                settle_page(page, headless=headless)

                cards = page.locator("div.card.mb-1").filter(has_text=re.compile(r"Registro\s+digital", re.I))
                n = cards.count()
                if n == 0:
                    log("Sin tarjetas; fin de paginación.")
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
                    if fetch_detail:
                        try:
                            texto = _fetch_bj_detail(page, reg, headless, log)
                        except Exception as e:
                            log(f"Detalle {reg} falló: {e!r}")
                        # volver al listado para el siguiente ítem
                        page.goto(list_url, wait_until="load")
                        settle_page(page, headless=headless)

                    rec = TesisRecord(
                        source="buscador_juridico",
                        numero_registro=reg,
                        rubro=parsed["rubro"],
                        texto_tesis=texto,
                        organo_emisor=parsed["organo_emisor"],
                        epoca=parsed["epoca"],
                        url_detalle=f"https://bj.scjn.gob.mx/documento/tesis/{reg}",
                        url_listado=list_url,
                        extra={"pagina_resultados": pn},
                    )
                    records.append(rec)
                    time.sleep(0.15)

                # si la página devolvió solo duplicados (vista repetida), cortar
                if pn == 1 and n == 0:
                    break
        finally:
            context.close()
            browser.close()

    return records


class BuscadorJuridicoAdapter:
    """Adaptador orientado a objeto (misma lógica que `scrape_buscador_juridico`)."""

    def search(
        self,
        params: SearchParams,
        *,
        max_pages: int | None = 20,
        fetch_detail: bool = True,
        headless: bool = True,
        log: LogFn | None = None,
    ) -> list[TesisRecord]:
        return scrape_buscador_juridico(
            params,
            max_pages=max_pages,
            fetch_detail=fetch_detail,
            headless=headless,
            log=log,
        )
