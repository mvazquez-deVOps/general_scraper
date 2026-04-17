"""
Semanario Judicial de la Federación (SJF 2.0) — https://sjf2.scjn.gob.mx/

Selectores probados:
- Búsqueda: input[name='search'], envío con Enter
- Resultados: a.list-group-item[href^='/detalle/tesis/']
- Rubro: p.block-with-text
- Localización / órgano: p.fc-localizacion
- Paginación: ul.pagination li.page-item.active y enlace numérico (página actual + 1)
- Detalle: /detalle/tesis/{registro}
"""

from __future__ import annotations

import re
import time
from typing import Callable

from playwright.sync_api import Locator, Page, sync_playwright

from scjn_tesis.browser import launch_browser, new_context, settle_page
from scjn_tesis.models import SearchParams, TesisRecord
from scjn_tesis.parsing import parse_organo_epoca_line, trim_footer

LogFn = Callable[[str], None]


def _log_default(msg: str) -> None:
    print(f"[sjf] {msg}", flush=True)


def _parse_list_row(link: Locator) -> dict[str, str]:
    href = link.get_attribute("href") or ""
    m = re.search(r"/detalle/tesis/(\d+)", href)
    reg = m.group(1) if m else ""
    try:
        tit = link.locator(".titulo").first.inner_text(timeout=8000)
    except Exception:
        tit = ""
    try:
        rub = link.locator("p.block-with-text").first.inner_text(timeout=8000)
    except Exception:
        rub = ""
    try:
        loc = link.locator("p.fc-localizacion").first.inner_text(timeout=8000)
    except Exception:
        loc = ""
    org, ep = parse_organo_epoca_line(loc)
    return {
        "numero_registro": reg,
        "rubro": rub.strip(),
        "organo_emisor": org,
        "epoca": ep,
        "meta_line": loc.strip(),
        "href": href,
        "titulo_line": tit.strip(),
    }


def _current_page_number(page: Page) -> int | None:
    try:
        t = page.locator("ul.pagination li.page-item.active a.page-link").first.inner_text(timeout=8000)
        return int(t.strip())
    except Exception:
        return None


def _click_next_result_page(page: Page, log: LogFn) -> bool:
    cur = _current_page_number(page)
    if cur is None:
        log("Paginación: no hay página activa.")
        return False
    nxt = str(cur + 1)
    link = page.locator("ul.pagination li.page-item:not(.disabled) a.page-link").filter(has_text=nxt)
    if link.count() == 0:
        log(f"Paginación: no existe enlace «{nxt}» (última página o ventana sin número siguiente).")
        return False
    link.first.click()
    time.sleep(1.5)
    try:
        page.wait_for_load_state("networkidle", timeout=90_000)
    except Exception:
        pass
    time.sleep(1.5)
    return True


def _fetch_detail_text(page: Page, registro: str, headless: bool, log: LogFn) -> str:
    url = f"https://sjf2.scjn.gob.mx/detalle/tesis/{registro}"
    log(f"Detalle: {url}")
    page.goto(url, wait_until="load")
    settle_page(page, headless=headless)
    body = page.locator("body").inner_text(timeout=90_000)
    return trim_footer(body).strip()


def scrape_semanario(
    params: SearchParams,
    *,
    max_pages: int | None = 20,
    fetch_detail: bool = True,
    headless: bool = True,
    log: LogFn | None = None,
) -> list[TesisRecord]:
    log = log or _log_default
    q = params.primary_query()
    if max_pages is None:
        max_pages = 200

    base_list = "https://sjf2.scjn.gob.mx/busqueda-principal-tesis"
    rows: list[dict[str, str]] = []
    detail_map: dict[str, str] = {}

    with sync_playwright() as p:
        browser = launch_browser(p, headless=headless)
        context = new_context(browser)
        page = context.new_page()
        page.set_default_timeout(120_000)
        try:
            log(f"Apertura {base_list}")
            page.goto(base_list, wait_until="load")
            settle_page(page, headless=headless)
            page.locator("input[name='search']").first.fill(q)
            page.keyboard.press("Enter")
            time.sleep(2)
            try:
                page.wait_for_load_state("networkidle", timeout=120_000)
            except Exception:
                pass
            settle_page(page, headless=headless)
            page.wait_for_selector("a.list-group-item[href^='/detalle/tesis/']", timeout=120_000)

            for pn in range(1, max_pages + 1):
                log(f"Listado página {pn} (resultados)")
                links = page.locator("a.list-group-item[href^='/detalle/tesis/']")
                n = links.count()
                if n == 0:
                    log("Sin filas de resultados.")
                    break
                for i in range(n):
                    try:
                        row = _parse_list_row(links.nth(i))
                        if row["numero_registro"]:
                            row["pagina_resultados"] = str(pn)
                            rows.append(row)
                    except Exception as e:
                        log(f"Fila {i}: {e!r}")
                if pn >= max_pages:
                    break
                if not _click_next_result_page(page, log):
                    break

            # Segunda fase: texto completo desde detalle
            if fetch_detail and rows:
                uniq = []
                s2 = set()
                for r in rows:
                    reg = r["numero_registro"]
                    if reg and reg not in s2:
                        s2.add(reg)
                        uniq.append(reg)
                log(f"Descargando detalle para {len(uniq)} registros únicos...")
                for reg in uniq:
                    try:
                        detail_map[reg] = _fetch_detail_text(page, reg, headless, log)
                    except Exception as e:
                        log(f"Detalle {reg}: {e!r}")
                        detail_map[reg] = ""

        finally:
            context.close()
            browser.close()

    records: list[TesisRecord] = []
    done: set[str] = set()
    for row in rows:
        reg = row["numero_registro"]
        if reg in done:
            continue
        done.add(reg)
        texto = (
            (detail_map.get(reg) or "")
            if fetch_detail
            else f"{row['rubro']}\n{row['meta_line']}".strip()
        )
        if fetch_detail and not texto.strip():
            texto = f"{row['rubro']}\n{row['meta_line']}".strip()
        records.append(
            TesisRecord(
                source="semanario_judicial",
                numero_registro=reg,
                rubro=row["rubro"],
                texto_tesis=texto,
                organo_emisor=row["organo_emisor"],
                epoca=row["epoca"],
                fuente="SJF",
                indice="tesis",
                url_detalle=f"https://sjf2.scjn.gob.mx/detalle/tesis/{reg}",
                url_listado=base_list,
                extra={
                    "pagina_resultados": int(row.get("pagina_resultados", 0) or 0),
                    "titulo_lista": row.get("titulo_line", ""),
                },
            )
        )

    return records


class SemanarioAdapter:
    def search(
        self,
        params: SearchParams,
        *,
        max_pages: int | None = 20,
        fetch_detail: bool = True,
        headless: bool = True,
        log: LogFn | None = None,
    ) -> list[TesisRecord]:
        return scrape_semanario(
            params,
            max_pages=max_pages,
            fetch_detail=fetch_detail,
            headless=headless,
            log=log,
        )
