"""
Mapeo interactivo de sitios (SCJN): búsqueda + screenshot + persistencia de selectores.
Usa scjn_tesis.browser (launch_browser, new_context, settle_page).
"""

from __future__ import annotations

import json
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from playwright.sync_api import sync_playwright

from scjn_tesis.bj_urls import BJ_TEMPLATE_TESIS_SJF, bj_busqueda_url
from scjn_tesis.browser import launch_browser, new_context, settle_page

CONFIG_MAPA_FILENAME = "config_mapa.json"

# Dominios que disparan aviso de “Extractor universal” (no solo bj.scjn.gob.mx)
_EXTERNAL_DOMAIN_MARKERS = (
    "exlibrisgroup.com",
    "oas.org",
    "corteconstitucional.gov.co",
    "cortecostituzionale.it",
    "corteidh.scjn.gob.mx",  # servicio distinto al buscador BJ
)

LogFn = Callable[[str], None]


def _log(msg: str) -> None:
    print(f"[site_map] {msg}", flush=True)


# Candidatos CSS (primer match visible gana)
DEFAULT_INPUT_SELECTORS = [
    "input[name='search']",
    "input[placeholder*='Buscar' i]",
    "input[placeholder*='buscar' i]",
    "input[type='search']",
    "input.mat-mdc-input-element",
    "mat-toolbar input",
    "header input[type='text']",
]

DEFAULT_SUBMIT_SELECTORS = [
    "button[aria-label*='Buscar' i]",
    "button:has-text('Buscar')",
    "button[type='submit']",
    "mat-icon[fonticon='search']",
    "button.mat-mdc-icon-button:has(mat-icon)",
]

RESULTS_WAIT_SELECTORS = [
    "div.card.mb-1",
    "a.list-group-item[href*='/detalle/tesis/']",
    "a[href*='/documento/tesis/']",
    "[class*='resultado']",
]

DATA_DIR = Path(__file__).resolve().parent / "data"
MAP_FILENAME = "mapa_scjn.json"


def map_path(almacen_dir: Path) -> Path:
    return almacen_dir / MAP_FILENAME


def load_site_map(almacen_dir: Path) -> dict[str, Any] | None:
    p = map_path(almacen_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_extraction_url(
    user_url: str,
    keyword: str | None,
    map_data: dict[str, Any] | None,
) -> str:
    """
    Si hay mapa guardado para el host y palabra clave, usa plantilla de URL de resultados (piso 2).
    """
    if not map_data or not keyword or not str(keyword).strip():
        return user_url
    host = urllib.parse.urlparse(user_url).netloc.lower()
    hosts = [h.lower() for h in map_data.get("hosts", [])]
    if host not in hosts:
        return user_url
    tpl = map_data.get("search_url_template")
    if not tpl:
        return user_url
    q = urllib.parse.quote(keyword.strip())
    return tpl.replace("{query}", q).replace("{q}", q)


def save_site_map(almacen_dir: Path, payload: dict[str, Any]) -> Path:
    p = map_path(almacen_dir)
    almacen_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def config_mapa_path(almacen_dir: Path) -> Path:
    return almacen_dir / CONFIG_MAPA_FILENAME


def load_config_mapa(almacen_dir: Path) -> dict[str, Any] | None:
    p = config_mapa_path(almacen_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_config_mapa(almacen_dir: Path, payload: dict[str, Any]) -> Path:
    p = config_mapa_path(almacen_dir)
    almacen_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def resolve_extraction_url_with_config(
    user_url: str,
    keyword: str | None,
    almacen_dir: Path,
) -> str:
    """
    Prioridad: ``config_mapa.json`` → ``mapa_scjn.json`` (legacy) → URL sin cambiar.
    """
    cfg = load_config_mapa(almacen_dir)
    if cfg and keyword and str(keyword).strip():
        host = urllib.parse.urlparse(user_url).netloc.lower()
        if "bj.scjn.gob.mx" in host:
            tpl = cfg.get("search_url_template")
            if tpl:
                q = urllib.parse.quote(keyword.strip())
                out = tpl.replace("{query}", q).replace("{q}", q).replace("{page}", "1")
                return out
            fa = cfg.get("fuente_api") or cfg.get("fuente")
            ix = cfg.get("indice")
            if fa is not None and ix:
                return bj_busqueda_url(keyword.strip(), page=1, fuente=str(fa), indice=str(ix))
    legacy = load_site_map(almacen_dir)
    return resolve_extraction_url(user_url, keyword, legacy)


def run_bj_canonical_preview(
    *,
    fuente_api: str,
    indice: str,
    keyword: str,
    headed: bool = True,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """
    Navega a la URL canónica del Buscador, captura pantalla (zona resultados si existe)
    y detecta dominios externos / PDF.
    """
    log = log or _log
    keyword = keyword.strip()
    if not keyword:
        raise ValueError("Indica una palabra clave.")

    url = bj_busqueda_url(keyword, page=1, fuente=fuente_api, indice=indice)
    log(f"Previsualización canónica: {url}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shot_path = DATA_DIR / "temp_map_preview.png"

    external_notes: list[str] = []
    pdf_directo = False
    final_url = url

    with sync_playwright() as p:
        browser = launch_browser(p, headless=not headed)
        context = new_context(browser)
        page = context.new_page()
        page.set_default_timeout(120_000)
        try:
            page.goto(url, wait_until="load")
            settle_page(page, headless=not headed)
            time.sleep(2)
            try:
                page.wait_for_load_state("networkidle", timeout=60_000)
            except Exception:
                pass
            time.sleep(1)

            final_url = page.url
            if final_url.lower().split("?", 1)[0].endswith(".pdf"):
                pdf_directo = True

            pu = urllib.parse.urlparse(final_url)
            if pu.netloc and "bj.scjn.gob.mx" not in pu.netloc.lower():
                external_notes.append(
                    f"Dominio externo detectado ({pu.netloc}): se usará Extractor universal (Trafilatura) si abres esa URL."
                )

            try:
                link_nodes = page.query_selector_all("a[href]")[:80]
            except Exception:
                link_nodes = []
            for node in link_nodes:
                try:
                    href = node.get_attribute("href") or ""
                    if not href:
                        continue
                    low = href.lower()
                    if ".pdf" in low and "bj.scjn.gob.mx" in href:
                        pdf_directo = True
                    h = urllib.parse.urlparse(href)
                    if not h.netloc:
                        continue
                    net = h.netloc.lower()
                    if "bj.scjn.gob.mx" in net:
                        continue
                    for mark in _EXTERNAL_DOMAIN_MARKERS:
                        if mark in net or mark in href.lower():
                            external_notes.append(
                                f"Dominio externo detectado ({net}): enlace usa Extractor universal (Trafilatura)."
                            )
                            break
                except Exception:
                    continue

            clip = None
            for sel in ("main", "[role='main']", "div.card", ".container-fluid"):
                try:
                    loc = page.locator(sel).first
                    if not loc.is_visible(timeout=2000):
                        continue
                    box = loc.bounding_box()
                    if box and box.get("width", 0) > 50:
                        clip = {
                            "x": max(0, box["x"]),
                            "y": max(0, box["y"]),
                            "width": min(box["width"], 1400),
                            "height": min(box["height"], 2000),
                        }
                        break
                except Exception:
                    continue

            log(f"Screenshot → {shot_path}")
            if clip:
                page.screenshot(path=str(shot_path), clip=clip)
            else:
                page.screenshot(path=str(shot_path), full_page=False)
        finally:
            context.close()
            browser.close()

    return {
        "canonical_url": url,
        "final_url": final_url,
        "screenshot": str(shot_path.as_posix()),
        "external_notes": list(dict.fromkeys(external_notes)),
        "pdf_directo": pdf_directo,
        "fuente_api": fuente_api,
        "indice": indice,
        "keyword_sample": keyword,
    }


def build_config_mapa_payload(
    *,
    fuente_ui: str,
    fuente_api: str,
    indice: str,
    indice_label: str,
    keyword_sample: str,
    pdf_directo: bool,
    canonical_url: str,
) -> dict[str, Any]:
    """Persistencia validada para ``almacen/config_mapa.json`` (Buscador canónico)."""
    fu_enc = urllib.parse.quote(fuente_api)
    template = (
        f"https://bj.scjn.gob.mx/busqueda?fuente={fu_enc}&indice={indice}"
        f"&page={{page}}&q={{query}}&semantica=0"
    )
    return {
        "version": 2,
        "fuente_ui": fuente_ui,
        "fuente_api": fuente_api,
        "fuente": fuente_api,
        "indice": indice,
        "indice_label": indice_label,
        "search_url_template": template,
        "keyword_sample": keyword_sample,
        "pdf_directo": bool(pdf_directo),
        "canonical_url": canonical_url,
        "hosts": ["bj.scjn.gob.mx"],
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    }


def _first_visible(page, selectors: list[str], timeout_ms: int = 2500):
    from playwright.sync_api import TimeoutError as PWTimeout

    for sel in selectors:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="visible", timeout=timeout_ms)
            return sel, loc
        except PWTimeout:
            continue
        except Exception:
            continue
    return None, None


def run_site_mapping_flow(
    *,
    start_url: str,
    keyword: str,
    search_input_css: str | None = None,
    search_submit_css: str | None = None,
    headless: bool = True,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """
    Abre la URL, intenta escribir la palabra clave y ejecutar búsqueda, captura pantalla en data/temp_map.png.
    Devuelve dict con selectores usados y metadatos (para guardar en mapa_scjn.json).
    """
    log = log or _log
    keyword = keyword.strip()
    if not keyword:
        raise ValueError("Indica una palabra clave.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shot_path = DATA_DIR / "temp_map.png"

    used_input: str | None = search_input_css
    used_submit: str | None = search_submit_css
    final_url = start_url
    mode = "dom"

    with sync_playwright() as p:
        browser = launch_browser(p, headless=headless)
        context = new_context(browser)
        page = context.new_page()
        page.set_default_timeout(120_000)
        try:
            log(f"Navegando a {start_url!r}...")
            page.goto(start_url, wait_until="load")
            settle_page(page, headless=headless)

            filled = False
            inps = [search_input_css] if search_input_css else DEFAULT_INPUT_SELECTORS
            if search_input_css:
                loc = page.locator(search_input_css).first
                loc.wait_for(state="visible", timeout=15_000)
                loc.fill(keyword)
                filled = True
                used_input = search_input_css
            else:
                sel, loc = _first_visible(page, DEFAULT_INPUT_SELECTORS)
                if sel and loc:
                    loc.fill(keyword)
                    filled = True
                    used_input = sel

            if filled:
                log("Campo de búsqueda rellenado; buscando botón enviar...")
                clicked = False
                subs = [search_submit_css] if search_submit_css else DEFAULT_SUBMIT_SELECTORS
                if search_submit_css:
                    page.locator(search_submit_css).first.click(timeout=15_000)
                    clicked = True
                    used_submit = search_submit_css
                else:
                    for s in subs:
                        try:
                            btn = page.locator(s).first
                            if btn.is_visible(timeout=2000):
                                btn.click()
                                clicked = True
                                used_submit = s
                                break
                        except Exception:
                            continue
                if not clicked:
                    log("Sin botón; envío con Enter.")
                    page.keyboard.press("Enter")
                    used_submit = used_submit or "__enter__"
            else:
                # Fallback Buscador Jurídico: URL de resultados directa
                if "bj.scjn.gob.mx" in start_url:
                    final_url = bj_busqueda_url(keyword, page=1, fuente="SJF", indice="tesis")
                    log(f"Sin input visible; usando URL de búsqueda BJ (Tesis SJF): {final_url}")
                    page.goto(final_url, wait_until="load")
                    mode = "url_fallback_bj"
                    used_input = "__fallback_busqueda_tesis__"
                    used_submit = "__none__"
                elif "sjf2.scjn.gob.mx" in start_url:
                    q = urllib.parse.quote(keyword)
                    final_url = f"{start_url.split('?')[0]}?q={q}"  # puede no funcionar; mejor solo fill
                    log("SJF: intentando solo Enter en página...")
                    page.keyboard.press("Enter")
                    mode = "enter_only"
                    used_input = "__not_found__"
                else:
                    raise RuntimeError(
                        "No se encontró campo de búsqueda visible. "
                        "Prueba selectores CSS personalizados o una URL de portal con caja de búsqueda."
                    )

            log("Esperando resultados...")
            time.sleep(2)
            try:
                page.wait_for_load_state("networkidle", timeout=60_000)
            except Exception:
                pass
            for rs in RESULTS_WAIT_SELECTORS:
                try:
                    page.wait_for_selector(rs, timeout=25_000)
                    log(f"Detectado contenedor de resultados: {rs!r}")
                    break
                except Exception:
                    continue
            settle_page(page, headless=headless)

            final_url = page.url
            log(f"Capturando screenshot → {shot_path}")
            page.screenshot(path=str(shot_path), full_page=False)
        finally:
            context.close()
            browser.close()

    host = urllib.parse.urlparse(start_url).netloc.lower()
    payload: dict[str, Any] = {
        "version": 1,
        "hosts": [host],
        "start_url": start_url,
        "keyword_sample": keyword,
        "mode": mode,
        "selectors": {
            "search_input": used_input,
            "search_submit": used_submit,
            "results_wait": RESULTS_WAIT_SELECTORS[0],
        },
        "search_url_template": (
            BJ_TEMPLATE_TESIS_SJF if "bj.scjn.gob.mx" in host else ""
        ),
        "screenshot": str(shot_path.as_posix()),
        "mapped_at": datetime.now(timezone.utc).isoformat(),
    }
    if not payload["search_url_template"]:
        del payload["search_url_template"]

    return payload
