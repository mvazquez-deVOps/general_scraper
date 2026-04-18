"""
Buscador Jurídico — https://bj.scjn.gob.mx/

Selectores probados (Angular):
- Tarjetas de resultado: div.card.mb-1 (contiene "Registro digital:")
- Paginación URL: &page=N
- Detalle: ``.text-container-html`` + ``.additional-text`` (notas/votaciones); respaldo ``.documento-content`` y body.
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
from scjn_tesis.parsing import (
    clean_legal_artifacts,
    extract_registro_digital,
    parse_organo_epoca_line,
    trim_footer,
)

LogFn = Callable[[str], None]
ProgressFn = Callable[[str, str, int], None]
OnRecordFn = Callable[[TesisRecord], None]


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


# Fragmentos del detalle BJ (Angular): criterio + notas/votaciones al final.
_TEXT_CONTAINER = ".text-container-html"
_ADDITIONAL_TEXT = ".additional-text"
# Fragmentos principales del detalle: `.text-container-html`, `.additional-text`.

# Respaldo si no hay contenedores HTML: `.documento-content` y anteriores.
_TESIS_CONTAINERS = (".documento-content", ".texto-tesis", "#divDetalle", ".cuerpo-tesis")
_TESIS_WAIT_SELECTORS = ", ".join(_TESIS_CONTAINERS)
_RUBRO_SELECTORS = (".rubro-tesis", "app-view-tesis .rubro-tesis", "app-cuerpo-tesis .rubro-tesis")

def _wait_text_container_html_ready(page: Page, log: LogFn) -> None:
    """
    Hasta 20s: existen nodos .text-container-html, visibles y con algo de texto
    (no se usa el body entero en la ruta principal).
    """
    try:
        page.wait_for_selector(_TEXT_CONTAINER, state="visible", timeout=20_000)
    except Exception as e:
        log(f"wait_for_selector {_TEXT_CONTAINER} (visible, 20s): {e!r}")
    try:
        page.wait_for_function(
            r"""
            () => {
                for (const el of document.querySelectorAll('.text-container-html')) {
                    if (el.offsetParent == null) continue;
                    if ((el.innerText || "").trim().length > 3) { return true; }
                }
                return false;
            }
            """,
            timeout=20_000,
        )
    except Exception as e:
        log(f"wait_for_function .text-container-html con texto (20s): {e!r}")


def _extract_rubro_bj_line(page: Page) -> str:
    """Encabezado: clase .rubro-tesis o strong (vista tesis) cuando no queda en el cuerpo."""
    for sel in _RUBRO_SELECTORS:
        loc = page.locator(sel)
        try:
            n = loc.count()
        except Exception:
            n = 0
        if n > 0:
            t = loc.first.inner_text(timeout=5_000).strip()
            if t and len(t) > 3:
                return t
    locs = page.locator("app-view-tesis strong")
    try:
        m = min(locs.count(), 10)
    except Exception:
        m = 0
    for i in range(m):
        t = locs.nth(i).inner_text(timeout=3_000).strip()
        if t and 15 < len(t) < 600 and "\n" not in t and not t.lower().startswith("scjn"):
            if "época" not in t.lower() and "materia" not in t.lower():
                return t
    return ""


def _encabezado_antes_de_text_containers(page: Page) -> str:
    """
    Incluye Registro / Instancia / Materia que a menudo preceden a los
    `p, span.text-container-html` (no van en body crudo, solo rango bajo
    contenedor tesis).
    """
    try:
        return (
            page.evaluate(
                r"""() => {
                const first = document.querySelector('.text-container-html');
                if (!first) { return ""; }
                const root =
                    (first.closest && (first.closest('.documento-content')
                        || first.closest('app-view-tesis')
                        || first.closest('app-cuerpo-tesis'))) || null;
                if (!root || !root.contains(first)) { return ""; }
                const r = document.createRange();
                r.setStart(root, 0);
                r.setEndBefore(first);
                return (r.toString() || "").trim();
            }""",
            )
            or ""
        ).strip()
    except Exception:
        return ""


def _inner_texto_from_text_container_html(
    page: Page, log: LogFn, *, do_wait: bool = True
) -> str | None:
    """
    Concatenación de todos los `.text-container-html` (orden de documento),
    encabezado hasta el 1.º contenedor y rubro vía .rubro-tesis / strong si aplica.
    Al final concatena todos los `.additional-text` (notas, jurisprudencia, votaciones).
    Los bloques `.additional-text` pasan por `clean_legal_artifacts` antes de unir al cuerpo;
    el resultado completo sigue procesándose en `_fetch_bj_detail` con trim + clean.
    Si do_wait, espera 20s a visibilidad y texto mínimo en `.text-container-html`.
    """
    if do_wait:
        _wait_text_container_html_ready(page, log)
    chunks: list[str] = []
    try:
        loc = page.locator(_TEXT_CONTAINER)
        n = min(loc.count(), 200)
    except Exception as e:
        log(f"text-container count: {e!r}")
        n = 0
    for i in range(n):
        try:
            t = loc.nth(i).inner_text(timeout=30_000)
        except Exception as e:
            log(f"inner_text {_TEXT_CONTAINER} [{i}]: {e!r}")
            continue
        t = t.strip() if t else ""
        if t and t not in chunks:
            chunks.append(t)
    cuerpo = "\n\n".join(chunks)
    if not cuerpo.strip():
        return None
    pre = _encabezado_antes_de_text_containers(page)
    if pre and pre[:200] not in cuerpo and not cuerpo.lstrip().lower().startswith(
        pre[:40].lower().strip()[:20]
    ):
        cuerpo = f"{pre}\n\n{cuerpo}"
    rubro = _extract_rubro_bj_line(page)
    if rubro:
        head = cuerpo.lstrip()[:200].lower()
        rlow = rubro[:80].lower()
        if rlow not in head and not cuerpo.strip().lower().startswith(
            rubro[:20].lower()
        ):
            cuerpo = f"{rubro}\n\n{cuerpo}"

    add_parts: list[str] = []
    try:
        aloc = page.locator(_ADDITIONAL_TEXT)
        na = min(aloc.count(), 200)
    except Exception as e:
        log(f"additional-text count: {e!r}")
        na = 0
    for i in range(na):
        try:
            at = aloc.nth(i).inner_text(timeout=30_000)
        except Exception as e:
            log(f"inner_text {_ADDITIONAL_TEXT} [{i}]: {e!r}")
            continue
        at = (at or "").strip()
        if at and at not in add_parts:
            add_parts.append(at)
    if add_parts:
        add_blob = clean_legal_artifacts("\n\n".join(add_parts))
        if add_blob.strip():
            cuerpo = f"{cuerpo}\n\n{add_blob}"

    return cuerpo or None


# Detalle: esperar texto sustantivo, no un shell de listado
_MIN_CUERPO_SIN_SENAL = 500
_MIN_CUERPO_CON_INSTANCIA = 200

# Playwright: al menos un contenedor con criterio típico o bloque largo
_RICH_TESIS_FUNCTION = r"""
(sels) => {
  const haveInstancia = (t) => t.includes('Instancia');
  for (const s of sels) {
    for (const el of document.querySelectorAll(s)) {
      const t = (el.innerText || '').trim();
      if (t.length >= 400 && haveInstancia(t)) { return true; }
      if (t.length >= 2000) { return true; }
    }
  }
  const b = (document.body && document.body.innerText) ? document.body.innerText : '';
  return b.length >= 1500 && haveInstancia(b);
}
"""


def _cuerpo_tesis_looks_complete(text: str | None) -> bool:
    if not text or not (t := text.strip()):
        return False
    n = len(t)
    if n < _MIN_CUERPO_CON_INSTANCIA:
        return False
    if "Instancia:" in t and n >= 250:
        return True
    if n >= _MIN_CUERPO_SIN_SENAL:
        return True
    return False


def _pick_largest_tesis_node(page: Page, log: LogFn) -> str | None:
    best_text = ""
    best_key = (0, 0)  # (score, length)
    for sel in _TESIS_CONTAINERS:
        loc = page.locator(sel)
        try:
            n = min(loc.count(), 20)
        except Exception:
            n = 0
        for i in range(n):
            try:
                el = loc.nth(i)
                t = el.inner_text(timeout=60_000)
            except Exception as e:
                log(f"inner_text {sel}[{i}]: {e!r}")
                continue
            if not t:
                continue
            t = t.strip()
            nch = len(t)
            if nch < 30:
                continue
            # Prioridad: longitud, desempate si ya trae "Instancia:"
            inst_boost = 50_000 if "Instancia:" in t else 0
            key = (inst_boost + nch, nch)
            if key > best_key:
                best_key = key
                best_text = t
    return best_text if best_text else None


def _wait_tesis_rich_and_visible(page: Page, log: LogFn) -> None:
    try:
        page.wait_for_selector(
            _TESIS_WAIT_SELECTORS,
            state="visible",
            timeout=30_000,
        )
    except Exception as e:
        log(f"wait_for_selector tesis (visible, 30s): {e!r}")
    try:
        page.wait_for_function(
            _RICH_TESIS_FUNCTION,
            arg=list(_TESIS_CONTAINERS),
            timeout=25_000,
        )
    except Exception as e:
        log(f"wait_for_function criterio/Instancia: {e!r}")


def _inner_texto_tesis_contenedor(page: Page, log: LogFn) -> str | None:
    _wait_tesis_rich_and_visible(page, log)
    return _pick_largest_tesis_node(page, log)


def _texto_tesis_from_body_page(page: Page, log: LogFn) -> str:
    log("Respaldo detalle: body con recorte desde 'Instancia:' o texto depurado.")
    body = page.locator("body").inner_text(timeout=60_000)
    body = trim_footer(body)
    m = re.search(
        r"(?is)(Instancia:.*?)(?:UBICACI[ÓO]N|CONT[ÁA]CTANOS|REDES SOCIALES|$)",
        body,
    )
    if m and len(m.group(1).strip()) > 200:
        return m.group(1).strip()
    return body.strip()


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
    Carga detalle: primero texto del DOM; después enlaces PDF y, si aplica, respuesta binaria PDF.
    Nunca se abandona el texto del cuerpo aunque haya PDFs.
    """
    path_rel = bj_documento_path(indice, registro)
    url = f"https://bj.scjn.gob.mx{path_rel}"
    log(f"Detalle BJ: {url}")
    saved_pdfs: list[str] = []

    page.goto(url, wait_until="load")
    settle_page(page, headless=headless)

    cuerpo = _inner_texto_from_text_container_html(page, log)
    if not _cuerpo_tesis_looks_complete(cuerpo):
        log("Reintento: scroll y relectura de .text-container-html…")
        try:
            page.evaluate(
                "() => { window.scrollTo(0, Math.min(document.body.scrollHeight * 0.5, 2200)); }"
            )
        except Exception as e:
            log(f"scroll: {e!r}")
        time.sleep(1.0)
        c2 = _inner_texto_from_text_container_html(page, log, do_wait=False)
        if c2 and (not cuerpo or len(c2.strip()) > len((cuerpo or "").strip())):
            cuerpo = c2

    if not _cuerpo_tesis_looks_complete(cuerpo):
        log("Ruta respaldo: contenedores .documento-content / tesis (sin body completo como primera opción).")
        c3 = _inner_texto_tesis_contenedor(page, log)
        if c3 and (not cuerpo or len(c3.strip()) > len((cuerpo or "").strip())):
            cuerpo = c3
    if not _cuerpo_tesis_looks_complete(cuerpo):
        log("Reintento (documento-content): scroll…")
        try:
            page.evaluate(
                "() => { window.scrollTo(0, Math.min(document.body.scrollHeight * 0.4, 1600)); }"
            )
        except Exception as e:
            log(f"scroll: {e!r}")
        time.sleep(1.2)
        c4 = _pick_largest_tesis_node(page, log)
        if c4 and (not cuerpo or len(c4.strip()) > len((cuerpo or "").strip())):
            cuerpo = c4
        if not _cuerpo_tesis_looks_complete(cuerpo):
            cuerpo = None

    if cuerpo is not None and _cuerpo_tesis_looks_complete(cuerpo):
        texto = clean_legal_artifacts(trim_footer(cuerpo))
    else:
        if cuerpo and not _cuerpo_tesis_looks_complete(cuerpo):
            log("Cuerpo BJ por debajo de umbral o sin señal 'Instancia:'; se respalda con recorte de body (Instancia:).")
        else:
            log("Sin cuerpo útil; se respalda con recorte de body (Instancia:), no con body crudo ruidoso.")
        raw = _texto_tesis_from_body_page(page, log)
        texto = clean_legal_artifacts(raw)

    if pdfs_dir:
        for href in _collect_pdf_links(page):
            name = safe_pdf_filename(href, registro, "enlace")
            out = download_file(api_request, href, pdfs_dir / name, log=log)
            if out:
                saved_pdfs.append(str(out))
        try:
            r0 = api_request.get(url, timeout=120_000)
            if r0.ok and r0.body() and pdfs_dir:
                ct0 = (r0.headers.get("content-type") or "").lower()
                if "pdf" in ct0:
                    name = safe_pdf_filename(url, registro, "detalle")
                    pth = pdfs_dir / name
                    pth.parent.mkdir(parents=True, exist_ok=True)
                    pth.write_bytes(r0.body())
                    log(f"PDF guardado → {pth}")
                    # Evitar duplicar si el mismo path ya se registró
                    s = str(pth)
                    if s not in saved_pdfs:
                        saved_pdfs.append(s)
        except Exception as e:
            log(f"GET detalle (PDF binario): {e!r}")

    if saved_pdfs:
        texto = texto + "\n\nArchivo PDF descargado en almacen/pdfs"
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
    on_record: OnRecordFn | None = None,
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

                    # Respaldo: rubro y resumen del listado si el detalle falla
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
                            log(f"Detalle {reg} falló: {e!r} (se conserva resumen de listado).")
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
                    if on_record:
                        try:
                            on_record(rec)
                        except Exception as e:
                            log(f"on_record {reg}: {e!r}")
                    time.sleep(0.15)

        finally:
            context.close()
            browser.close()

    return records, fin_mensaje


class BuscadorJuridicoAdapter:
    """
    Misma lógica que `scrape_buscador_juridico`.
    Cada `TesisRecord.texto_tesis` con detalle BJ: ``.text-container-html`` + ``.additional-text``
    (+ rubro ``.rubro-tesis`` / ``strong``), luego ``.documento-content``; último, recorte ``Instancia:`` vía body.
    """

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
        on_record: OnRecordFn | None = None,
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
            on_record=on_record,
        )
