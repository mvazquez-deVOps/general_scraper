"""
Aplicación Streamlit: extracción con Playwright + Trafilatura y vista de entrenamiento.
Los JSON se guardan en la carpeta local ./almacen
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import tempfile
import urllib.parse
from datetime import datetime
from pathlib import Path

# Playwright arranca Chromium por subproceso; en Windows el bucle asyncio por defecto
# puede no implementar subprocess (p. ej. con Python 3.12+), lo que provoca NotImplementedError.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import trafilatura
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

from site_mapping import (
    build_config_mapa_payload,
    load_config_mapa,
    load_site_map,
    resolve_extraction_url_with_config,
    run_bj_canonical_preview,
    run_site_mapping_flow,
    save_config_mapa,
    save_site_map,
)

from scjn_tesis.bj_fuentes_catalog import (
    etiqueta_indice_por_slug,
    fuente_api_desde_ui,
    indices_para_fuente,
    listar_fuentes_ui,
)

from scjn_tesis.buscador_juridico import scrape_buscador_juridico
from scjn_tesis.models import SearchParams, TesisRecord

# --- Rutas y constantes ---
BASE_DIR = Path(__file__).resolve().parent
ALMACEN_DIR = BASE_DIR / "almacen"
ALMACEN_DIR.mkdir(parents=True, exist_ok=True)
PDFS_DIR = ALMACEN_DIR / "pdfs"
PDFS_DIR.mkdir(parents=True, exist_ok=True)

PAGE_HOME = "inicio"
PAGE_EXTRAER = "extraer"
PAGE_ENTRENAR = "entrenar"
PAGE_ALMACEN = "almacen"
PAGE_MAPEAR = "mapear"

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TEMP_MAP_PNG = DATA_DIR / "temp_map.png"
TEMP_MAP_PREVIEW_PNG = DATA_DIR / "temp_map_preview.png"

MODE_URL = "URL Directa"
MODE_BUSQUEDA = "Búsqueda por Palabras"

# Navegador “real” para sitios que renderizan con JS o filtran automation (p. ej. Blazor WASM)
_CHROMIUM_ARGS = (
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
)
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
# Script anti-detección básico (muchas plantillas comprobarán navigator.webdriver)
_STEALTH_INIT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['es-MX', 'es', 'en-US', 'en'] });
"""

# Orden: primero Chrome real (mejor compat. con WASM / antibot), luego Chromium embebido;
# si sigue la plantilla de error Blazor, se prueba ventana visible (a veces el WASM solo arranca ahí).
_EXTRACTION_STRATEGIES: tuple[tuple[str, bool, str | None], ...] = (
    ("Chrome headless", True, "chrome"),
    ("Chromium Playwright headless", True, None),
    ("Chrome ventana visible", False, "chrome"),
    ("Chromium Playwright ventana visible", False, None),
)


def log_step(msg: str) -> None:
    """Registro en consola del backend."""
    print(f"[extractor] {msg}", flush=True)


def format_user_error(exc: BaseException) -> str:
    """Incluye el tipo: str() de NotImplementedError() es vacío en Python."""
    name = type(exc).__name__
    msg = str(exc).strip()
    if msg:
        return f"{name}: {msg}"
    return f"{name} (sin mensaje; revisa la consola del servidor)"


def _launch_chromium(p, *, headless: bool, channel: str | None) -> object:
    """Lanza Chromium; si `channel='chrome'` no está instalado, cae al embebido."""
    kw: dict = {
        "headless": headless,
        "args": list(_CHROMIUM_ARGS),
        "ignore_default_args": ["--enable-automation"],
    }
    if channel:
        kw["channel"] = channel
    try:
        return p.chromium.launch(**kw)
    except Exception as e:
        if channel:
            log_step(f"No se pudo usar channel={channel!r} ({e!r}); reintentando con Chromium embebido.")
            kw.pop("channel", None)
            return p.chromium.launch(**kw)
        raise


def _create_context(browser) -> object:
    ctx = browser.new_context(
        user_agent=_USER_AGENT,
        viewport={"width": 1366, "height": 768},
        locale="es-MX",
        timezone_id="America/Mexico_City",
        extra_http_headers={
            "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Upgrade-Insecure-Requests": "1",
        },
    )
    ctx.add_init_script(_STEALTH_INIT)
    return ctx


def _new_browser_context(p, *, headless: bool = True, channel: str | None = "chrome"):
    """Un intento: Chromium + contexto (p. ej. búsqueda Bing)."""
    browser = _launch_chromium(p, headless=headless, channel=channel)
    context = _create_context(browser)
    return browser, context


def _is_blazor_or_js_placeholder(text: str) -> bool:
    """Detecta plantilla de error típica (Blazor / ‘enable JavaScript’)."""
    if not text or not str(text).strip():
        return True
    low = text.lower()
    if "you must enable javascript" in low:
        return True
    if "an error has occurred" in low and "usual error causes" in low:
        return True
    if "an error has occurred :-(" in low and len(text.strip()) < 2500:
        return True
    return False


def _body_inner_text(page) -> str:
    try:
        return page.locator("body").inner_text(timeout=20_000)
    except Exception:
        return ""


def _settle_after_navigation(page, *, headless: bool) -> None:
    log_step("Esperando networkidle (hasta 60s)...")
    try:
        page.wait_for_load_state("networkidle", timeout=60_000)
    except Exception as ex:
        log_step(f"networkidle no alcanzado: {type(ex).__name__}")
    try:
        page.evaluate("window.scrollTo(0, Math.min(document.body.scrollHeight, 4000))")
        time.sleep(0.5)
        page.evaluate("window.scrollTo(0, 0)")
    except Exception:
        pass
    extra = 3 if headless else 6
    log_step(f"Pausa final {extra}s (SPA / WASM)...")
    time.sleep(extra)


def _extract_text_hybrid(page, html: str, final_url: str) -> tuple[str, str]:
    """Trafilatura + texto visible del DOM; elige el que no parezca página de error."""
    meta = trafilatura.extract_metadata(html, default_url=final_url)
    title = (meta.title or "") if meta else ""
    t_traf = trafilatura.extract(
        html,
        url=final_url,
        include_comments=False,
        include_tables=True,
    )
    if t_traf is None:
        t_traf = ""
    t_body = _body_inner_text(page)

    if _is_blazor_or_js_placeholder(t_traf) and t_body.strip() and not _is_blazor_or_js_placeholder(t_body):
        log_step("Usando texto del DOM (Trafilatura devolvía plantilla de error).")
        return t_body.strip(), title
    if not _is_blazor_or_js_placeholder(t_traf) and len(t_traf.strip()) >= 80:
        return t_traf.strip(), title
    if len(t_body.strip()) > len(t_traf.strip()):
        log_step("Usando texto del DOM por ser más largo que Trafilatura.")
        return t_body.strip(), title
    return t_traf.strip(), title


def _one_extraction_attempt(
    p,
    target_url: str,
    *,
    label: str,
    headless: bool,
    channel: str | None,
) -> tuple[str, str, str]:
    log_step(f"--- Intento: {label} ---")
    browser = _launch_chromium(p, headless=headless, channel=channel)
    context = _create_context(browser)
    try:
        page = context.new_page()
        page.set_default_timeout(120_000)
        log_step("Esperando carga (load)...")
        page.goto(target_url, wait_until="load")
        _settle_after_navigation(page, headless=headless)
        final_url = page.url
        log_step("Leyendo HTML y texto visible...")
        html = page.content()
        texto, title = _extract_text_hybrid(page, html, final_url)
        log_step(f"Caracteres extraídos: {len(texto)}.")
        return texto, title, final_url
    finally:
        context.close()
        browser.close()


def sanitize_filename_from_url(url: str, max_len: int = 150) -> str:
    """Parte del nombre de archivo derivada de la URL (sin caracteres inválidos)."""
    safe = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", url.strip())
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe or "sin_url"


def build_json_path(url: str) -> Path:
    """Nombre: fecha + URL sanitizada; evita colisión con sufijo hora."""
    fecha = datetime.now().strftime("%Y-%m-%d")
    base = f"{fecha}_{sanitize_filename_from_url(url)}.json"
    path = ALMACEN_DIR / base
    if path.exists():
        t = datetime.now().strftime("%H%M%S")
        base = f"{fecha}_{sanitize_filename_from_url(url)}_{t}.json"
        path = ALMACEN_DIR / base
    return path


def extract_with_playwright(target_url: str) -> tuple[str, str, str]:
    """
    Varios intentos (Chrome/Chromium, headless/visible) para sitios difíciles (p. ej. Blazor WASM).
    Combina Trafilatura con texto visible del DOM.
    """
    log_step(f"Extrayendo {target_url!r}...")
    last_ok: tuple[str, str, str] | None = None
    with sync_playwright() as p:
        for label, headless, channel in _EXTRACTION_STRATEGIES:
            try:
                texto, title, final_url = _one_extraction_attempt(
                    p,
                    target_url,
                    label=label,
                    headless=headless,
                    channel=channel,
                )
                last_ok = (texto, title, final_url)
                if not _is_blazor_or_js_placeholder(texto):
                    log_step(f"Contenido aceptable con estrategia: {label}")
                    return texto, title, final_url
                log_step(
                    f"El resultado sigue pareciendo plantilla de error; probando siguiente estrategia..."
                )
            except Exception as e:
                log_step(f"Estrategia {label!r} falló: {format_user_error(e)}")
                continue
    if last_ok:
        log_step(
            "Advertencia: todas las estrategias devolvieron plantilla de error o solo la última respondió."
        )
        return last_ok
    raise RuntimeError(
        "No se pudo completar la extracción con ninguna estrategia de navegador."
    )


def search_bing_first_result_url(query: str) -> str | None:
    """Abre Bing, intenta obtener la URL del primer resultado orgánico."""
    q = urllib.parse.quote_plus(query)
    search_url = f"https://www.bing.com/search?q={q}"
    log_step(f"Búsqueda por palabras: {query!r} → {search_url}")
    with sync_playwright() as p:
        browser, context = _new_browser_context(p)
        try:
            page = context.new_page()
            page.set_default_timeout(60_000)
            page.goto(search_url, wait_until="load")
            log_step("Localizando primer resultado en la SERP...")
            link = page.query_selector("li.b_algo h2 a")
            if link:
                href = link.get_attribute("href")
                if href:
                    log_step(f"Primer resultado encontrado: {href!r}")
                    return href
            log_step("No se encontró enlace orgánico típico; se usará la URL de la propia SERP.")
            return None
        finally:
            context.close()
            browser.close()


def guardar_registro_bj_almacen(record: TesisRecord) -> Path:
    """JSON compatible con Entrenar: incluye fuente e índice para filtrar en fase 2."""
    fecha = datetime.now().strftime("%Y-%m-%d")
    ind_safe = re.sub(r'[<>:"/\\|?*]', "_", (record.indice or "tesis"))[:50]
    fname = f"{fecha}_bj_{ind_safe}_{record.numero_registro}.json"
    path = ALMACEN_DIR / fname
    payload = {
        "url": record.url_detalle,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "titulo": (record.rubro or "")[:500],
        "texto": record.texto_tesis,
        "fuente": record.fuente,
        "indice": record.indice,
        "source": record.source,
        "numero_registro": record.numero_registro,
        "organo_emisor": record.organo_emisor,
        "epoca": record.epoca,
        "url_listado": record.url_listado,
        "extra": record.extra,
        "scraped_at": record.scraped_at,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_step(f"Registro BJ guardado: {path}")
    return path


def guardar_json_almacen(url_ref: str, texto: str, titulo: str) -> Path:
    """Guarda extracción en JSON bajo almacen/."""
    log_step("Guardando en almacén (JSON)...")
    path = build_json_path(url_ref)
    payload = {
        "url": url_ref,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "titulo": titulo,
        "texto": texto,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_step(f"Archivo guardado: {path}")
    return path


def leer_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def leer_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(parts)


def listar_almacen_json() -> list[Path]:
    return sorted(ALMACEN_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def listar_almacen_json_entrenamiento() -> list[Path]:
    """Excluye configuración de mapa; no es un JSON de texto extraído."""
    skip = {"mapa_scjn.json", "config_mapa.json"}
    return [p for p in listar_almacen_json() if p.name not in skip]


def init_session() -> None:
    if "pagina" not in st.session_state:
        st.session_state.pagina = PAGE_HOME
    if "preview_almacen_file" not in st.session_state:
        st.session_state.preview_almacen_file = None
    if "map_pending" not in st.session_state:
        st.session_state.map_pending = None
    if "map_bj_pending" not in st.session_state:
        st.session_state.map_bj_pending = None
    cfg0 = load_config_mapa(ALMACEN_DIR)
    if "bj_bulk_fuente" not in st.session_state:
        st.session_state.bj_bulk_fuente = (
            str(cfg0.get("fuente_api") or cfg0.get("fuente") or "SJF") if cfg0 else "SJF"
        )
    if "bj_bulk_indice" not in st.session_state:
        st.session_state.bj_bulk_indice = str(cfg0.get("indice") or "tesis") if cfg0 else "tesis"


def ir(pagina: str) -> None:
    st.session_state.pagina = pagina


def preview_snippet(text: str, max_chars: int = 400) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def render_home() -> None:
    st.title("Extractor universal")
    st.markdown("### ¿Qué deseas hacer?")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### Extraer")
            st.caption("Scraping con Playwright y texto limpio con Trafilatura.")
            if st.button("Ir a Extraer", key="btn_extraer", use_container_width=True):
                ir(PAGE_EXTRAER)
                st.rerun()
    with c2:
        with st.container(border=True):
            st.markdown("#### Entrenar")
            st.caption("Revisar texto desde el almacén o archivos subidos.")
            if st.button("Ir a Entrenar", key="btn_entrenar", use_container_width=True):
                ir(PAGE_ENTRENAR)
                st.rerun()
    with c3:
        with st.container(border=True):
            st.markdown("#### Mapear Sitio")
            st.caption("Previsualiza fuente/índice del Buscador Jurídico y confirma el mapa.")
            if st.button("Ir a Mapear", key="btn_mapear", use_container_width=True):
                ir(PAGE_MAPEAR)
                st.rerun()

    st.markdown("")
    ac1, ac2, ac3 = st.columns([1, 2, 1])
    with ac2:
        with st.container(border=True):
            st.markdown(
                "<div style='text-align:center'>📦 <strong>Almacén</strong></div>",
                unsafe_allow_html=True,
            )
            st.caption("Archivos JSON guardados automáticamente.")
            if st.button("Ver Almacén", key="btn_almacen", use_container_width=True):
                ir(PAGE_ALMACEN)
                st.rerun()


def render_extraer() -> None:
    st.header("Extraer")
    if st.button("← Volver al inicio"):
        ir(PAGE_HOME)
        st.rerun()

    with st.expander("Descarga masiva — Buscador Jurídico (bj.scjn.gob.mx)", expanded=False):
        cfg_bulk = load_config_mapa(ALMACEN_DIR)
        st.markdown(
            "Paginación `page=1..N`, detección de fin de resultados, guardado por registro con "
            "**fuente** e **índice**, y PDFs en `almacen/pdfs/`."
        )
        if cfg_bulk:
            st.success(
                "Hay **`almacen/config_mapa.json`**: los valores de fuente/índice y el modo PDF "
                "provienen del mapeo confirmado en **Mapear sitio**."
            )
            st.caption(
                f"Mapeo: **{cfg_bulk.get('fuente_ui', '?')}** · "
                f"{cfg_bulk.get('indice_label', cfg_bulk.get('indice'))} · "
                f"PDF directo: {cfg_bulk.get('pdf_directo', False)}"
            )
        dq = st.text_input("Palabra clave", key="bj_bulk_q", placeholder="pagare")
        cf = st.text_input("Fuente", key="bj_bulk_fuente", placeholder="SJF")
        ci = st.text_input("Índice", key="bj_bulk_indice", placeholder="tesis")
        max_p = st.number_input("Máximo de páginas", min_value=1, max_value=500, value=5, key="bj_bulk_pages")
        det = st.checkbox("Abrir cada detalle (más lento)", value=True, key="bj_bulk_detail")
        bj_head = st.checkbox("Navegador visible (BJ masivo)", key="bj_bulk_headed")
        prog = st.empty()
        if st.button("Iniciar descarga masiva", type="primary", key="bj_bulk_run"):
            if not dq or not dq.strip():
                st.warning("Indica una palabra clave.")
            else:
                params = SearchParams(texto=dq.strip())

                def on_prog(fuente: str, indice: str, page: int) -> None:
                    line = f"Procesando Fuente: {fuente} | Índice: {indice} | Página: {page}"
                    log_step(line)
                    prog.markdown(
                        f"Procesando **Fuente:** `{fuente}` | **Índice:** `{indice}` | **Página:** `{page}`"
                    )

                def on_record_bj_save(rec: TesisRecord) -> None:
                    try:
                        guardar_registro_bj_almacen(rec)
                    except Exception as e:
                        log_step(
                            f"Error al guardar tesis {rec.numero_registro!r} "
                            f"(el scraping continúa con la siguiente): {e!r}"
                        )

                try:
                    with st.spinner("Descargando…"):
                        prefer_pdf = bool(cfg_bulk.get("pdf_directo")) if cfg_bulk else False
                        records, fin_msg = scrape_buscador_juridico(
                            params,
                            max_pages=int(max_p),
                            fetch_detail=det,
                            headless=not bj_head,
                            fuente=cf.strip() or "SJF",
                            indice=ci.strip() or "tesis",
                            pdfs_dir=PDFS_DIR,
                            prefer_direct_pdf=prefer_pdf,
                            log=log_step,
                            on_progress=on_prog,
                            on_record=on_record_bj_save,
                        )
                    prog.markdown(
                        f"Procesando **Fuente:** `{cf}` | **Índice:** `{ci}` | **Listo.**"
                    )
                    if fin_msg:
                        st.warning(fin_msg)
                    st.success(f"Guardados {len(records)} registros en `./almacen` (PDFs en `./almacen/pdfs`).")
                except Exception as e:
                    log_step(f"BJ masivo: {e!r}")
                    st.error(format_user_error(e))

    modo = st.radio(
        "Modo",
        [MODE_URL, MODE_BUSQUEDA],
        horizontal=True,
        key="radio_extraer_modo",
    )
    st.caption(
        "URLs difíciles (p. ej. [Buscador Jurídico SCJN](https://bj.scjn.gob.mx/)): el backend prueba "
        "Chrome instalado, Chromium embebido, modo visible y mezcla Trafilatura + texto del DOM. "
        "Recomendado: `playwright install chrome`. Pueden abrirse ventanas de Chromium unos segundos."
    )

    site_map = load_site_map(ALMACEN_DIR)
    cfg_mapa = load_config_mapa(ALMACEN_DIR)
    if cfg_mapa:
        st.info(
            "Hay **`almacen/config_mapa.json`** (mapeo canónico del Buscador). "
            "Con URL del Buscador Jurídico y palabra clave, se construye la URL de resultados con la **fuente** e **índice** confirmados."
        )
    if site_map:
        st.info(
            "Hay un **mapa legado** (`almacen/mapa_scjn.json`). "
            "Si no aplica `config_mapa`, se usa como respaldo para la URL de resultados."
        )
    if cfg_mapa or site_map:
        kw_map = st.text_input(
            "Palabra clave (opcional, para aplicar mapa / Buscador Jurídico)",
            placeholder="ej. amparo, tesis…",
            key="extraer_kw_mapa",
        )
    else:
        kw_map = ""

    if "ultimo_texto_extraido" not in st.session_state:
        st.session_state.ultimo_texto_extraido = ""

    if modo == MODE_URL:
        url_in = st.text_input("URL directa", placeholder="https://ejemplo.com/articulo")
        if st.button("Extraer y guardar", type="primary"):
            if not url_in or not url_in.strip():
                st.warning("Indica una URL.")
            else:
                u = url_in.strip()
                if not u.startswith(("http://", "https://")):
                    u = "https://" + u
                target = resolve_extraction_url_with_config(u, kw_map or None, ALMACEN_DIR)
                if target != u:
                    log_step(f"URL ajustada por mapa: {target!r}")
                    st.caption(f"Extrayendo resultados: `{target}`")
                try:
                    with st.spinner("Extrayendo..."):
                        texto, titulo, final_u = extract_with_playwright(target)
                        guardar_json_almacen(final_u, texto, titulo)
                        st.session_state.ultimo_texto_extraido = texto
                        st.success("Guardado en ./almacen")
                except Exception as e:
                    log_step(f"Error: {e!r}")
                    st.error(f"Error al extraer: {format_user_error(e)}")
    else:
        q = st.text_input("Palabras de búsqueda", placeholder="ej. python streamlit tutorial")
        if st.button("Buscar, abrir primer resultado y guardar", type="primary"):
            if not q or not q.strip():
                st.warning("Escribe términos de búsqueda.")
            else:
                try:
                    with st.spinner("Buscando y extrayendo..."):
                        first = search_bing_first_result_url(q.strip())
                        target = first
                        if not target:
                            q_enc = urllib.parse.quote_plus(q.strip())
                            target = f"https://www.bing.com/search?q={q_enc}"
                        texto, titulo, final_u = extract_with_playwright(target)
                        guardar_json_almacen(final_u, texto, titulo)
                        st.session_state.ultimo_texto_extraido = texto
                        st.success("Guardado en ./almacen")
                except Exception as e:
                    log_step(f"Error: {e!r}")
                    st.error(f"Error: {format_user_error(e)}")

    if st.session_state.ultimo_texto_extraido:
        st.subheader("Vista previa del último texto extraído")
        st.text_area(
            "Contenido",
            value=st.session_state.ultimo_texto_extraido[:8000],
            height=240,
            disabled=True,
            key="preview_extraido",
        )


def render_mapear() -> None:
    st.header("Mapear sitio")
    if st.button("← Volver al inicio", key="vol_map"):
        ir(PAGE_HOME)
        st.rerun()

    tab_canon, tab_legacy = st.tabs(["Mapear sitio (Buscador SCJN)", "Mapeo manual (portada)"])

    with tab_canon:
        st.markdown(
            "Elige **fuente** e **índice** del [Buscador Jurídico](https://bj.scjn.gob.mx/) y una palabra clave. "
            "Se abre la **URL canónica** de resultados, se captura la zona de listado y, al confirmar, se guarda "
            "`almacen/config_mapa.json` para que **Extraer** y la descarga masiva usen esos parámetros."
        )
        fuentes = listar_fuentes_ui()
        fuente_ui = st.selectbox("Fuente", fuentes, key="map_bj_fuente")
        opts = indices_para_fuente(fuente_ui)
        slug_list = [s for s, _ in opts]
        indice_slug = st.selectbox(
            "Índice",
            slug_list,
            format_func=lambda s: etiqueta_indice_por_slug(fuente_ui, s),
            key="map_bj_indice",
        )
        kw_bj = st.text_input("Palabra clave de prueba", placeholder="pagare", key="map_bj_kw")
        headed_bj = st.checkbox(
            "Mostrar navegador (headed, recomendado para validar)",
            value=True,
            key="map_bj_headed",
        )
        if st.button("Previsualizar resultados (captura)", type="primary", key="map_bj_run"):
            if not kw_bj or not kw_bj.strip():
                st.warning("Escribe una palabra clave de prueba.")
            else:
                try:
                    api = fuente_api_desde_ui(fuente_ui)
                    with st.spinner("Navegando al listado canónico y capturando…"):
                        prev = run_bj_canonical_preview(
                            fuente_api=api,
                            indice=indice_slug,
                            keyword=kw_bj.strip(),
                            headed=headed_bj,
                            log=log_step,
                        )
                        prev["fuente_ui"] = fuente_ui
                        prev["indice_label"] = etiqueta_indice_por_slug(fuente_ui, indice_slug)
                        st.session_state.map_bj_pending = prev
                    st.success(
                        f"Captura en `{TEMP_MAP_PREVIEW_PNG}`. Si ves tarjetas de resultados, el scraper masivo está alineado."
                    )
                except Exception as e:
                    log_step(f"Mapeo canónico: {e!r}")
                    st.error(format_user_error(e))

        pending_bj = st.session_state.map_bj_pending
        if TEMP_MAP_PREVIEW_PNG.is_file():
            st.subheader("Vista del listado (última previsualización)")
            st.image(str(TEMP_MAP_PREVIEW_PNG), use_container_width=True)

        if pending_bj:
            st.caption(f"URL canónica: `{pending_bj.get('canonical_url', '')}`")
            if pending_bj.get("pdf_directo"):
                st.warning(
                    "Se detectó **enlace o respuesta PDF** en el resultado: la descarga masiva usará **descarga directa** "
                    "de PDF en lugar de texto del listado."
                )
            for note in pending_bj.get("external_notes") or []:
                st.warning(note)
            if st.button("Confirmar mapeo de fuente", type="primary", key="map_bj_confirm"):
                api = fuente_api_desde_ui(pending_bj.get("fuente_ui") or fuente_ui)
                ind = pending_bj.get("indice") or indice_slug
                payload = build_config_mapa_payload(
                    fuente_ui=pending_bj.get("fuente_ui") or fuente_ui,
                    fuente_api=api,
                    indice=ind,
                    indice_label=pending_bj.get("indice_label")
                    or etiqueta_indice_por_slug(pending_bj.get("fuente_ui") or fuente_ui, ind),
                    keyword_sample=str(pending_bj.get("keyword_sample") or kw_bj or ""),
                    pdf_directo=bool(pending_bj.get("pdf_directo")),
                    canonical_url=str(pending_bj.get("canonical_url") or ""),
                )
                save_config_mapa(ALMACEN_DIR, payload)
                st.session_state.map_bj_pending = None
                st.success(
                    "Guardado en `almacen/config_mapa.json`. **Extraer** y la descarga masiva leerán esta configuración."
                )
                st.rerun()
        elif load_config_mapa(ALMACEN_DIR):
            st.caption(
                "Ya existe `config_mapa.json`. Puedes previsualizar de nuevo y confirmar para sobrescribirlo."
            )

    with tab_legacy:
        st.markdown(
            "Indica la URL inicial (p. ej. portada del [Buscador Jurídico](https://bj.scjn.gob.mx/)) "
            "y una palabra clave. El navegador intentará **buscar**, capturará la pantalla y podrás **confirmar** "
            "el mapeo para que **Extraer** use la URL de resultados automáticamente (`mapa_scjn.json`)."
        )

        url_map = st.text_input(
            "URL del sitio",
            value="https://bj.scjn.gob.mx/",
            key="map_url",
        )
        kw_map = st.text_input("Palabra clave de prueba", placeholder="amparo", key="map_kw")

        with st.expander("Selectores CSS opcionales (avanzado)"):
            css_in = st.text_input(
                "Campo de búsqueda (CSS)",
                placeholder="vacío = detección automática",
                key="map_css_in",
            )
            css_sub = st.text_input(
                "Botón enviar (CSS)",
                placeholder="vacío = detección automática o Enter",
                key="map_css_sub",
            )

        col_a, col_b = st.columns(2)
        with col_a:
            run_map = st.button("Ejecutar búsqueda y capturar pantalla", type="primary", key="map_run")
        with col_b:
            headed = st.checkbox("Mostrar navegador (headed)", key="map_headed")

        if run_map:
            if not kw_map or not kw_map.strip():
                st.warning("Escribe una palabra clave de prueba.")
            else:
                try:
                    with st.spinner("Navegando y capturando…"):
                        payload = run_site_mapping_flow(
                            start_url=url_map.strip(),
                            keyword=kw_map.strip(),
                            search_input_css=css_in.strip() or None,
                            search_submit_css=css_sub.strip() or None,
                            headless=not headed,
                            log=log_step,
                        )
                        st.session_state.map_pending = payload
                    st.success(f"Captura guardada en `{TEMP_MAP_PNG}`. Revisa la imagen y confirma si ves resultados.")
                except Exception as e:
                    log_step(f"Mapeo: {e!r}")
                    st.error(format_user_error(e))

        if TEMP_MAP_PNG.is_file():
            st.subheader("Vista del robot (última captura — manual)")
            st.image(str(TEMP_MAP_PNG), use_container_width=True)

        pending = st.session_state.map_pending
        if pending:
            st.json(pending)
            if st.button("Confirmar mapeo (legado)", type="primary", key="map_confirm"):
                save_site_map(ALMACEN_DIR, pending)
                st.session_state.map_pending = None
                st.success(
                    "Guardado en `almacen/mapa_scjn.json`. En **Extraer** → URL directa, usa la misma URL base y la palabra clave para ir al listado."
                )
                st.rerun()
        elif load_site_map(ALMACEN_DIR):
            st.caption("Ya existe un mapa legado guardado. Puedes ejecutar una nueva captura y confirmar para sobrescribirlo.")


def render_entrenar() -> None:
    st.header("Entrenar")
    if st.button("← Volver al inicio", key="vol_ent"):
        ir(PAGE_HOME)
        st.rerun()

    if "texto_entrenar" not in st.session_state:
        st.session_state.texto_entrenar = ""

    opcion = st.radio(
        "Origen del texto",
        ["Opción A: Usar Almacén", "Opción B: Subir información"],
        key="radio_entrenar_origen",
    )

    if opcion.startswith("Opción A"):
        st.caption(
            "Los JSON de descarga masiva del Buscador incluyen **fuente** e **índice**; "
            "puedes reconocerlos por el nombre `YYYY-MM-DD_bj_<índice>_<registro>.json`."
        )
        archivos = listar_almacen_json_entrenamiento()
        if not archivos:
            st.info("No hay archivos JSON en ./almacen. Extrae contenido primero.")
        else:
            nombres = [p.name for p in archivos]
            elegido = st.selectbox("Selecciona un archivo", nombres, key="sel_almacen_train")
            path = ALMACEN_DIR / elegido
            if st.button("Cargar texto para revisión", type="primary"):
                log_step(f"Leyendo JSON del almacén: {path}")
                data = json.loads(path.read_text(encoding="utf-8"))
                st.session_state.texto_entrenar = data.get("texto") or ""
                st.success("Texto cargado.")
    else:
        up = st.file_uploader("Sube PDF o TXT", type=["pdf", "txt"], key="up_train")
        if up is not None and st.button("Leer archivo subido", type="primary"):
            log_step(f"Leyendo archivo subido: {up.name!r} ({up.size} bytes)")
            suffix = Path(up.name).suffix.lower()
            raw = up.getvalue()
            if suffix == ".txt":
                st.session_state.texto_entrenar = raw.decode("utf-8", errors="replace")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(raw)
                    tmp_path = f.name
                try:
                    st.session_state.texto_entrenar = leer_pdf(Path(tmp_path))
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            st.success("Archivo leído.")

    st.subheader("Texto que se usaría para entrenar (confirmación)")
    vista = st.session_state.texto_entrenar
    if vista:
        st.text_area("Vista previa", value=vista[:12000], height=320, key="ta_train")
    else:
        st.caption("Carga un JSON del almacén o sube un archivo para ver el texto aquí.")


def render_almacen() -> None:
    st.header("📦 Almacén")
    if st.button("← Volver al inicio", key="vol_alm"):
        ir(PAGE_HOME)
        st.rerun()

    archivos = listar_almacen_json()
    if not archivos:
        st.info("Aún no hay archivos en ./almacen.")
        return

    rows = []
    for p in archivos:
        st_sz = p.stat().st_size
        rows.append({"Archivo": p.name, "Tamaño (bytes)": st_sz})

    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Vista previa rápida")
    sel = st.selectbox("Archivo", [p.name for p in archivos], key="alm_sel_preview")
    path = ALMACEN_DIR / sel
    if st.button("Mostrar vista previa", key="alm_btn_prev"):
        st.session_state.preview_almacen_file = sel

    if st.session_state.preview_almacen_file:
        p = ALMACEN_DIR / st.session_state.preview_almacen_file
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            txt = data.get("texto") or ""
            st.text_area(
                "Contenido",
                value=preview_snippet(txt, 2000),
                height=200,
                key="alm_ta",
            )
        except Exception as e:
            st.error(f"No se pudo leer el JSON: {format_user_error(e)}")


def main() -> None:
    init_session()
    pagina = st.session_state.pagina
    if pagina == PAGE_HOME:
        render_home()
    elif pagina == PAGE_EXTRAER:
        render_extraer()
    elif pagina == PAGE_ENTRENAR:
        render_entrenar()
    elif pagina == PAGE_ALMACEN:
        render_almacen()
    elif pagina == PAGE_MAPEAR:
        render_mapear()
    else:
        st.session_state.pagina = PAGE_HOME
        st.rerun()


if __name__ == "__main__":
    main()
