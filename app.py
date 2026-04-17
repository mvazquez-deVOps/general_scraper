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

# --- Rutas y constantes ---
BASE_DIR = Path(__file__).resolve().parent
ALMACEN_DIR = BASE_DIR / "almacen"
ALMACEN_DIR.mkdir(parents=True, exist_ok=True)

PAGE_HOME = "inicio"
PAGE_EXTRAER = "extraer"
PAGE_ENTRENAR = "entrenar"
PAGE_ALMACEN = "almacen"

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


def init_session() -> None:
    if "pagina" not in st.session_state:
        st.session_state.pagina = PAGE_HOME
    if "preview_almacen_file" not in st.session_state:
        st.session_state.preview_almacen_file = None


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
    c1, c2 = st.columns(2)
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
                try:
                    with st.spinner("Extrayendo..."):
                        texto, titulo, final_u = extract_with_playwright(u)
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
        archivos = listar_almacen_json()
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
    else:
        st.session_state.pagina = PAGE_HOME
        st.rerun()


if __name__ == "__main__":
    main()
