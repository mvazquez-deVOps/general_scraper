"""
Aplicación Streamlit: extracción con Playwright + Trafilatura y vista de entrenamiento.
Los JSON se guardan en la carpeta local ./almacen
"""

from __future__ import annotations

import asyncio
import base64
import html
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

from scjn_tesis import hf_integration
import training_connection

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
AVATAR_DIR = DATA_DIR / "avatars"
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
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


SKIP_ALMACEN_JSON = frozenset({"mapa_scjn.json", "config_mapa.json"})

GPU_TIER_LABELS: dict[str, str] = {
    "bajo": "Bajo - RTX 3090",
    "medio": "Medio - A6000",
    "alto": "Alto - A100",
}
GPU_TIER_TO_RUNPOD: dict[str, str] = {"bajo": "rtx3090", "medio": "a6000", "alto": "a100"}
BOT_GPU_FEEDBACK: dict[str, str] = {
    "bajo": "¡Genial! Con una RTX 3090 voy con calma y voy aprendiendo firme.",
    "medio": "¡Buen equilibrio! El A6000 es un compañero rápido y eficiente.",
    "alto": "¡Excelente elección, con una A100 aprenderé leyes en minutos!",
}

# Núcleo (Model Hub): stats 1–5 (velocidad, memoria VRAM estimada, precisión legal heurística)
MODEL_HUB_PRESETS: dict[str, dict[str, object]] = {
    "gemma2": {
        "label": "Gemma 2 (Google)",
        "repo_id": "google/gemma-2-9b",
        "tagline": "Equilibrado.",
        "speed": 4,
        "memory": 4,
        "legal": 4,
        "demand_rank": 3,
    },
    "llama3": {
        "label": "Llama 3 (Meta)",
        "repo_id": "meta-llama/Meta-Llama-3-8B",
        "tagline": "Potente.",
        "speed": 3,
        "memory": 4,
        "legal": 5,
        "demand_rank": 3,
    },
    "mistral": {
        "label": "Mistral",
        "repo_id": "mistralai/Mistral-7B-v0.3",
        "tagline": "Rápido.",
        "speed": 5,
        "memory": 3,
        "legal": 4,
        "demand_rank": 2,
    },
}

# Avatares fijos (selección tipo videojuego) → subcarpetas de almacén + imagen (data/avatars)
AVATAR_KNOWLEDGE_PROFILES: tuple[dict[str, str], ...] = (
    {
        "id": "avatar:tesis_sjf",
        "nombre": "Tesis SJF",
        "subtitle": "JUXA legaltech",
        "subdir": "tesis",
        "image_sleep": "juxa_mimido.png",
        "image_awake": "juxa_despierto.png",
        "object_position": "50% 35%",
    },
    {
        "id": "avatar:sentencias_sij",
        "nombre": "Sentencias SIJ",
        "subtitle": "Corte / análisis",
        "subdir": "sentencias",
        "image_sleep": "sentencias_sij_mimido.png",
        "image_awake": "sentencias_sij_despierto.png",
        "object_position": "50% 48%",
    },
    {
        "id": "avatar:internacional",
        "nombre": "Internacional",
        "subtitle": "Corpus global",
        "subdir": "internacional",
        "image_sleep": "internacional_mimido.png",
        "image_awake": "internacional_despierto.png",
        "object_position": "50% 45%",
    },
)


def _image_file_to_data_uri(file_path: Path) -> str:
    """PNG/JPG embebido para ``st.markdown(..., unsafe_allow_html=True)``."""
    if not file_path.is_file():
        return ""
    raw = file_path.read_bytes()
    ext = file_path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def resolve_avatar_knowledge_profiles() -> list[dict]:
    """Crea si hace falta ``almacen/tesis``, ``sentencias``, ``internacional`` y cuenta JSON."""
    out: list[dict] = []
    for cfg in AVATAR_KNOWLEDGE_PROFILES:
        path = ALMACEN_DIR / cfg["subdir"]
        path.mkdir(parents=True, exist_ok=True)
        n = len([p for p in path.glob("*.json") if p.name not in SKIP_ALMACEN_JSON])
        out.append({**cfg, "path": path, "n_json": n})
    return out


def power_stats_table_rows(speed: int, memory: int, legal: int) -> list[dict[str, object]]:
    """Filas para tabla de Power Stats (velocidad ⚡, RAM 💾, balanza ⚖️)."""
    sp = max(1, min(5, int(speed)))
    mem = max(1, min(5, int(memory)))
    leg = max(1, min(5, int(legal)))
    return [
        {
            "Métrica": "Velocidad",
            "Indicador": "⚡" * sp + "·" * (5 - sp),
            "Nivel": f"{sp}/5",
        },
        {
            "Métrica": "Memoria",
            "Indicador": "💾" * mem + "·" * (5 - mem),
            "Nivel": f"{mem}/5",
        },
        {
            "Métrica": "Precisión legal",
            "Indicador": "⚖️" * leg + "·" * (5 - leg),
            "Nivel": f"{leg}/5",
        },
    ]


def _render_fase2_stepper(current: int) -> None:
    """Indicador de pasos 1–3 (Avatar → Núcleo → Armor)."""
    current = max(1, min(3, int(current)))
    h = ['<div class="provision-stepper" role="navigation" aria-label="Etapas Fase 2">']
    for n, name in ((1, "Avatar"), (2, "Núcleo"), (3, "Armor")):
        if n < current:
            cls = "provision-step-pill provision-step-done"
        elif n == current:
            cls = "provision-step-pill provision-step-active"
        else:
            cls = "provision-step-pill provision-step-todo"
        h.append(f'<span class="{cls}"><b>{n}</b> {html.escape(name)}</span>')
        if n < 3:
            h.append('<span class="arr" aria-hidden="true">→</span>')
    h.append("</div>")
    st.markdown(PROVISION_CSS + "".join(h), unsafe_allow_html=True)


def _fase2_bump_to_step3() -> None:
    if st.session_state.get("provision_model_choice") == "custom":
        cid = (st.session_state.get("provision_model_custom_id") or "").strip()
        if not cid:
            return
    st.session_state.provision_fase2_step = 3


# Máxima exigencia de cómputo (1–5) aceptada por cada tier RunPod (fine-tuning cuantizado aprox.)
GPU_TIER_MAX_DEMAND: dict[str, int] = {
    "bajo": 3,
    "medio": 4,
    "alto": 5,
}


def _infer_custom_model_demand_rank(model_id: str) -> int:
    s = (model_id or "").lower().replace(" ", "")
    if any(x in s for x in ["70b", "-70-", "72b", "405b", "172b", "161b"]):
        return 5
    if any(x in s for x in ["65b", "34b", "35b", "40b"]):
        return 4
    if any(x in s for x in ["13b", "14b", "15b", "16b", "22b"]):
        return 3
    if any(x in s for x in ["8b", "9b", "7b", "mistral-7b"]):
        return 3
    return 3


def _infer_custom_power_stats(model_id: str) -> tuple[int, int, int]:
    """Velocidad, memoria, precisión legal (1–5) para IDs personalizados según heurística de tamaño."""
    r = _infer_custom_model_demand_rank(model_id)
    if r >= 5:
        return 2, 5, 5
    if r == 4:
        return 3, 5, 4
    if r == 3:
        return 4, 4, 4
    return 4, 3, 4


def _resolve_provision_model_id(choice: str, custom_id: str) -> str:
    if choice == "custom":
        return (custom_id or "").strip()
    preset = MODEL_HUB_PRESETS.get(choice)
    return str(preset["repo_id"]) if preset else ""


def _provision_model_demand_rank(choice: str, custom_id: str) -> int:
    if choice == "custom":
        return _infer_custom_model_demand_rank(custom_id)
    preset = MODEL_HUB_PRESETS.get(choice)
    return int(preset["demand_rank"]) if preset else 3


def _model_gpu_mismatch(tier: str, choice: str, custom_id: str) -> bool:
    need = _provision_model_demand_rank(choice, custom_id)
    cap = GPU_TIER_MAX_DEMAND.get(tier, 3)
    return need > cap


KAWAII_FACE_PAIRS = (
    ("😴", "🤖✨"),
    ("💤", "⚖️💫"),
    ("😪", "📚✨"),
    ("🐱💤", "🦊⚡"),
)

PROVISION_CSS = """
<style>
.provision-kawaii-wrap {
  font-family: ui-sans-serif, system-ui, sans-serif;
}
.provision-kawaii-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  justify-content: center;
  align-items: stretch;
  margin: 0.5rem 0 1.25rem 0;
}
.provision-kawaii-bot {
  border: 3px solid #45475a;
  border-radius: 18px;
  padding: 14px 12px 10px 12px;
  text-align: center;
  background: linear-gradient(165deg, #1e1e2e 0%, #313244 55%, #181825 100%);
  min-width: 118px;
  max-width: 160px;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
  cursor: default;
}
.provision-kawaii-bot--illustrated {
  min-width: 198px;
  max-width: 220px;
}
.provision-avatar-skin {
  position: relative;
  width: 100%;
  margin: 0 auto 10px;
  border-radius: 16px;
  overflow: hidden;
  background: linear-gradient(180deg, #1e1e2e, #11111b);
  border: 2px solid #3d3d52;
  line-height: 0;
}
/* Un solo PNG (Sentencias / Internacional): atenuar hasta hover o selección */
.provision-avatar-skin--single img {
  width: 100%;
  height: 200px;
  object-fit: cover;
  object-position: var(--avatar-obj, 50% 40%);
  display: block;
  filter: brightness(0.7) saturate(0.8) contrast(0.95);
  transform: scale(0.99);
  transition: filter 0.35s ease, transform 0.35s ease, opacity 0.35s ease;
  box-shadow: inset 0 0 0 1px rgba(0,0,0,0.2);
}
.provision-kawaii-bot--illustrated:hover .provision-avatar-skin--single img,
.provision-kawaii-bot--illustrated.selected .provision-avatar-skin--single img {
  filter: brightness(1.06) saturate(1.18) contrast(1) drop-shadow(0 0 12px rgba(120, 200, 255, 0.42));
  transform: scale(1.01);
}
/* Par mimido + despierto: mimido por defecto si no está elegido; despierto en hover o al elegir */
.provision-avatar-skin--dual {
  height: 200px;
  overflow: hidden;
}
.provision-avatar-skin--dual .layer-sleep,
.provision-avatar-skin--dual .layer-awake {
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: var(--avatar-obj, 50% 40%);
  border-radius: 15px;
  display: block;
  pointer-events: none;
  transition: opacity 0.5s cubic-bezier(0.4, 0, 0.2, 1), filter 0.5s ease, transform 0.5s ease;
  backface-visibility: hidden;
  will-change: opacity, transform, filter;
}
.provision-avatar-skin--dual .layer-sleep {
  opacity: 1;
  z-index: 1;
  filter: saturate(0.9);
}
/* mimido: visible mientras el avatar no está elegido ni hover */
.provision-avatar-skin--dual .layer-awake {
  opacity: 0;
  z-index: 2;
  filter: blur(0.3px) brightness(0.82) saturate(0.9);
  transform: scale(0.992);
}
/* despertar: mouse sobre la tarjeta, o ya elegido */
.provision-kawaii-bot--illustrated:hover .provision-avatar-skin--dual .layer-awake,
.provision-kawaii-bot--illustrated.selected .provision-avatar-skin--dual .layer-awake {
  opacity: 1;
  filter: blur(0) brightness(1) saturate(1.12) drop-shadow(0 0 8px rgba(100, 180, 255, 0.25));
  transform: scale(1.008);
}
.provision-kawaii-bot--illustrated:hover .provision-avatar-skin--dual .layer-sleep,
.provision-kawaii-bot--illustrated.selected .provision-avatar-skin--dual .layer-sleep {
  opacity: 0;
  filter: blur(0.2px) brightness(0.65);
  transform: scale(0.97);
}
.provision-avatar-dim {
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  bottom: 0;
  pointer-events: none;
  opacity: 0.5;
  background: linear-gradient(180deg, rgba(0,0,0,0.15), rgba(8,0,20,0.12));
  transition: opacity 0.3s ease;
  border-radius: 15px;
  z-index: 3;
}
.provision-avatar-skin--single .provision-avatar-dim { z-index: 1; }
.provision-avatar-skin--dual .provision-avatar-dim { display: none; }
.provision-kawaii-bot--illustrated:hover .provision-avatar-dim,
.provision-kawaii-bot--illustrated.selected .provision-avatar-dim {
  opacity: 0;
}
.provision-kawaii-bot:hover {
  transform: translateY(-6px) scale(1.04);
  box-shadow: 0 12px 28px rgba(137, 180, 250, 0.38);
  border-color: #89b4fa;
}
.provision-kawaii-bot.selected {
  border-color: #fab387;
  box-shadow: 0 0 0 4px rgba(250, 179, 135, 0.45);
}
.provision-kawaii-bot .face-sleep {
  font-size: 3.2rem;
  line-height: 1.1;
  display: inline-block;
}
.provision-kawaii-bot .face-awake {
  font-size: 3.2rem;
  line-height: 1.1;
  display: none;
}
.provision-kawaii-bot:hover .face-sleep,
.provision-kawaii-bot.selected .face-sleep {
  display: none;
}
.provision-kawaii-bot:hover .face-awake,
.provision-kawaii-bot.selected .face-awake {
  display: inline-block;
}
.provision-kawaii-name {
  font-size: 0.82rem;
  font-weight: 600;
  color: #cdd6f4;
  margin-top: 6px;
  word-break: break-word;
}
.provision-kawaii-meta {
  font-size: 0.72rem;
  color: #a6adc8;
  margin-top: 4px;
}
.provision-kawaii-bot.training {
  border-color: #a6e3a1;
  animation: provision-kawaii-pulse 1.6s ease-in-out infinite alternate;
}
@keyframes provision-kawaii-pulse {
  from { box-shadow: 0 0 6px rgba(166, 227, 161, 0.35); }
  to { box-shadow: 0 0 22px rgba(166, 227, 161, 0.9); }
}
.provision-kawaii-bot.training .face-sleep { display: none; }
.provision-kawaii-bot.training .face-awake { display: inline-block; }
.provision-power-bar {
  height: 10px;
  border-radius: 6px;
  background: linear-gradient(90deg, #89dceb, #cba6f7);
  margin-top: 4px;
}
.provision-kawaii-worried {
  border: 3px solid #eba0ac !important;
  border-radius: 18px;
  padding: 14px 16px;
  background: linear-gradient(165deg, #2d1f2a 0%, #3d2c35 100%);
  text-align: center;
  max-width: 420px;
  margin: 12px auto;
}
.provision-stepper {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 0.4rem 0.6rem;
  margin: 0.5rem 0 1.15rem;
  font-size: 0.88rem;
  font-family: ui-sans-serif, system-ui, sans-serif;
}
.provision-stepper .arr { color: #6c7086; user-select: none; }
.provision-step-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.4rem 0.85rem;
  border-radius: 999px;
  border: 2px solid #45475a;
  color: #a6adc8;
  background: rgba(30, 30, 46, 0.5);
}
.provision-step-pill b { color: #7f849c; }
.provision-step-active {
  border-color: #89b4fa;
  color: #dce0ee;
  background: rgba(137, 180, 250, 0.15);
  box-shadow: 0 0 0 1px rgba(137, 180, 250, 0.25);
}
.provision-step-active b { color: #89b4fa; }
.provision-step-done {
  border-color: #a6e3a1;
  color: #a6e3a1;
  background: rgba(166, 227, 161, 0.08);
}
.provision-step-done b { color: #a6e3a1; }
.provision-step-todo { opacity: 0.55; }
</style>
"""


def render_provision_training_monitor(repo_id: str, *, sleep_face: str, wake_face: str) -> None:
    """Lee ``training_logs/progress.json`` y ``events.jsonl`` del dataset en el Hub."""
    prog = hf_integration.fetch_training_progress(repo_id)
    pct = hf_integration.progress_fraction(prog)
    events = hf_integration.fetch_training_events_tail(repo_id, max_lines=15)

    training_active = prog is None or prog.get("status") not in ("completed", "stopped")

    if training_active:
        mini = f"""<div class="provision-kawaii-bot training" style="max-width:280px;margin:8px auto;">
  <span class="face-sleep">{sleep_face}</span>
  <span class="face-awake">{wake_face}</span>
  <div class="provision-kawaii-name">Entrenando…</div>
  <div class="provision-kawaii-meta">Leyendo logs del Hub</div>
</div>"""
        st.markdown(PROVISION_CSS + mini, unsafe_allow_html=True)

    if prog and prog.get("status") == "completed":
        label = f"{wake_face} ¡Entrenamiento completado!"
    elif prog:
        step = prog.get("step", 0)
        mx = prog.get("max_steps")
        loss = prog.get("loss")
        tail = ""
        if loss is not None:
            try:
                tail = f" · loss {float(loss):.4f}"
            except (TypeError, ValueError):
                tail = ""
        label = f"{wake_face} Entrenando… paso {step}/{mx}{tail}"
    else:
        label = f"{wake_face} Esperando logs del entrenador en Hugging Face (`training_logs/progress.json`)…"

    st.progress(pct, text=label)

    if events:
        with st.expander("Últimos eventos del entrenamiento (log estilo Hugging Face)", expanded=False):
            st.json(events[-8:])


def init_session() -> None:
    hf_integration.load_dotenv_from_project()
    if "pagina" not in st.session_state:
        st.session_state.pagina = PAGE_HOME
    if "preview_almacen_file" not in st.session_state:
        st.session_state.preview_almacen_file = None
    if "map_pending" not in st.session_state:
        st.session_state.map_pending = None
    if "map_bj_pending" not in st.session_state:
        st.session_state.map_bj_pending = None
    if "provision_profile_id" not in st.session_state:
        st.session_state.provision_profile_id = "avatar:tesis_sjf"
    if "provision_compute_tier" not in st.session_state:
        st.session_state.provision_compute_tier = "medio"
    if "provision_last_repo" not in st.session_state:
        st.session_state.provision_last_repo = None
    if "provision_monitor_bot_idx" not in st.session_state:
        st.session_state.provision_monitor_bot_idx = 0
    if "provision_fase2_step" not in st.session_state:
        st.session_state.provision_fase2_step = 1
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
    st.header("Entrenar — Fase 2 · Aprovisionamiento")
    if st.button("← Volver al inicio", key="vol_ent"):
        ir(PAGE_HOME)
        st.rerun()

    if "texto_entrenar" not in st.session_state:
        st.session_state.texto_entrenar = ""

    st.markdown(
        "### Fase 2 · Equipo de aprovisionamiento\n"
        "Sigue el orden: **1 Avatar** → **2 Núcleo** (Model Hub) → **3 Armor** (RunPod, GPU y repositorio)."
    )
    st.markdown(PROVISION_CSS + '<div class="provision-kawaii-wrap"></div>', unsafe_allow_html=True)

    perfiles = resolve_avatar_knowledge_profiles()
    sel_id = st.session_state.provision_profile_id
    perfil_early = next((p for p in perfiles if p["id"] == sel_id), None)
    fase2_step = int(st.session_state.get("provision_fase2_step", 1))
    if fase2_step >= 2 and not perfil_early:
        st.session_state.provision_fase2_step = 1
        fase2_step = 1
        st.rerun()
    _render_fase2_stepper(fase2_step)

    sel_id = st.session_state.provision_profile_id
    perfil_actual = next((p for p in perfiles if p["id"] == sel_id), None)

    if fase2_step == 1:
        st.markdown("#### 1 · Selección de Avatar")
        st.caption("Tres bots kawaii — hover despierta al bot. Cada uno corresponde a una carpeta del almacén.")
        avatar_cols = st.columns(3)
        for idx, prof in enumerate(perfiles):
            sel_class = "selected" if sel_id == prof["id"] else ""
            nombre_safe = html.escape(str(prof["nombre"]))
            subtitle_safe = html.escape(str(prof["subtitle"]))
            subdir_safe = html.escape(str(prof["subdir"]))
            obj_pos = html.escape(str(prof.get("object_position") or "50% 50%"))
            card_html = ""
            s_fn, a_fn = prof.get("image_sleep"), prof.get("image_awake")
            if s_fn and a_fn:
                uri_s = _image_file_to_data_uri(AVATAR_DIR / str(s_fn))
                uri_a = _image_file_to_data_uri(AVATAR_DIR / str(a_fn))
                if uri_s and uri_a:
                    card_html = f"""
<div class="provision-kawaii-bot provision-kawaii-bot--illustrated {sel_class}">
  <div class="provision-avatar-skin provision-avatar-skin--dual" style="--avatar-obj: {obj_pos};" title="Mimido: elige o pasa el mouse para despertar">
    <img class="layer-sleep" src="{uri_s}" alt="" aria-hidden="true" />
    <img class="layer-awake" src="{uri_a}" alt="{nombre_safe}" />
  </div>
  <div class="provision-kawaii-name">{nombre_safe}</div>
  <div class="provision-kawaii-meta">{subtitle_safe}</div>
  <div class="provision-kawaii-meta">`almacen/{subdir_safe}/` · {prof["n_json"]} JSON</div>
</div>"""
            if not card_html:
                img_file = str(prof.get("image") or "")
                if not img_file:
                    img_file = "juxa_legaltech.png"
                data_uri = _image_file_to_data_uri(AVATAR_DIR / img_file)
                if data_uri:
                    card_html = f"""
<div class="provision-kawaii-bot provision-kawaii-bot--illustrated {sel_class}">
  <div class="provision-avatar-skin provision-avatar-skin--single" style="--avatar-obj: {obj_pos};">
    <div class="provision-avatar-dim" title="Reposo: hover o elige el avatar"></div>
    <img src="{data_uri}" alt="{nombre_safe}" />
  </div>
  <div class="provision-kawaii-name">{nombre_safe}</div>
  <div class="provision-kawaii-meta">{subtitle_safe}</div>
  <div class="provision-kawaii-meta">`almacen/{subdir_safe}/` · {prof["n_json"]} JSON</div>
</div>"""
            if not card_html:
                sleep_e, wake_e = KAWAII_FACE_PAIRS[idx % len(KAWAII_FACE_PAIRS)]
                card_html = f"""
<div class="provision-kawaii-bot {sel_class}">
  <span class="face-sleep" title="zzz…">{sleep_e}</span>
  <span class="face-awake" title="¡Despierto!">{wake_e}</span>
  <div class="provision-kawaii-name">{nombre_safe}</div>
  <div class="provision-kawaii-meta">Coloca PNG en `data/avatars/`</div>
  <div class="provision-kawaii-meta">`almacen/{subdir_safe}/` · {prof["n_json"]} JSON</div>
</div>"""
            with avatar_cols[idx]:
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("Elegir", key=f"provision_avatar_pick_{prof['id']}", use_container_width=True):
                    st.session_state.provision_profile_id = prof["id"]
                    st.session_state.provision_fase2_step = 2
                    st.rerun()
        if perfil_actual and st.button("Usar el avatar preseleccionado y continuar al núcleo", key="fase2_use_default_avatar_to_2"):
            st.session_state.provision_fase2_step = 2
            st.rerun()
    else:
        with st.expander("1 · Selección de Avatar — listo", expanded=False):
            if perfil_actual:
                st.markdown(
                    f"**{perfil_actual['nombre']}** — `{perfil_actual['path']}` · {perfil_actual['n_json']} JSON"
                )
            if st.button("Cambiar avatar", key="fase2_back_to_avatar"):
                st.session_state.provision_fase2_step = 1
                st.rerun()

    if fase2_step == 2 or fase2_step == 3:
        perfil_actual = next((p for p in perfiles if p["id"] == st.session_state.provision_profile_id), None)

    _fmt_model = (
        lambda k: f"{MODEL_HUB_PRESETS[k]['label']} (`{MODEL_HUB_PRESETS[k]['repo_id']}`) — {MODEL_HUB_PRESETS[k]['tagline']}"
        if k in MODEL_HUB_PRESETS
        else k
    )
    if fase2_step == 2:
        st.markdown("#### 2 · Catálogo de Núcleos (Model Hub)")
        with st.container(border=True):
            st.caption(
                "Modelos base en Hugging Face para fine-tuning (PEFT). "
                "Al elegir otro núcleo o pulsar el botón, pasarás a Armor (paso 3)."
            )
            st.radio(
                "Catálogo de núcleos",
                options=["gemma2", "llama3", "mistral", "custom"],
                format_func=lambda k: _fmt_model(k) if k != "custom" else "Personalizado (cualquier ID de Hugging Face)",
                horizontal=True,
                key="provision_model_choice",
                on_change=_fase2_bump_to_step3,
            )
            if st.session_state.get("provision_model_choice") == "custom":
                st.text_input(
                    "ID del modelo (org/nombre)",
                    placeholder="ej. meta-llama/Meta-Llama-3-70B-Instruct",
                    key="provision_model_custom_id",
                )
            mchoice = st.session_state.get("provision_model_choice", "gemma2")
            custom_mid = (st.session_state.get("provision_model_custom_id") or "")
            if not isinstance(custom_mid, str):
                custom_mid = ""
            eff_id2 = _resolve_provision_model_id(mchoice, custom_mid)
            if mchoice in MODEL_HUB_PRESETS:
                sp2, mem2, leg2 = (
                    int(MODEL_HUB_PRESETS[mchoice]["speed"]),
                    int(MODEL_HUB_PRESETS[mchoice]["memory"]),
                    int(MODEL_HUB_PRESETS[mchoice]["legal"]),
                )
            else:
                sp2, mem2, leg2 = _infer_custom_power_stats(custom_mid or "user/model")

            st.markdown("**Power stats**")
            st.dataframe(
                power_stats_table_rows(sp2, mem2, leg2),
                hide_index=True,
                use_container_width=True,
            )
            st.caption("**BASE_MODEL activo:** " + (f"`{html.escape(eff_id2)}`" if eff_id2 else "—"))
        c1, _c2 = st.columns(2)
        with c1:
            if st.button("Siguiente: Armor & Power (RunPod)", type="primary", key="fase2_btn_nucleo_to_armor"):
                mc = st.session_state.get("provision_model_choice", "gemma2")
                cus = (st.session_state.get("provision_model_custom_id") or "").strip()
                if mc == "custom" and not cus:
                    st.warning("En **Personalizado** indica el ID del modelo en Hugging Face antes de continuar.")
                else:
                    st.session_state.provision_fase2_step = 3
                    st.rerun()
        st.caption("Los modelos preestablecidos del catálogo avanzan al paso 3 al **cambiar** la selección. Con **Personalizado**, completa el ID y pulsa **Siguiente**.")
    elif fase2_step == 3:
        with st.expander("2 · Núcleo (Model Hub) — listo", expanded=False):
            mchoice_r = st.session_state.get("provision_model_choice", "gemma2")
            cm = (st.session_state.get("provision_model_custom_id") or "")
            eid = _resolve_provision_model_id(mchoice_r, cm)
            st.markdown(f"Núcleo: **{html.escape(_fmt_model(mchoice_r) if mchoice_r != 'custom' else (cm or '—'))}**  \n`{html.escape(eid or '—')}`")
            if st.button("Cambiar núcleo (volver al paso 2)", key="fase2_back_to_nucleo"):
                st.session_state.provision_fase2_step = 2
                st.rerun()

    model_choice = st.session_state.get("provision_model_choice", "gemma2")
    if model_choice not in ("gemma2", "llama3", "mistral", "custom"):
        model_choice = "gemma2"
    custom_model_id = st.session_state.get("provision_model_custom_id") or ""
    if not isinstance(custom_model_id, str):
        custom_model_id = ""
    eff_id = _resolve_provision_model_id(model_choice, custom_model_id)
    if model_choice in MODEL_HUB_PRESETS:
        sp, mem, leg = (
            int(MODEL_HUB_PRESETS[model_choice]["speed"]),
            int(MODEL_HUB_PRESETS[model_choice]["memory"]),
            int(MODEL_HUB_PRESETS[model_choice]["legal"]),
        )
    else:
        sp, mem, leg = _infer_custom_power_stats(custom_model_id or "user/model")
    if fase2_step == 1:
        st.caption("Pulsa **Elegir** o **Usar el avatar preseleccionado** en el paso 1 para abrir el catálogo de núcleos (paso 2).")
    elif fase2_step == 2:
        st.caption("Paso 2: el **núcleo**; en **Siguiente** o al cambiar de modelo (excepto con ID personalizado vacío) pasas a Armor (paso 3).")

    if fase2_step == 3:
        st.markdown("#### 3 · Armor & Power (RunPod)")
        with st.container(border=True):
            st.caption(
                "Repositorio dataset en el Hub y GPU para el Pod. Une visualmente armadura (hardware) y despliegue."
            )
            tier = st.radio(
                "Selector de GPU · aprovisionamiento",
                options=list(GPU_TIER_LABELS.keys()),
                format_func=lambda k: GPU_TIER_LABELS[k],
                horizontal=True,
                key="provision_gpu_tier",
            )
            st.session_state.provision_compute_tier = tier
            repo_hf = st.text_input(
                "Repositorio destino del dataset (Hugging Face Hub)",
                placeholder="usuario/mi-dataset-legal",
                key="provision_repo_hf",
                help="Dataset privado: fuente de verdad para el entrenamiento.",
            )

            if _model_gpu_mismatch(tier, model_choice, custom_model_id) and eff_id:
                wface = "😟"
                if perfil_actual:
                    wface = KAWAII_FACE_PAIRS[perfiles.index(perfil_actual) % len(KAWAII_FACE_PAIRS)][1]
                st.markdown(
                    PROVISION_CSS
                    + f"""
<div class="provision-kawaii-worried">
  <div style="font-size:2.8rem;line-height:1;">{wface}</div>
  <div style="color:#f5c2e7;font-weight:600;margin-top:10px;line-height:1.35;">
    ¡Cuidado! Ese modelo es muy pesado para esta armadura. Necesitamos más poder.
  </div>
</div>""",
                    unsafe_allow_html=True,
                )

            if perfil_actual:
                wake_e = KAWAII_FACE_PAIRS[perfiles.index(perfil_actual) % len(KAWAII_FACE_PAIRS)][1]
                msg = BOT_GPU_FEEDBACK.get(tier, "")
                st.markdown(
                    f"<div style='margin-top:12px;padding:12px 14px;border-radius:12px;"
                    f"background:rgba(137,180,250,0.12);border:1px solid #89b4fa;font-size:1.05rem;'>"
                    f"{wake_e} &nbsp; {msg}</div>",
                    unsafe_allow_html=True,
                )

    deploy = False
    if fase2_step == 3:
        deploy = st.button("Iniciar Aprovisionamiento", type="primary", key="provision_deploy_btn")
    else:
        st.caption("**Iniciar aprovisionamiento** se habilita al llegar al paso 3 (Avatar → Núcleo → Armor).")

    if deploy:
        if not perfil_actual:
            st.error("Selecciona un Avatar antes de desplegar.")
        elif perfil_actual["n_json"] == 0:
            st.error(
                f"No hay JSON en `{perfil_actual['path']}`. Descarga extractos en esa carpeta e inténtalo de nuevo."
            )
        elif not (repo_hf or "").strip():
            st.error("Indica el nombre del repositorio en Hugging Face.")
        elif model_choice == "custom" and not (custom_model_id or "").strip():
            st.error("En Personalizado debes pegar el ID del modelo en Hugging Face.")
        else:
            repo_clean = repo_hf.strip()
            effective_base_model = _resolve_provision_model_id(model_choice, custom_model_id)
            gpu_code = GPU_TIER_TO_RUNPOD.get(tier, "a6000")
            idx_bot = perfiles.index(perfil_actual) if perfil_actual in perfiles else 0
            face_bot = KAWAII_FACE_PAIRS[idx_bot % len(KAWAII_FACE_PAIRS)][1]

            try:
                # Mensaje inicial del bot al pulsar Iniciar (antes de Hub / entrenamiento).
                launch_msg = training_connection.BOT_MESSAGE_HF_CONNECT

                with st.status(f"{face_bot} {launch_msg}", expanded=True) as status:

                    def _provision_phase(name: str) -> None:
                        if name == "hub_push_sample":
                            status.update(label=f"{face_bot} Subiendo 10 JSON de prueba al Hub...", state="running")
                        elif name == "local_train":
                            short_m = training_connection.LOCAL_TRIAL_BASE_MODEL.split("/")[-1]
                            status.update(
                                label=f"{face_bot} Entrenamiento local ({short_m}, CPU/GPU)...",
                                state="running",
                            )
                        elif name == "hub_push_full":
                            status.update(label=f"{face_bot} Subiendo dataset completo al Hub...", state="running")
                        elif name == "runpod_pod":
                            status.update(label=f"{face_bot} Creando Pod RunPod (GPU)...", state="running")
                        elif name == "runpod_train":
                            status.update(
                                label=f"{face_bot} Preparando comando remoto (dataset desde HF)...",
                                state="running",
                            )

                    status.update(label=f"{face_bot} Arrancando pipeline...", state="running")
                    bundle = training_connection.run_training_pipeline(
                        repo_id=repo_clean,
                        source_dir=perfil_actual["path"],
                        gpu_code=gpu_code,
                        ui_base_model_id=effective_base_model,
                        log=log_step,
                        on_phase=_provision_phase,
                    )
                    status.update(label="¡Cadena completada!", state="complete")
                    st.session_state.provision_last_repo = repo_clean
                    st.session_state.provision_monitor_bot_idx = idx_bot

                with st.expander("Resultado técnico", expanded=False):
                    st.json(bundle)

                r_train = bundle.get("remote_train") or {}
                if r_train.get("shell_setup_and_train"):
                    with st.expander("Comando en el Pod (descarga dataset desde Hugging Face)", expanded=False):
                        st.code(r_train["shell_setup_and_train"], language="bash")

                hub_s = bundle.get("hub_sample") or bundle.get("hub_full") or {}
                pod_s = bundle.get("runpod_pod") or {}
                if hub_s.get("simulated") or pod_s.get("simulated"):
                    st.info(
                        "Modo simulación o parcial: define **HF_TOKEN**, **RUNPOD_API_KEY** y plantillas "
                        "`RUNPOD_TEMPLATE_*`; para GraphQL real activa **RUNPOD_DEPLOY_REAL=1**."
                    )
                elif not training_connection.USE_RUNPOD and bundle.get("local_training", {}).get("ok"):
                    st.success("Prueba local completada (modelo pequeño). Revisa **Resultado técnico**.")
                else:
                    st.balloons()
            except Exception as e:
                log_step(f"Aprovisionamiento: {e!r}")
                st.error(format_user_error(e))

    if st.session_state.get("provision_last_repo"):
        st.markdown("---")
        mc1, mc2 = st.columns([4, 1])
        with mc1:
            st.subheader("Monitoreo · logs en Hugging Face Hub")
            st.caption(
                "La barra usa `training_logs/progress.json` y `training_logs/events.jsonl` "
                "subidos por el script de entrenamiento en RunPod."
            )
        with mc2:
            if st.button("Ocultar monitoreo", key="provision_hide_monitor"):
                st.session_state.provision_last_repo = None
                st.rerun()

        watch_repo = st.session_state.provision_last_repo
        midx = int(st.session_state.get("provision_monitor_bot_idx") or 0)
        s_face, w_face = KAWAII_FACE_PAIRS[midx % len(KAWAII_FACE_PAIRS)]
        frag = getattr(st, "fragment", None)
        if frag:
            @frag(run_every=3.0)
            def _provision_train_monitor_frag():
                render_provision_training_monitor(
                    watch_repo,
                    sleep_face=s_face,
                    wake_face=w_face,
                )

            _provision_train_monitor_frag()
        else:
            render_provision_training_monitor(
                watch_repo,
                sleep_face=s_face,
                wake_face=w_face,
            )
            st.caption("Sin `@st.fragment`: pulsa **R** o recarga para actualizar el progreso.")

    with st.expander("Modo clásico — revisión de texto (sin nube)", expanded=False):
        st.caption(
            "Los JSON de descarga masiva del Buscador incluyen **fuente** e **índice**; "
            "nombre típico: `YYYY-MM-DD_bj_<índice>_<registro>.json`."
        )
        opcion = st.radio(
            "Origen del texto",
            ["Usar Almacén", "Subir PDF o TXT"],
            key="radio_entrenar_origen",
            horizontal=True,
        )
        if opcion.startswith("Usar"):
            archivos = listar_almacen_json_entrenamiento()
            if not archivos:
                st.info("No hay archivos JSON en ./almacen.")
            else:
                nombres = [p.name for p in archivos]
                elegido = st.selectbox("Archivo", nombres, key="sel_almacen_train")
                path = ALMACEN_DIR / elegido
                if st.button("Cargar texto para revisión", key="btn_train_load_alm"):
                    log_step(f"Leyendo JSON del almacén: {path}")
                    data = json.loads(path.read_text(encoding="utf-8"))
                    st.session_state.texto_entrenar = data.get("texto") or ""
                    st.success("Texto cargado.")
        else:
            up = st.file_uploader("Archivo", type=["pdf", "txt"], key="up_train")
            if up is not None and st.button("Leer archivo", key="btn_train_load_up"):
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

        st.subheader("Vista previa")
        vista = st.session_state.texto_entrenar
        if vista:
            st.text_area("Texto", value=vista[:12000], height=280, key="ta_train")
        else:
            st.caption("Carga un JSON o archivo para ver el texto aquí.")


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
