"""Configuración compartida de Playwright (Windows + anti-detección básica)."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Playwright

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

_CHROMIUM_ARGS = (
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
)
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_STEALTH_INIT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['es-MX', 'es', 'en-US', 'en'] });
"""


def launch_browser(p: Playwright, *, headless: bool = True, channel: str | None = "chrome") -> Browser:
    kw: dict = {
        "headless": headless,
        "args": list(_CHROMIUM_ARGS),
        "ignore_default_args": ["--enable-automation"],
    }
    if channel:
        kw["channel"] = channel
    try:
        return p.chromium.launch(**kw)
    except Exception:
        kw.pop("channel", None)
        return p.chromium.launch(**kw)


def new_context(browser: Browser):
    ctx = browser.new_context(
        user_agent=_USER_AGENT,
        viewport={"width": 1366, "height": 768},
        locale="es-MX",
        timezone_id="America/Mexico_City",
        extra_http_headers={
            "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
        },
    )
    ctx.add_init_script(_STEALTH_INIT)
    return ctx


def settle_page(page, *, headless: bool = True) -> None:
    import time

    try:
        page.wait_for_load_state("networkidle", timeout=90_000)
    except Exception:
        pass
    try:
        page.evaluate("window.scrollTo(0, Math.min(document.body.scrollHeight, 4000))")
        time.sleep(0.4)
        page.evaluate("window.scrollTo(0, 0)")
    except Exception:
        pass
    time.sleep(3 if headless else 5)
