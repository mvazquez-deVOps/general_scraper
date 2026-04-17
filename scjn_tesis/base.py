"""Contrato común para adaptadores de fuentes SCJN."""

from __future__ import annotations

from typing import Callable, Protocol

from scjn_tesis.models import SearchParams, TesisRecord

LogFn = Callable[[str], None]


class TesisSourceAdapter(Protocol):
    def search(
        self,
        params: SearchParams,
        *,
        max_pages: int | None,
        fetch_detail: bool,
        headless: bool,
        log: LogFn | None,
    ) -> list[TesisRecord]: ...
