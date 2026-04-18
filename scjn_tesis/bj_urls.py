"""
Patrones de URL del Buscador Jurídico (bj.scjn.gob.mx).

Referencia (búsqueda ejemplo «pagare» / «pagaré»):
- Palabra base (todas las fuentes): /busqueda?q=...&page=1&semantica=0
- Tesis SJF: /busqueda?fuente=SJF&indice=tesis&page=1&q=...&semantica=0
- Detalle tesis: /documento/tesis/{registro}

Otros índices (votos, ejecutorias, fuentes SIJ/SIL/Portal, etc.) usan el mismo esquema
cambiando ``fuente`` e ``indice`` (ver documentación oficial / tu mapa de fuentes).
"""

from __future__ import annotations

import urllib.parse


BJ_ORIGIN = "https://bj.scjn.gob.mx"


def bj_busqueda_url(
    query: str,
    *,
    page: int = 1,
    semantica: int = 0,
    fuente: str | None = "SJF",
    indice: str | None = "tesis",
) -> str:
    """
    Construye URL de listado de resultados del Buscador Jurídico.

    - Si ``fuente`` e ``indice`` son None: búsqueda general
      ``/busqueda?q=...&page=&semantica=``.
    - Si ambos están definidos: filtra por fuente e índice (p. ej. Tesis SJF).
    """
    q_enc = urllib.parse.quote(query.strip())
    base = f"{BJ_ORIGIN}/busqueda"
    if fuente is None or indice is None:
        return f"{base}?q={q_enc}&page={page}&semantica={semantica}"
    fu = urllib.parse.quote(fuente)
    return (
        f"{base}?fuente={fu}&indice={indice}&page={page}&q={q_enc}&semantica={semantica}"
    )


# Plantilla para mapa_scjn.json (placeholder {query} = término ya codificado por resolve_extraction_url)
BJ_TEMPLATE_TESIS_SJF = (
    "https://bj.scjn.gob.mx/busqueda?fuente=SJF&indice=tesis&page=1&q={query}&semantica=0"
)

BJ_TEMPLATE_BUSQUEDA_SIMPLE = (
    "https://bj.scjn.gob.mx/busqueda?q={query}&page=1&semantica=0"
)


def parse_bj_busqueda_url(url: str) -> dict[str, str | int | None]:
    """
    Extrae parámetros de una URL de listado /busqueda (fuente, indice, page, q, semantica).
    """
    p = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qs(p.query)
    fuente = (q.get("fuente") or [None])[0]
    indice = (q.get("indice") or [None])[0]
    page_s = (q.get("page") or ["1"])[0]
    try:
        page = int(page_s)
    except (TypeError, ValueError):
        page = 1
    qry = (q.get("q") or [""])[0]
    sem = (q.get("semantica") or ["0"])[0]
    try:
        semantica = int(sem)
    except (TypeError, ValueError):
        semantica = 0
    return {
        "fuente": fuente or "",
        "indice": indice or "",
        "page": page,
        "q": urllib.parse.unquote(qry) if qry else "",
        "semantica": semantica,
    }


def bj_documento_path(indice: str, registro: str) -> str:
    """
    Ruta relativa /documento/{carpeta}/{id} según índice (heurística para BJ).
    """
    ind = (indice or "tesis").lower()
    # Alias (índice listado → segmento en /documento/…); el BJ suele repetir el slug del índice.
    mapping = {
        "tesis": "tesis",
        "ejecutorias": "ejecutorias",
        "votos": "ejecutorias",
        "acuerdos": "acuerdos",
        "sentencias_pub": "sentencias_pub",
        "expedientes_pub": "expedientes_pub",
        "votos_sentencias_pub": "votos_sentencias_pub",
        "legislacion": "legislacion",
        "biblioteca": "biblioteca",
        "vtaquigraficas": "vtaquigraficas",
        "ccj_cursos": "ccj_cursos",
        "comunicado": "comunicado",
        "discursos_mp": "discursos_mp",
        "resoluciones_pleno": "resoluciones_pleno",
        "cronicas": "cronicas",
        "listas_sesion_pub": "listas_sesion_pub",
        "hudoc": "hudoc",
        "cidh": "cidh",
        "cij": "cij",
        "suniversal": "suniversal",
        "bjdh_coidh": "bjdh_coidh",
        "cadh": "cadh",
        "tcchile": "tcchile",
        "csjnargentina": "csjnargentina",
        "tcespanol": "tcespanol",
        "cccolombia": "cccolombia",
        "supremecourtusa": "supremecourtusa",
        "corteuk": "corteuk",
        "tcaleman": "tcaleman",
        "ccitaliana": "ccitaliana",
    }
    carpeta = mapping.get(ind, ind or "tesis")
    return f"/documento/{carpeta}/{registro}"
