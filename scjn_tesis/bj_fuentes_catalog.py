"""
Catálogo fuente → índices para el Buscador Jurídico (parámetros `fuente` e `indice` en /busqueda).
Valores alineados con las URLs canónicas del portal SCJN.
"""

from __future__ import annotations

# clave UI → (valor API para query `fuente`, lista (indice_api, etiqueta))
BJ_FUENTES_INDICES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "SJF": (
        "SJF",
        [
            ("tesis", "Tesis"),
            ("votos", "Votos"),
            ("acuerdos", "Acuerdos"),
            ("ejecutorias", "Precedentes"),
        ],
    ),
    "SIJ": (
        "SIJ",
        [
            ("sentencias_pub", "Sentencias"),
            ("expedientes_pub", "Expedientes"),
            ("votos_sentencias_pub", "Votos sentencias"),
        ],
    ),
    "SIL": ("SIL", [("legislacion", "Ordenamientos")]),
    "Portal General": (
        "Portal General",
        [
            ("biblioteca", "Acervo bibliotecario"),
            ("vtaquigraficas", "Versiones taquigráficas"),
            ("comunicado", "Comunicados"),
            ("cronicas", "Crónicas"),
        ],
    ),
    "Unidad de Transparencia de la SCJN": (
        "Unidad de Transparencia de la SCJN",
        [
            ("rest_api_transp_resoluciones_comite_transparencia", "Resoluciones SCJN"),
        ],
    ),
    "Órganos internacionales": (
        "Órganos internacionales",
        [
            ("cidh", "Corte Interamericana de DD.HH."),
            ("hudoc", "Corte Europea de DD.HH."),
            ("bjdh_coidh", "Comisión Interamericana de DD.HH."),
        ],
    ),
    "Tribunales y Cortes Constitucionales": (
        "Tribunales y Cortes Constitucionales",
        [
            ("cccolombia", "Corte Constitucional Colombia"),
            ("tcchile", "Tribunal Constitucional Chile"),
            ("csjnargentina", "Corte Suprema Argentina"),
            ("tcespanol", "Tribunal Constitucional España"),
            ("ccitaliana", "Corte Constitucional Italia"),
        ],
    ),
}


def listar_fuentes_ui() -> list[str]:
    return list(BJ_FUENTES_INDICES.keys())


def fuente_api_desde_ui(etiqueta: str) -> str:
    return BJ_FUENTES_INDICES[etiqueta][0]


def indices_para_fuente(etiqueta: str) -> list[tuple[str, str]]:
    return BJ_FUENTES_INDICES[etiqueta][1]


def etiqueta_indice_por_slug(fuente_ui: str, slug: str) -> str:
    for s, lab in indices_para_fuente(fuente_ui):
        if s == slug:
            return lab
    return slug
