"""
Catálogo fuente → índices para el Buscador Jurídico (parámetros `fuente` e `indice` en /busqueda).
Valores alineados con las URLs canónicas del portal SCJN.
"""

from __future__ import annotations

# clave UI → (valor API para query `fuente`, lista (indice_api, etiqueta))
# Fuente de verdad: MAPEO.md (URLs canónicas bj.scjn.gob.mx/busqueda).
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
            ("ccj_cursos", "Casa de la cultura jurídica"),
            ("comunicado", "Comunicados"),
            ("discursos_mp", "Discursos presidencia"),
            ("resoluciones_pleno", "Resoluciones del Pleno"),
            ("cronicas", "Crónicas"),
            ("listas_sesion_pub", "Listas oficiales"),
            # «Informe de labores» en MAPEO usa solo /busqueda?q=… (sin fuente/indice); no hay slug BJ único aquí.
        ],
    ),
    "Unidad de Transparencia de la SCJN": (
        "Unidad de Transparencia de la SCJN",
        [
            ("rest_api_transp_resoluciones_comite_transparencia", "Resoluciones SCJN"),
            # «Resoluciones comité especializado» en MAPEO es búsqueda global q=… sin fuente/indice documentada.
        ],
    ),
    "Órganos internacionales": (
        "Órganos internacionales",
        [
            ("cidh", "Corte Interamericana de Derechos Humanos"),
            ("hudoc", "Corte Europea de Derechos Humanos"),
            ("cij", "Corte Internacional de Justicia"),
            ("suniversal", "Sistema de Naciones Unidas"),
            ("bjdh_coidh", "Comisión Interamericana de Derechos Humanos"),
            ("cadh", "Corte Africana de Derechos Humanos y de los Pueblos"),
            # Corte Penal Internacional: sin URL de listado en MAPEO (índice pendiente).
        ],
    ),
    "Tribunales y Cortes Constitucionales": (
        "Tribunales y Cortes Constitucionales",
        [
            ("cccolombia", "Corte Constitucional de la República de Colombia"),
            ("tcchile", "Tribunal Constitucional de Chile"),
            ("csjnargentina", "Corte Suprema de Justicia de Argentina"),
            ("tcespanol", "Tribunal Constitucional de España"),
            ("supremecourtusa", "Suprema Corte de Estados Unidos de América"),
            ("corteuk", "Suprema Corte de Reino Unido"),
            ("tcaleman", "Tribunal Constitucional de Alemania"),
            ("ccitaliana", "Corte Constitucional de Italia"),
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
