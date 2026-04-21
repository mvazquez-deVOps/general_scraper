"""
Conector Hugging Face: todos los JSON bajo ``almacen/`` → ``datasets.Dataset`` → Hub.

La subida usa ``token`` desde Streamlit (**``st.secrets["HF_TOKEN"]``**) por defecto.
Configura ``.streamlit/secrets.toml`` con ``HF_TOKEN = "hf_..."`` (sin commitear ese archivo).

En RunPod exporta ``HF_TOKEN`` para que el script pueda hacer ``load_dataset("usuario/mi-dataset")``.
El modelo base de entrenamiento (p. ej. Gemma) es otro repo; no confundir con el dataset desplegado desde aquí.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scjn_tesis.hf_integration import SKIP_ALMACEN_JSON, load_record_from_json


def default_almacen_dir() -> Path:
    """Raíz del proyecto → ``almacen/``."""
    return Path(__file__).resolve().parent.parent / "almacen"


def collect_json_paths_recursive(almacen_dir: Path) -> list[Path]:
    """Todos los ``*.json`` bajo ``almacen_dir`` (subcarpetas incluidas), menos archivos de sistema."""
    base = Path(almacen_dir)
    out: list[Path] = []
    for p in sorted(base.rglob("*.json"), key=lambda x: str(x).lower()):
        if p.name in SKIP_ALMACEN_JSON:
            continue
        out.append(p)
    return out


def token_from_streamlit_secrets() -> str:
    """Lee ``HF_TOKEN`` desde la app Streamlit (``.streamlit/secrets.toml``)."""
    import streamlit as st

    tok = st.secrets["HF_TOKEN"]
    s = str(tok).strip()
    if not s:
        raise ValueError("HF_TOKEN en secrets está vacío.")
    return s


def build_dataset_from_almacen_tree(
    almacen_dir: Path | None = None,
    *,
    log: Any = print,
):
    """
    Lee todos los JSON del árbol ``almacen/`` y construye un ``datasets.Dataset``.
    """
    try:
        from datasets import Dataset
    except ImportError as e:
        raise ImportError("Instala el paquete `datasets`.") from e

    root = Path(almacen_dir or default_almacen_dir())
    paths = collect_json_paths_recursive(root)
    if not paths:
        raise FileNotFoundError(f"No hay JSON de contenido en {root}")

    rows: list[dict[str, Any]] = []
    for p in paths:
        try:
            rec = load_record_from_json(p)
            rec["relative_path"] = p.relative_to(root).as_posix()
            rows.append(rec)
        except Exception as ex:
            log(f"[hf_connector] Omitiendo {p}: {ex!r}")

    if not rows:
        raise ValueError("Ningún JSON pudo leerse correctamente.")
    return Dataset.from_list(rows)


def push_almacen_to_hub(
    repo_id: str,
    *,
    almacen_dir: Path | None = None,
    private: bool = True,
    token: str | None = None,
    max_shard_size: str = "50MB",
    log: Any = print,
) -> dict[str, Any]:
    """
    Crea el dataset en el Hub (si no existe) y ejecuta ``Dataset.push_to_hub``.

    Parameters
    ----------
    repo_id
        Identificador ``usuario/nombre-repositorio`` del **dataset** en Hugging Face.
    token
        Si es ``None``, se usa ``st.secrets["HF_TOKEN"]`` (solo dentro de Streamlit).
    """
    repo_id = (repo_id or "").strip()
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id debe ser `usuario/nombre` del dataset en Hugging Face.")

    tok = (token or "").strip() or token_from_streamlit_secrets()

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError("Instala `huggingface_hub`.") from e

    root = Path(almacen_dir or default_almacen_dir())
    ds = build_dataset_from_almacen_tree(root, log=log)

    api = HfApi(token=tok)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    ds.push_to_hub(repo_id, private=private, token=tok, max_shard_size=max_shard_size)

    log(f"[hf_connector] push_to_hub OK: {repo_id} ({len(ds)} filas).")
    return {
        "ok": True,
        "repo_id": repo_id,
        "private": private,
        "num_rows": len(ds),
        "source_root": str(root.resolve()),
    }
