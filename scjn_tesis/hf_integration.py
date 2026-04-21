"""
Integración Hugging Face Hub: JSON del almacén → Dataset + push privado como fuente de verdad.

Credenciales para el token (tras ``load_dotenv(..., override=True)``):

1. Token explícito en ``ensure_hf_login(token=...)``.
2. ``os.getenv("HF_TOKEN")`` / ``os.getenv("HUGGING_FACE_HUB_TOKEN")`` (el ``.env`` rellena ``os.environ``).
3. ``st.secrets["HF_TOKEN"]`` en Streamlit si no hay token en entorno.

Repo opcional: ``os.getenv("HF_REPO")``; modelo local: ``os.getenv("HF_MODEL")``.

El entrenamiento remoto (RunPod) debe escribir ``training_logs/progress.json`` en el mismo
repositorio del dataset para que la app muestre progreso real.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SKIP_ALMACEN_JSON = frozenset({"mapa_scjn.json", "config_mapa.json"})

DEFAULT_PROGRESS_PATH = "training_logs/progress.json"
DEFAULT_EVENTS_PATH = "training_logs/events.jsonl"

def project_root() -> Path:
    """Raíz del repo (directorio que contiene ``almacen/``)."""
    return Path(__file__).resolve().parent.parent


def load_dotenv_from_project(*, override: bool = True) -> bool:
    """
    Carga ``.env`` en la raíz del proyecto con ``override=True`` por defecto,
    para que los valores del archivo sustituyan variables vacías del sistema (p. ej. Windows).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    env_path = project_root() / ".env"
    if not env_path.is_file():
        return False
    load_dotenv(env_path, override=override)
    return True


def try_streamlit_hf_token() -> str | None:
    """Lee ``HF_TOKEN`` desde Streamlit secrets si la app está en ejecución."""
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            raw = st.secrets.get("HF_TOKEN")
            if raw is not None and str(raw).strip():
                return str(raw).strip()
    except Exception:
        return None
    return None


def resolve_hf_token() -> str | None:
    """
    Token Hub: primero ``os.getenv`` tras cargar ``.env``; si falta, Streamlit secrets.
    """
    load_dotenv_from_project()
    tok = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if tok:
        return tok
    ts = try_streamlit_hf_token()
    return (ts or "").strip() or None


def get_hf_repo_from_env() -> str | None:
    """``HF_REPO`` para push del adaptador (lee ``.env`` vía ``load_dotenv`` + ``os.getenv``)."""
    load_dotenv_from_project()
    r = os.getenv("HF_REPO")
    return (r or "").strip() or None


def get_hf_model_from_env() -> str | None:
    """``HF_MODEL`` opcional para la prueba local."""
    load_dotenv_from_project()
    m = os.getenv("HF_MODEL")
    m = (m or "").strip().rstrip(".")
    return m or None


def get_hf_token() -> str | None:
    """Compatibilidad: igual que ``resolve_hf_token()``."""
    return resolve_hf_token()


def ensure_hf_login(*, token: str | None = None, log: Any = print) -> str | None:
    """Login en el Hub con ``huggingface_hub.login`` (token explícito o resuelto)."""
    tok = (token or "").strip() if token else None
    tok = tok or resolve_hf_token()
    if not tok:
        return None
    try:
        from huggingface_hub import login

        login(token=tok, add_to_git_credential=False)
    except ImportError:
        log("[hf_integration] huggingface_hub no instalado.")
        return None
    except Exception as e:
        log(f"[hf_integration] login falló: {e!r}")
        return None
    return tok


def collect_json_paths(source_dir: Path) -> list[Path]:
    base = Path(source_dir)
    return sorted(
        [p for p in base.glob("*.json") if p.name not in SKIP_ALMACEN_JSON],
        key=lambda p: p.name.lower(),
    )


def load_record_from_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    texto = raw.get("texto") or ""
    titulo = raw.get("titulo") or ""
    return {
        "text": texto.strip(),
        "titulo": str(titulo)[:2000],
        "url": str(raw.get("url") or ""),
        "fuente": str(raw.get("fuente") or ""),
        "indice": str(raw.get("indice") or ""),
        "numero_registro": str(raw.get("numero_registro") or ""),
        "titulo_record": str(titulo)[:500],
        "source_file": path.name,
        "meta_json": json.dumps(raw, ensure_ascii=False),
    }


def build_dataset_from_almacen(source_dir: Path, *, log: Any = print):
    """
    Convierte todos los JSON válidos de ``source_dir`` en un ``datasets.Dataset``.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Instala el paquete `datasets` para construir el dataset.") from None

    paths = collect_json_paths(Path(source_dir))
    if not paths:
        raise FileNotFoundError(f"No hay JSON de contenido en {source_dir}")

    rows = []
    for p in paths:
        try:
            rows.append(load_record_from_json(p))
        except Exception as e:
            log(f"[hf_integration] Omitiendo {p.name}: {e!r}")

    if not rows:
        raise ValueError("Ningún JSON pudo leerse correctamente.")
    return Dataset.from_list(rows)


def push_almacen_sample_to_hub(
    repo_id: str,
    source_dir: Path,
    *,
    n: int = 10,
    token: str | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """
    Antes del entrenamiento: convierte hasta ``n`` JSON del directorio del almacén (p. ej. avatar)
    en un ``datasets.Dataset`` y lo publica con ``Dataset.push_to_hub(repo_id)`` en Hugging Face Hub.

    ``repo_id`` debe ser ``TU_USUARIO/nombre-del-repo`` del dataset.
    """
    repo_id = (repo_id or "").strip()
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id debe ser `usuario/nombre` del dataset en Hugging Face.")

    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        log("[hf_integration] Instala `datasets` y `huggingface_hub`.")
        return {"ok": False, "simulated": True, "reason": "missing_dependency"}

    tok = ensure_hf_login(token=token, log=log)
    if not tok:
        log("[hf_integration] HF_TOKEN no definido (Streamlit secrets, `.env` o variables de entorno).")
        return {"ok": False, "simulated": True, "reason": "missing_token"}

    paths = collect_json_paths(Path(source_dir))[: max(1, int(n))]
    rows = []
    for p in paths:
        try:
            rows.append(load_record_from_json(p))
        except Exception as e:
            log(f"[hf_integration] Omitiendo {p.name}: {e!r}")
    if not rows:
        raise FileNotFoundError(f"No hay JSON válidos en {source_dir}")

    ds = Dataset.from_list(rows)
    api = HfApi(token=tok)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    ds.push_to_hub(repo_id, private=True, token=tok, max_shard_size="50MB")
    log(f"[hf_integration] push_to_hub muestra ({len(rows)} filas) → {repo_id}")
    return {
        "ok": True,
        "simulated": False,
        "repo_id": repo_id,
        "num_rows": len(rows),
        "n_requested": n,
        "private": True,
    }


def upload_training_bundle(
    repo_id: str,
    *,
    token: str,
    log: Any = print,
) -> list[str]:
    """Sube scripts de entrenamiento generados por ``train_config`` al dataset en el Hub."""
    import tempfile

    from train_config import write_training_bundle_to_directory

    try:
        from huggingface_hub import HfApi
    except ImportError:
        log("[hf_integration] huggingface_hub no disponible; no se suben scripts.")
        return []

    api = HfApi(token=token)
    uploaded: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        paths_rel = write_training_bundle_to_directory(root, dataset_repo_id=repo_id)
        for rel in paths_rel:
            fp = root / rel
            dest_in_repo = Path(rel).as_posix()
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=dest_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Añadir {dest_in_repo}",
            )
            uploaded.append(dest_in_repo)

    log(f"[hf_integration] Scripts de entrenamiento subidos: {len(uploaded)} archivos.")
    return uploaded


def push_dataset_to_hub_private(
    repo_id: str,
    source_dir: Path,
    *,
    upload_scripts: bool = True,
    log: Any = print,
) -> dict[str, Any]:
    """
    Login con HF_TOKEN, construye Dataset desde JSON del almacén y **push_to_hub** como repositorio **privado**.
    """
    repo_id = (repo_id or "").strip()
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id debe ser `usuario/nombre` del dataset en Hugging Face.")

    tok = ensure_hf_login(log=log)
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log("[hf_integration] huggingface_hub no instalado; modo simulado.")
        paths = collect_json_paths(Path(source_dir))
        return {
            "ok": True,
            "simulated": True,
            "repo_id": repo_id,
            "n_files": len(paths),
            "reason": "missing_dependency",
        }

    if not tok:
        log("[hf_integration] HF_TOKEN no definido; modo simulado.")
        paths = collect_json_paths(Path(source_dir))
        return {
            "ok": True,
            "simulated": True,
            "repo_id": repo_id,
            "n_files": len(paths),
            "reason": "missing_token",
        }

    ds = build_dataset_from_almacen(Path(source_dir), log=log)
    api = HfApi(token=tok)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    ds.push_to_hub(repo_id, private=True, token=tok, max_shard_size="50MB")

    uploaded_scripts: list[str] = []
    if upload_scripts:
        try:
            uploaded_scripts = upload_training_bundle(repo_id, token=tok, log=log)
        except Exception as e:
            log(f"[hf_integration] No se pudieron subir scripts de entrenamiento: {e!r}")

    return {
        "ok": True,
        "simulated": False,
        "repo_id": repo_id,
        "private": True,
        "num_rows": len(ds),
        "training_scripts": uploaded_scripts,
    }


def push_dataset(
    repo_id: str,
    source_dir: Path,
    *,
    repo_type: str = "dataset",
    log: Any = print,
) -> dict[str, Any]:
    """
    Compatibilidad con la API anterior: equivale a ``push_dataset_to_hub_private`` (ignora ``repo_type`` duplicado).
    """
    return push_dataset_to_hub_private(repo_id, Path(source_dir), upload_scripts=True, log=log)


def fetch_training_progress(
    repo_id: str,
    *,
    path_in_repo: str = DEFAULT_PROGRESS_PATH,
    token: str | None = None,
) -> dict[str, Any] | None:
    """
    Descarga ``training_logs/progress.json`` del dataset (escrito por el entrenador en RunPod).

    Devuelve un dict con al menos ``step``, ``max_steps``, ``epoch``, ``loss``, ``status`` si existe el archivo.
    """
    repo_id = (repo_id or "").strip()
    if not repo_id:
        return None
    tok = token or get_hf_token()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        p = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=tok,
        )
        raw = Path(p).read_text(encoding="utf-8")
        return json.loads(raw)
    except Exception:
        return None


def fetch_training_events_tail(
    repo_id: str,
    *,
    path_in_repo: str = DEFAULT_EVENTS_PATH,
    token: str | None = None,
    max_lines: int = 20,
) -> list[dict[str, Any]]:
    """Últimas líneas JSON de ``training_logs/events.jsonl`` (log estilo Hugging Face Trainer)."""
    repo_id = (repo_id or "").strip()
    if not repo_id:
        return []
    tok = token or get_hf_token()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return []
    try:
        p = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=tok,
        )
        lines = Path(p).read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for ln in lines[-max_lines:]:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out
    except Exception:
        return []


def progress_fraction(progress: dict[str, Any] | None) -> float:
    """0..1 para la barra de Streamlit."""
    if not progress:
        return 0.0
    mx = progress.get("max_steps")
    st = progress.get("step") or progress.get("global_step")
    if mx is None or st is None:
        return 0.0
    try:
        mx = float(mx)
        st = float(st)
        if mx <= 0:
            return 0.0
        return max(0.0, min(1.0, st / mx))
    except (TypeError, ValueError):
        return 0.0
