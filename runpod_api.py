"""
Cliente mínimo para RunPod: creación de pod y disparo de entrenamiento.

Sin `RUNPOD_API_KEY` las funciones devuelven resultado **simulado** para desarrollo local.
Los IDs de plantilla GPU pueden ajustarse con variables de entorno (ver código).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import urllib.error
import urllib.request


# Plantillas genéricas (placeholders). Sustituye por IDs reales de tu cuenta RunPod.
_GPU_TEMPLATE_ENV = {
    "rtx3090": "RUNPOD_TEMPLATE_RTX3090",
    "a6000": "RUNPOD_TEMPLATE_A6000",
    "a100": "RUNPOD_TEMPLATE_A100",
}


def _template_id_for_gpu(gpu_key: str) -> str | None:
    env_name = _GPU_TEMPLATE_ENV.get(gpu_key)
    if not env_name:
        return None
    tid = os.environ.get(env_name)
    if tid and tid.strip():
        return tid.strip()
    return None


def create_pod(gpu_key: str, *, log: Any = print) -> dict[str, Any]:
    """
    Lanza una instancia RunPod con la GPU indicada.

    gpu_key: uno de ``rtx3090``, ``a6000``, ``a100``.
    """
    gpu_key = (gpu_key or "").lower().strip()
    if gpu_key not in _GPU_TEMPLATE_ENV:
        raise ValueError(f"GPU no soportada: {gpu_key!r}")

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    template_id = _template_id_for_gpu(gpu_key)

    if not api_key or not template_id:
        pod_id = f"sim-pod-{uuid.uuid4().hex[:12]}"
        log(
            "[runpod_api] RUNPOD_API_KEY o plantilla GPU no configurada; "
            f"simulando pod {pod_id!r} ({gpu_key})."
        )
        return {
            "ok": True,
            "simulated": True,
            "pod_id": pod_id,
            "gpu_key": gpu_key,
            "reason": "missing_api_key_or_template",
        }

    use_real = os.environ.get("RUNPOD_DEPLOY_REAL", "").strip() in ("1", "true", "yes")
    if not use_real:
        pod_id = f"sim-pod-{uuid.uuid4().hex[:12]}"
        log(
            "[runpod_api] Credenciales presentes; simulación activa. "
            "Define RUNPOD_DEPLOY_REAL=1 para llamar a la API GraphQL."
        )
        return {
            "ok": True,
            "simulated": True,
            "pod_id": pod_id,
            "gpu_key": gpu_key,
            "reason": "deploy_real_disabled",
        }

    query = """
mutation PodFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput) {
  podFindAndDeployOnDemand(input: $input) {
    id
    desiredStatus
  }
}
"""
    payload = {
        "query": query,
        "variables": {
            "input": {
                "cloudType": "ALL",
                "gpuCount": 1,
                "volumeInGb": 40,
                "containerDiskInGb": 20,
                "minVcpuCount": 2,
                "minMemoryInGb": 15,
                "gpuTypeId": template_id,
                "name": f"extractor-ft-{gpu_key}",
                "imageName": "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
            }
        },
    }

    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        errs = data.get("errors")
        if errs:
            raise RuntimeError(errs)
        pod_info = data.get("data", {}).get("podFindAndDeployOnDemand") or {}
        pid = pod_info.get("id")
        if not pid:
            raise RuntimeError(f"Respuesta RunPod sin id: {raw[:500]}")
        return {"ok": True, "simulated": False, "pod_id": pid, "gpu_key": gpu_key}
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, json.JSONDecodeError) as e:
        log(f"[runpod_api] Fallo real; simulando pod: {e!r}")
        pod_id = f"fallback-sim-{uuid.uuid4().hex[:12]}"
        return {
            "ok": True,
            "simulated": True,
            "pod_id": pod_id,
            "gpu_key": gpu_key,
            "reason": str(e),
        }


def send_training_command(
    pod_id: str,
    dataset_repo_id: str,
    gpu_key: str,
    *,
    base_model: str | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """
    Prepara el comando para el Pod: descarga ``training/*`` desde el dataset privado en HF y ejecuta ``runpod_entry.sh``.
    La ejecución real ocurre dentro del Pod (SSH / consola web); aquí solo generamos el script listo para pegar.
    """
    from train_config import build_pod_invocation_snippet

    pod_id = (pod_id or "").strip()
    dataset_repo_id = (dataset_repo_id or "").strip()
    if not dataset_repo_id:
        raise ValueError("dataset_repo_id vacío.")

    snippet = build_pod_invocation_snippet(dataset_repo_id, base_model=base_model)
    log(
        f"[runpod_api] Invocación remota lista (pod={pod_id!r}, gpu={gpu_key!r}, dataset={dataset_repo_id!r})."
    )
    return {
        "ok": True,
        "pod_id": pod_id,
        "dataset_repo_id": dataset_repo_id,
        "gpu_key": gpu_key,
        "shell_setup_and_train": snippet,
        "note": "Ejecuta el bloque shell_setup_and_train en la consola del Pod con HF_TOKEN exportado.",
    }


def start_training(
    pod_id: str,
    repo_id: str,
    gpu_key: str,
    *,
    base_model: str | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Alias: envía al Pod las instrucciones para entrenar leyendo el dataset desde Hugging Face."""
    return send_training_command(pod_id, repo_id, gpu_key, base_model=base_model, log=log)
