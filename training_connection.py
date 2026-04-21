"""
Orquestación del entrenamiento: entorno local (CPU/GPU) vs RunPod.

- ``USE_RUNPOD = False``: sube una muestra de JSON al Hub, entrena en máquina local con un modelo pequeño.
- ``USE_RUNPOD = True``: push completo del dataset + Pod RunPod + instrucciones remotas (activar credenciales reales).

Las funciones RunPod están implementadas y listas; la rama local no las ejecuta.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from scjn_tesis import hf_integration

# Evita avisos amarillos en Windows sobre symlinks al cachear modelos/datasets del Hub.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# -----------------------------------------------------------------------------
# Interruptor de entorno (cambiar a True cuando RunPod + API estén listos).
# -----------------------------------------------------------------------------
USE_RUNPOD = False

# Mensajes del bot kawaii (Streamlit ``st.status`` / etiquetas).
BOT_MESSAGE_HF_CONNECT = "¡Conectando con mi base en Hugging Face! 🚀"
BOT_MESSAGE_LOCAL_START = "Iniciando prueba local con 10 archivos..."
BOT_MESSAGE_CLOUD_START = "¡Despegando a la nube de RunPod!"
# Al pulsar Iniciar, la UI usa primero BOT_MESSAGE_HF_CONNECT; las fases siguen con _provision_phase.

# Modelo pequeño para validar el flujo en CPU (SmolLM2 135M — documentación HF).
LOCAL_TRIAL_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"

# Producción en RunPod (mayor VRAM — no usar en la prueba local actual):
# RUNPOD_BASE_MODEL_GEMMA = "google/gemma-2-9b-it"


def push_first_n_json_to_hub(
    repo_id: str,
    source_dir: Path,
    *,
    n: int = 10,
    token: str | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Delega en ``hf_integration.push_almacen_sample_to_hub`` (misma API)."""
    return hf_integration.push_almacen_sample_to_hub(
        repo_id, Path(source_dir), n=n, token=token, log=log
    )


def create_runpod_pod(gpu_key: str, *, log: Any = print) -> dict[str, Any]:
    """
    Crea un Pod en RunPod (GPU).

    Antes de producción:
    - Define ``RUNPOD_API_KEY`` y plantillas ``RUNPOD_TEMPLATE_RTX3090`` / ``RUNPOD_TEMPLATE_A6000`` /
      ``RUNPOD_TEMPLATE_A100`` según la SKU (A6000 vs A100).
    - ``RUNPOD_DEPLOY_REAL=1`` para llamada GraphQL real (ver ``runpod_api.create_pod``).

    # DESCOMENTAR ESTO PARA FASE DE PRODUCCIÓN EN NUBE:
    # export RUNPOD_API_KEY="..."
    # export RUNPOD_TEMPLATE_A6000="<template_id>"   # o A100: RUNPOD_TEMPLATE_A100
    # export RUNPOD_DEPLOY_REAL=1
    """
    import runpod_api

    return runpod_api.create_pod(gpu_key, log=log)


def start_remote_train(
    pod_id: str,
    dataset_repo_id: str,
    gpu_key: str,
    *,
    base_model: str | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """
    Genera el comando para el Pod: el worker debe tener ``HF_TOKEN`` y ejecutará en esencia:

        load_dataset("usuario/repo-dataset", split="train", token=hf_token)

    junto con el script de entrenamiento generado (ver ``runpod_api.send_training_command``).

    # DESCOMENTAR ESTO PARA FASE DE PRODUCCIÓN EN NUBE — ejemplo en consola del Pod:
    # export HF_TOKEN="hf_..."
    # pip install -q datasets transformers
    # python - <<'PY'
    # from datasets import load_dataset
    # import os
    # ds = load_dataset("tu_usuario/tu_dataset", split="train", token=os.environ["HF_TOKEN"])
    # print(ds)
    # PY
    """
    import runpod_api

    return runpod_api.send_training_command(
        pod_id,
        dataset_repo_id,
        gpu_key,
        base_model=base_model,
        log=log,
    )


def run_local_minimal_training(
    source_dir: Path,
    *,
    max_samples: int = 10,
    max_steps: int = 5,
    token: str | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """
    Fine-tuning mínimo (PEFT) sobre JSON locales. ``HF_MODEL`` en ``.env`` sobreescribe el base model;
    al terminar, si ``HF_REPO`` está en ``.env``, sube adaptador + tokenizer (``push_to_hub``).
    Solo CPU: ``use_cpu=True``, ``fp16=False``, dataloader en hilo principal.
    """
    import os

    # Cargar .env antes de leer HF_MODEL / HF_REPO / HF_TOKEN.
    hf_integration.load_dotenv_from_project()
    tok_hf = hf_integration.ensure_hf_login(token=token, log=log)

    # Antes de importar torch: oculta GPUs para que el entrenamiento de prueba sea solo CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    paths = hf_integration.collect_json_paths(Path(source_dir))[:max_samples]
    if not paths:
        raise FileNotFoundError(f"No hay JSON en {source_dir}")

    rows = []
    for p in paths:
        rows.append(hf_integration.load_record_from_json(p))
    ds = Dataset.from_list(rows)

    base = hf_integration.get_hf_model_from_env() or LOCAL_TRIAL_BASE_MODEL
    # Mismo ID para tokenizer y modelo (por defecto SmolLM2-135M o ``HF_MODEL`` en .env).
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tok_fn(ex):
        return tok(
            ex["text"],
            truncation=True,
            max_length=min(256, getattr(tok, "model_max_length", 2048) or 2048),
            padding=False,
        )

    cols = list(ds.column_names)
    tokenized = ds.map(tok_fn, batched=False, remove_columns=cols)

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    lm = base.lower()
    if "gpt2" in lm:
        targets = ["c_attn", "c_proj"]
    elif "llama" in lm or "mistral" in lm or "gemma" in lm or "smollm" in lm:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        targets = ["c_attn", "c_proj"]

    peft_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, peft_cfg)
    model = model.to(torch.device("cpu"))

    log(f"[training_connection] Entrenamiento local solo en **CPU** ({base}), max_steps={max_steps}")

    out_dir = Path.cwd() / ".training_local_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=99999,
        report_to=[],
        fp16=False,
        bf16=False,
        use_cpu=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()

    # Vuelve a cargar .env y resolver token/repo (por si el proceso arrancó sin archivo o sin override).
    hf_integration.load_dotenv_from_project()
    tok_hf = hf_integration.ensure_hf_login(token=token, log=log)
    repo_push = hf_integration.get_hf_repo_from_env() or ""

    hub_adapter: dict[str, Any] = {"ok": False, "skipped": True}
    if repo_push and "/" in repo_push:
        if not tok_hf:
            hub_adapter = {"ok": False, "skipped": True, "reason": "missing_token_for_push"}
            log("[training_connection] HF_REPO definido pero no hay HF_TOKEN para push_to_hub.")
        else:
            try:
                from huggingface_hub import HfApi

                api = HfApi(token=tok_hf)
                api.create_repo(repo_push, repo_type="model", private=True, exist_ok=True)
                trainer.model.push_to_hub(repo_push, token=tok_hf, private=True)
                tok.push_to_hub(repo_push, token=tok_hf, private=True)
                hub_adapter = {"ok": True, "repo_id": repo_push, "repo_type": "model"}
                log(f"[training_connection] Adaptador + tokenizer → push_to_hub OK: {repo_push}")
            except Exception as e:
                log(f"[training_connection] push_to_hub adaptador falló: {e!r}")
                hub_adapter = {"ok": False, "repo_id": repo_push, "error": str(e)}
    else:
        log("[training_connection] Sin HF_REPO en .env; no se sube el adaptador tras entrenar.")

    return {
        "ok": True,
        "device": "cpu",
        "base_model": base,
        "max_steps": max_steps,
        "num_samples": len(paths),
        "output_dir": str(out_dir.resolve()),
        "hub_adapter_push": hub_adapter,
    }


def run_training_pipeline(
    *,
    repo_id: str,
    source_dir: Path,
    gpu_code: str,
    ui_base_model_id: str | None,
    token: str | None = None,
    log: Any = print,
    on_phase: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Punto único invocado desde la UI (botón Iniciar).

    - ``USE_RUNPOD == False``: ``push_to_hub`` (10 JSON) + entrenamiento local.
    - ``USE_RUNPOD == True``: push **completo** del avatar + Pod + snippet remoto (activar API/tokens).
    """
    hf_integration.load_dotenv_from_project()

    out: dict[str, Any] = {"use_runpod": USE_RUNPOD, "repo_id": repo_id}

    def _phase(name: str) -> None:
        if on_phase:
            on_phase(name)

    tok = hf_integration.ensure_hf_login(token=token, log=log)

    if not USE_RUNPOD:
        _phase("hub_push_sample")
        out["hub_sample"] = hf_integration.push_almacen_sample_to_hub(
            repo_id, Path(source_dir), n=10, token=tok, log=log
        )
        _phase("local_train")
        out["local_training"] = run_local_minimal_training(
            Path(source_dir), max_samples=10, max_steps=5, token=tok, log=log
        )
        return out

    # -------------------------------------------------------------------------
    # DESCOMENTAR ESTO PARA FASE DE PRODUCCIÓN EN NUBE (rama RunPod activa con USE_RUNPOD=True):
    # - Dataset completo en el Hub para que el Pod no se quede sin datos.
    # - ``gpu_code``: "a6000" | "a100" | "rtx3090" — alinear con RUNPOD_TEMPLATE_*.
    # - ``base_model`` en el snippet remoto: usar Gemma / modelo UI (VRAM A100/A6000).
    # -------------------------------------------------------------------------
    _phase("hub_push_full")
    out["hub_full"] = hf_integration.push_dataset_to_hub_private(
        repo_id, Path(source_dir), log=log
    )

    _phase("runpod_pod")
    # GPU sugerida: A6000 (``gpu_code="a6000"``) o A100 (``gpu_code="a100"``) según disponibilidad/cuenta.
    out["runpod_pod"] = create_runpod_pod(gpu_code, log=log)

    remote_model = ui_base_model_id or "google/gemma-2-9b-it"
    _phase("runpod_train")
    out["remote_train"] = start_remote_train(
        (out["runpod_pod"].get("pod_id") or ""),
        repo_id,
        gpu_code,
        base_model=remote_model,
        log=log,
    )

    return out


# ------------------------------------------------------------------------------
# Referencia rápida — copiar al activar RunPod en la cuenta (no se ejecuta solo):
#
# # DESCOMENTAR ESTO PARA FASE DE PRODUCCIÓN EN NUBE
# # export RUNPOD_API_KEY="..."
# # export RUNPOD_DEPLOY_REAL=1
# # export RUNPOD_TEMPLATE_A6000="<id>"   # GPU A6000 en plantilla RunPod
# # export RUNPOD_TEMPLATE_A100="<id>"    # alternativa A100
# #
# # En el Pod (después de ``create_runpod_pod`` / SSH o consola web):
# #   export HF_TOKEN="hf_..."
# #   pip install -q "datasets>=2.16" "transformers>=4.36"
# #   python -c "from datasets import load_dataset; import os; print(load_dataset('usuario/dataset-repo', split='train', token=os.environ['HF_TOKEN']))"
# ------------------------------------------------------------------------------
