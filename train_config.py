"""
Genera archivos de configuración y scripts para RunPod:

- ``runpod_config.json``: metadatos y lista de paquetes pip.
- ``training/remote_train.py``: fine-tuning mínimo con PEFT que sube progreso al Hub.
- ``training/runpod_entry.sh``: preparación del entorno e inicio del entrenamiento.

El modelo base por defecto es ``distilgpt2`` (liviano); en RunPod define ``BASE_MODEL`` para otro modelo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PIP_PACKAGES = [
    "torch",
    "transformers>=4.36.0",
    "accelerate>=0.26.0",
    "peft>=0.8.0",
    "bitsandbytes>=0.41.0",
    "datasets>=2.16.0",
    "huggingface_hub>=0.20.0",
    "sentencepiece",
    "safetensors",
]

REMOTE_TRAIN_PY = '''#!/usr/bin/env python3
"""Generado por train_config.py — fine-tuning con feedback al Hub (training_logs/)."""
from __future__ import annotations

import json
import os
from pathlib import Path

_REPO = os.environ.get("HF_DATASET_REPO", "").strip()
_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
_BASE_MODEL = os.environ.get("BASE_MODEL", "distilgpt2")


def _upload_hub(path_in_repo: str, body: bytes, msg: str) -> None:
    import io

    from huggingface_hub import HfApi

    api = HfApi(token=_TOKEN)
    api.upload_file(
        path_or_fileobj=io.BytesIO(body),
        path_in_repo=path_in_repo,
        repo_id=_REPO,
        repo_type="dataset",
        commit_message=msg,
    )


def _append_event(payload: dict) -> None:
    line = json.dumps(payload, ensure_ascii=False) + "\\n"
    try:
        from huggingface_hub import hf_hub_download

        try:
            prev = hf_hub_download(
                repo_id=_REPO,
                filename="training_logs/events.jsonl",
                repo_type="dataset",
                token=_TOKEN,
            )
            old = Path(prev).read_text(encoding="utf-8")
        except Exception:
            old = ""
        combined = (old + line).encode("utf-8")
        _upload_hub(
            "training_logs/events.jsonl",
            combined,
            "training log",
        )
    except Exception:
        pass


def main() -> None:
    if not _REPO or not _TOKEN:
        raise SystemExit("Define HF_DATASET_REPO y HF_TOKEN")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    ds = load_dataset(_REPO, split="train", token=_TOKEN)
    tok = AutoTokenizer.from_pretrained(_BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tok_fn(ex):
        return tok(
            ex["text"],
            truncation=True,
            max_length=min(512, tok.model_max_length),
            padding=False,
        )

    cols = list(ds.column_names)
    tokenized = ds.map(tok_fn, batched=False, remove_columns=cols)

    model = AutoModelForCausalLM.from_pretrained(_BASE_MODEL)
    _lm = _BASE_MODEL.lower()
    if "gpt2" in _lm:
        _targets = ["c_attn", "c_proj"]
    elif "llama" in _lm or "mistral" in _lm or "phi" in _lm:
        _targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        _targets = ["c_attn", "c_proj"]

    peft_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_targets,
    )
    model = get_peft_model(model, peft_cfg)

    max_steps = int(os.environ.get("MAX_STEPS", "200"))
    out_dir = Path("/tmp/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    class HubProgress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero:
                return
            ms = getattr(args, "max_steps", None) or max_steps
            loss = None
            if logs:
                loss = logs.get("loss")
            payload = {
                "step": int(state.global_step),
                "max_steps": int(ms),
                "epoch": float(state.epoch or 0.0),
                "loss": loss,
                "status": "training",
                "model": _BASE_MODEL,
            }
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            _upload_hub("training_logs/progress.json", body, f"step {state.global_step}")
            evt = dict(payload)
            evt["logs"] = logs or {}
            _append_event(evt)

        def on_train_end(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            body = json.dumps(
                {
                    "step": int(state.global_step),
                    "max_steps": int(max_steps),
                    "status": "completed",
                    "epoch": float(state.epoch or 0.0),
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            _upload_hub("training_logs/progress.json", body, "training completed")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=99999,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tok,
        callbacks=[HubProgress()],
    )
    trainer.train()


if __name__ == "__main__":
    main()
'''


RUNPOD_ENTRY_SH_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN to a write token}"
export HF_DATASET_REPO="{dataset_repo_id}"
export BASE_MODEL="${BASE_MODEL:-distilgpt2}"
export MAX_STEPS="${{MAX_STEPS:-200}}"
echo "[runpod] Instalando dependencias de entrenamiento..."
pip install -q -U {pip_line}
mkdir -p /workspace/scjn_train && cd /workspace/scjn_train
echo "[runpod] Descargando scripts desde Hugging Face Hub..."
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="{dataset_repo_id}",
    repo_type="dataset",
    local_dir="./repo",
    allow_patterns=["training/**"],
    token=__import__("os").environ.get("HF_TOKEN"),
)
PY
echo "[runpod] Iniciando entrenamiento..."
python ./repo/training/remote_train.py
echo "[runpod] Listo."
"""


def generate_runpod_config(
    dataset_repo_id: str,
    *,
    gpu_sku: str = "A6000",
    base_model_env: str = "BASE_MODEL",
    max_steps_default: int = 200,
) -> dict[str, Any]:
    """Estructura de configuración consumible por tooling o documentación."""
    return {
        "version": 1,
        "dataset_repo_id": dataset_repo_id,
        "environment": {
            "python": ">=3.10",
            "pip_upgrade": True,
            "pip_packages": PIP_PACKAGES,
        },
        "training": {
            "base_model_env": base_model_env,
            "max_steps_env": "MAX_STEPS",
            "max_steps_default": max_steps_default,
            "progress_path_in_repo": "training_logs/progress.json",
            "events_path_in_repo": "training_logs/events.jsonl",
        },
        "runpod": {"suggested_gpu": gpu_sku},
    }


def write_training_bundle_to_directory(dest: Path, dataset_repo_id: str) -> list[str]:
    """
    Escribe ``training/remote_train.py``, ``training/runpod_entry.sh`` y ``runpod_config.json``.
    Devuelve rutas relativas respecto a ``dest``.
    """
    dest = Path(dest)
    training_dir = dest / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    (training_dir / "remote_train.py").write_text(REMOTE_TRAIN_PY, encoding="utf-8")

    pip_line = " ".join(f"'{p}'" for p in PIP_PACKAGES)
    entry = RUNPOD_ENTRY_SH_TEMPLATE.format(
        dataset_repo_id=dataset_repo_id,
        pip_line=pip_line,
    )
    (training_dir / "runpod_entry.sh").write_text(entry, encoding="utf-8")

    cfg = generate_runpod_config(dataset_repo_id)
    (dest / "runpod_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return [
        "training/remote_train.py",
        "training/runpod_entry.sh",
        "runpod_config.json",
    ]


def write_runpod_bundle_at_repo_root(repo_root: Path | None = None, dataset_repo_id: str = "") -> Path:
    """Opcional: escribe el bundle en el directorio actual para inspección local."""
    root = Path(repo_root or Path.cwd())
    paths = write_training_bundle_to_directory(root, dataset_repo_id=dataset_repo_id or "usuario/dataset")
    return root / paths[0]


def build_pod_invocation_snippet(
    dataset_repo_id: str,
    *,
    base_model: str | None = None,
) -> str:
    """
    Texto para ejecutar en el Pod (RunPod) una vez que el dataset privado ya está en el Hub.
    Requiere ``HF_TOKEN`` con permiso de lectura sobre el dataset.
    """
    rid = dataset_repo_id.strip()
    bm_line = ""
    if base_model and str(base_model).strip():
        bm_line = f'export BASE_MODEL="{str(base_model).strip()}"\n'
    return f"""# Dataset (fuente de verdad): {rid}
export HF_DATASET_REPO="{rid}"
{bm_line}export HF_TOKEN="${{HF_TOKEN:?}}"
pip install -q -U huggingface_hub>=0.20
python - <<'PY'
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id={rid!r},
    repo_type="dataset",
    local_dir="./hub_ds",
    allow_patterns=["training/**"],
    token=os.environ["HF_TOKEN"],
)
PY
chmod +x ./hub_ds/training/runpod_entry.sh
./hub_ds/training/runpod_entry.sh
"""


__all__ = [
    "PIP_PACKAGES",
    "generate_runpod_config",
    "write_training_bundle_to_directory",
    "REMOTE_TRAIN_PY",
]
