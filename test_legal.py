#!/usr/bin/env python3
"""
Prueba rápida: base SmolLM2-135M + adaptadores LoRA locales → una pregunta jurídica
inspirada en una tesis real del almacén.

Uso (desde la raíz del proyecto):
  python test_legal.py
  python test_legal.py --adapter outputs
  python test_legal.py --adapter .training_local_output/checkpoint-5

Si copias la carpeta del checkpoint a ``outputs/``, el script la detecta sola.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SKIP_JSON = frozenset({"mapa_scjn.json", "config_mapa.json"})


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=True)
    except ImportError:
        pass


def find_adapter_dir(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        cfg = p / "adapter_config.json"
        if cfg.is_file():
            return p.resolve()
        raise FileNotFoundError(f"No hay adapter_config.json en {p}")

    candidates: list[Path] = []
    out = ROOT / "outputs"
    if out.is_dir():
        candidates.append(out)
        candidates.extend(sorted(out.glob("checkpoint-*")))
    candidates.append(ROOT / ".training_local_output")
    if (ROOT / ".training_local_output").is_dir():
        candidates.extend(sorted((ROOT / ".training_local_output").glob("checkpoint-*")))

    for c in candidates:
        if (c / "adapter_config.json").is_file():
            return c.resolve()

    raise FileNotFoundError(
        "No se encontró adapter_config.json. Entrena primero o copia el checkpoint a "
        "``outputs/`` o pasa ``--adapter ruta/al/checkpoint``."
    )


def pick_first_thesis_record() -> tuple[str, str, Path]:
    alm = ROOT / "almacen"
    if not alm.is_dir():
        raise FileNotFoundError(f"No existe la carpeta {alm}")

    paths = sorted(
        p
        for p in alm.rglob("*.json")
        if p.name not in SKIP_JSON and "config" not in p.name.lower()
    )
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        titulo = (raw.get("titulo") or "").strip()
        texto = (raw.get("texto") or "").strip()
        if titulo or texto:
            return titulo or "(sin título)", texto[:4000], path
    raise FileNotFoundError("No hay JSON de tesis con texto/título en almacen/")


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Prueba SmolLM2 + LoRA sobre corpus legal.")
    parser.add_argument(
        "--adapter",
        default=None,
        help="Carpeta que contiene adapter_config.json (por defecto: outputs/ o .training_local_output/checkpoint-*)",
    )
    parser.add_argument(
        "--base",
        default=os.getenv("HF_MODEL") or "HuggingFaceTB/SmolLM2-135M",
        help="ID del modelo base en Hugging Face",
    )
    args = parser.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print("Instala dependencias: pip install torch transformers peft", file=sys.stderr)
        raise SystemExit(1) from e

    adapter_dir = find_adapter_dir(args.adapter)
    titulo, texto_muestra, json_path = pick_first_thesis_record()

    base_id = args.base.strip().rstrip(".")
    print(f"[test_legal] Base: {base_id}")
    print(f"[test_legal] Adaptador: {adapter_dir}")
    print(f"[test_legal] Tesis de ejemplo: {json_path.relative_to(ROOT)}")
    print(f"[test_legal] Título: {titulo[:200]}{'…' if len(titulo) > 200 else ''}\n")

    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    model.to("cpu")

    pregunta = (
        "Según el estilo y el vocabulario propios de este corpus de jurisprudencia mexicana, "
        "¿cuál es el tema central o el rubro jurídico que atraviesa el siguiente asunto? "
        "Responde en español, de forma breve y técnica."
    )
    contexto = f"Título: {titulo}\nFragmento: {texto_muestra[:2500]}"

    messages = [{"role": "user", "content": f"{pregunta}\n\n{contexto}"}]
    try:
        if getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            raise ValueError("no chat template")
    except Exception:
        prompt = f"{pregunta}\n\n{contexto}\n\nRespuesta:"

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.pad_token_id,
        )
    gen = out_ids[0, inputs["input_ids"].shape[1] :]
    respuesta = tok.decode(gen, skip_special_tokens=True).strip()

    print("--- Pregunta (resumen) ---")
    print(pregunta)
    print("\n--- Respuesta del modelo ---")
    print(respuesta or "(vacío)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
