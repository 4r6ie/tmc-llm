from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .versioning import VERSIONS_FILE, update_version_artifacts


def _read_version(adapter_dir: Path) -> str | None:
    """Read the version recorded by train_lora from the adapter's metadata.json."""
    metadata_path = adapter_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version = metadata.get("version")
    return str(version) if version else None


def merge_lora(base_model: str, adapter_dir: Path, output_dir: Path) -> None:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)

    version = _read_version(adapter_dir)
    if version:
        update_version_artifacts(version, merged_dir=output_dir, status="merged")
        print(f"Registered merged model for version {version} in {VERSIONS_FILE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into TinyLlama.")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter-dir", type=Path, default=Path("models/adapters/tmc-lm-tinyllama-lora-v1.0"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/merged/tmc-lm-tinyllama-v1.0"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_lora(args.base_model, args.adapter_dir, args.output_dir)


if __name__ == "__main__":
    main()
