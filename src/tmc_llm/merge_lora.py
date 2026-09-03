from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(base_model: str, adapter_dir: Path, output_dir: Path) -> None:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)


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
