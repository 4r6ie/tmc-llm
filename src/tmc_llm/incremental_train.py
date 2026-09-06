from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .dataset_builder import build_dataset
from .train_lora import (
    PromptAwareDataCollator,
    build_training_arguments,
    load_jsonl,
    pick_dtype,
    quiet_external_noise,
    tokenize_dataset,
)

logger = logging.getLogger(__name__)


def get_next_version(base_dir: Path, current_version: str) -> str:
    """Increment the minor version number."""
    parts = current_version.split(".")
    if len(parts) == 2:
        major, minor = int(parts[0]), int(parts[1])
        return f"{major}.{minor + 1}"
    return current_version + ".1"


def merge_adapter_into_model(model: Any, adapter_dir: Path) -> Any:
    """Load an existing adapter and merge it before applying a new one."""
    if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        logger.info("Merged existing adapter from %s", adapter_dir)
    return model


def incremental_train(
    config_path: Path,
    new_source: Path | None = None,
    new_source_dir: Path | None = None,
    base_adapter: Path | None = None,
) -> dict[str, Any]:
    """Run incremental fine-tuning on new documents."""
    quiet_external_noise()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    base_model = config["base_model"]
    current_version = config.get("version", "1.0")
    next_version = get_next_version(Path(config["output_dir"]), current_version)
    logger.info("Incremental training: %s -> %s", current_version, next_version)

    # Rebuild dataset with new source documents
    output_dir = Path(config["dataset_train"]).parent
    logger.info("Rebuilding dataset with new sources...")
    metadata = build_dataset(new_source, output_dir, new_source_dir)
    logger.info("Dataset rebuilt: %d total examples", metadata["total_examples"])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model: Any = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=pick_dtype(config),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    # Merge any existing adapter before applying new LoRA
    adapter_dir = Path(config["output_dir"])
    if base_adapter:
        adapter_dir = base_adapter
    model = merge_adapter_into_model(model, adapter_dir)

    # Apply new LoRA configuration
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    train_dataset = tokenize_dataset(
        tokenizer,
        load_jsonl(Path(config["dataset_train"])),
        config["max_seq_length"],
    )
    validation_dataset = tokenize_dataset(
        tokenizer,
        load_jsonl(Path(config["dataset_validation"])),
        config["max_seq_length"],
    )

    if len(train_dataset) == 0:
        raise ValueError("No training examples survived tokenization.")
    if len(validation_dataset) == 0:
        raise ValueError("No validation examples survived tokenization.")

    # Update output directory for new version
    new_output_dir = adapter_dir.parent / f"tmc-lm-tinyllama-lora-v{next_version}"
    new_output_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs = {
        "output_dir": str(new_output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": config["num_train_epochs"],
        "max_steps": config["max_steps"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "warmup_steps": config["warmup_steps"],
        "logging_steps": config["logging_steps"],
        "save_steps": config["save_steps"],
        "eval_steps": config["eval_steps"],
        "save_total_limit": 2,
        "report_to": [],
        "fp16": bool(config.get("fp16", False)) and torch.cuda.is_available(),
        "bf16": bool(config.get("bf16", False)) and torch.cuda.is_available(),
        "lr_scheduler_type": config.get("lr_scheduler_type", "linear"),
        "weight_decay": config.get("weight_decay", 0.0),
        "dataloader_pin_memory": False,
    }

    supported_args = set(inspect.signature(TrainingArguments).parameters)
    if "eval_strategy" in supported_args:
        training_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported_args:
        training_kwargs["evaluation_strategy"] = "steps"

    args = build_training_arguments(training_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=PromptAwareDataCollator(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(new_output_dir)
    tokenizer.save_pretrained(new_output_dir)

    result_metadata = {
        "previous_version": current_version,
        "new_version": next_version,
        "base_model": base_model,
        "base_adapter": str(base_adapter) if base_adapter else None,
        "new_source": str(new_source) if new_source else None,
        "new_source_dir": str(new_source_dir) if new_source_dir else None,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "output_dir": str(new_output_dir),
    }
    meta_path = new_output_dir / "metadata.json"
    meta_path.write_text(json.dumps(result_metadata, indent=2), encoding="utf-8")
    logger.info("Incremental training complete. v%s saved to %s", next_version, new_output_dir)
    return result_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incremental fine-tuning for TMC-LM with new documents.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_lora.yaml"))
    parser.add_argument("--new-source", type=Path, default=None, help="New single source file.")
    parser.add_argument("--new-source-dir", type=Path, default=None, help="Directory with new source documents.")
    parser.add_argument("--base-adapter", type=Path, default=None, help="Path to existing adapter to build on.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    result = incremental_train(args.config, args.new_source, args.new_source_dir, args.base_adapter)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
