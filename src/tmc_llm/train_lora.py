from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import warnings
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as transformers_logging


def quiet_external_noise() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    transformers_logging.set_verbosity_error()
    warnings.filterwarnings(
        "ignore",
        message=r".*unauthenticated requests to the HF Hub.*",
    )


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_messages(tokenizer: AutoTokenizer, messages: list[dict]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    parts = []
    for message in messages:
        role = message["role"].upper()
        parts.append(f"{role}: {message['content']}")
    return "\n".join(parts) + tokenizer.eos_token


def tokenize_dataset(tokenizer: AutoTokenizer, rows: list[dict], max_length: int) -> Dataset:
    samples: list[dict] = []
    for row in rows:
        messages = row["messages"]
        if not messages:
            continue

        assistant_index = None
        for index, message in enumerate(messages):
            if message.get("role") == "assistant":
                assistant_index = index
                break
        if assistant_index is None:
            continue

        prompt_messages = messages[:assistant_index]
        full_messages = messages[: assistant_index + 1]

        prompt_text = format_messages(tokenizer, prompt_messages)
        full_text = format_messages(tokenizer, full_messages)

        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]
        full_ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]

        assistant_len = len(full_ids) - len(prompt_ids)
        if assistant_len <= 0:
            continue
        if assistant_len < 5:
            continue

        labels = [-100] * len(full_ids)
        labels[-assistant_len:] = full_ids[-assistant_len:]

        samples.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})

    return Dataset.from_list(samples)


class PromptAwareDataCollator:
    """Pads a batch without overwriting precomputed loss labels.

    DataCollatorForLanguageModeling(mlm=False) clones input_ids into labels,
    which destroys the -100 masks set in tokenize_dataset(). This collator only
    pads input_ids/attention_mask/labels and leaves existing label values alone.
    """

    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in features)
        if self.pad_to_multiple_of is not None:
            max_length = self.pad_to_multiple_of * (max_length // self.pad_to_multiple_of + 1)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def build_training_arguments(training_kwargs: dict) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments)
    supported_keys = set(signature.parameters)
    filtered_kwargs = {
        key: value
        for key, value in training_kwargs.items()
        if key in supported_keys
    }
    skipped_keys = sorted(set(training_kwargs) - set(filtered_kwargs))
    if skipped_keys:
        print(f"Skipped unsupported TrainingArguments keys: {', '.join(skipped_keys)}")

    return TrainingArguments(**filtered_kwargs)


def train(config_path: Path) -> None:
    quiet_external_noise()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_model = config["base_model"]
    version = config.get("version", "0.0")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=pick_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

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

    output_dir = config["output_dir"]
    versioned_dir = output_dir
    training_kwargs = {
        "output_dir": versioned_dir,
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
    trainer.save_model(versioned_dir)
    tokenizer.save_pretrained(versioned_dir)

    metadata = {
        "version": version,
        "base_model": base_model,
        "dataset_train": config["dataset_train"],
        "dataset_validation": config["dataset_validation"],
        "num_train_epochs": config["num_train_epochs"],
        "max_steps": config["max_steps"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "lora": config["lora"],
        "output_dir": versioned_dir,
    }
    metadata_path = Path(versioned_dir) / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Training complete. Model version {version} saved to {versioned_dir}")
    print(f"Metadata written to {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama with LoRA for TMC-LM.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_lora.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
