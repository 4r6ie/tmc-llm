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
    DataCollatorForLanguageModeling,
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
    texts = [format_messages(tokenizer, row["messages"]) for row in rows]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


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
    training_kwargs = {
        "output_dir": output_dir,
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
        "fp16": torch.cuda.is_available(),
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
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama with LoRA for TMC-LM.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_lora.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
