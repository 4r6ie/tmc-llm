"""Standalone QA fine-tuning script.

This mirrors `tmc_llm.train_lora` but only trains on the curated QA pairs in
`data/raw/tmc_sources/qa_for_training.jsonl`. It reuses the shared tokenization
and data-collation helpers so prompt tokens are masked out of the loss (labels
of -100), exactly like the main training pipeline.
"""

import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from src.tmc_llm.train_lora import (
    PromptAwareDataCollator,
    build_training_arguments,
    tokenize_dataset,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "models/adapters/tmc-lm-tinyllama-lora-qa-v1.0"
QA_PATH = Path("data/raw/tmc_sources/qa_for_training.jsonl")
MAX_SEQ_LENGTH = 768


def load_qa_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    rows = load_qa_rows(QA_PATH)
    print("Loaded", len(rows), "QA rows")

    dataset = tokenize_dataset(tokenizer, rows, max_length=MAX_SEQ_LENGTH)
    print("Tokenized dataset size:", len(dataset))
    if len(dataset) == 0:
        raise SystemExit("No training examples survived tokenization.")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_kwargs = {
        "output_dir": OUTPUT_DIR,
        "overwrite_output_dir": True,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 0.0002,
        "warmup_steps": 10,
        "logging_steps": 10,
        "save_steps": 50,
        "report_to": [],
        "fp16": torch.cuda.is_available(),
    }
    args = build_training_arguments(training_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=PromptAwareDataCollator(tokenizer=tokenizer),
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete and adapter saved!")


if __name__ == "__main__":
    main()
