from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are TMC-LM, an offline assistant for Trinidad Municipal College. "
    "Answer using only the provided official TMC knowledge. "
    "If the source does not contain the answer, say that the available TMC source does not contain it."
)

NEGATIVE_KEYWORDS = [
    "does not contain this information",
    "does not contain the answer",
    "not available in the TMC source",
    "contact the appropriate",
    "contact the",
]


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_qa_pairs(entries: list[dict]) -> list[tuple[str, str]]:
    """Extract (user_question, expected_answer) pairs from dataset entries."""
    pairs = []
    for entry in entries:
        messages = entry.get("messages", [])
        user_msg = None
        assistant_msg = None
        for msg in messages:
            if msg.get("role") == "user":
                user_msg = msg["content"]
            elif msg.get("role") == "assistant":
                assistant_msg = msg["content"]
        if user_msg and assistant_msg:
            pairs.append((user_msg, assistant_msg))
    return pairs


def generate_predictions(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    max_new_tokens: int = 256,
) -> list[str]:
    """Generate predictions for a list of questions."""
    predictions = []
    model.eval()
    for question in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        if getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = f"SYSTEM: {SYSTEM_PROMPT}\nUSER: {question}\nASSISTANT:"

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=768)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        predictions.append(response)

    return predictions


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Compute exact match accuracy (case-insensitive, stripped)."""
    if not predictions:
        return 0.0
    correct = sum(
        1
        for pred, ref in zip(predictions, references, strict=False)
        if pred.strip().lower() == ref.strip().lower()
    )
    return correct / len(predictions)


def compute_keyword_overlap(predictions: list[str], references: list[str]) -> float:
    """Compute average keyword overlap (Jaccard similarity of word sets)."""
    if not predictions:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references, strict=False):
        pred_words = set(re.findall(r"\w+", pred.lower()))
        ref_words = set(re.findall(r"\w+", ref.lower()))
        if not ref_words:
            scores.append(1.0 if not pred_words else 0.0)
            continue
        intersection = pred_words & ref_words
        union = pred_words | ref_words
        scores.append(len(intersection) / len(union) if union else 0.0)
    return sum(scores) / len(scores)


def compute_negative_rejection(predictions: list[str], references: list[str]) -> float:
    """Measure how often negative questions are answered with a rejection."""
    if not predictions:
        return 0.0
    rejected = 0
    for pred, ref in zip(predictions, references, strict=False):
        is_negative_ref = any(kw in ref.lower() for kw in NEGATIVE_KEYWORDS)
        if is_negative_ref:
            has_rejection = any(kw in pred.lower() for kw in NEGATIVE_KEYWORDS)
            if has_rejection:
                rejected += 1
    negative_count = sum(1 for ref in references if any(kw in ref.lower() for kw in NEGATIVE_KEYWORDS))
    return rejected / negative_count if negative_count > 0 else 1.0


def compute_average_response_length(predictions: list[str]) -> float:
    """Average character length of predictions."""
    if not predictions:
        return 0.0
    return sum(len(p) for p in predictions) / len(predictions)


def compute_perplexity(model: Any, tokenizer: Any, texts: list[str], max_length: int = 512) -> float:
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()

        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item() * labels.shape[1]
            total_tokens += labels.shape[1]

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return float(torch.exp(torch.tensor(avg_loss)).item())


def run_evaluation(
    test_path: Path,
    model: Any | None = None,
    tokenizer: Any | None = None,
    sample_size: int = 50,
) -> dict[str, Any]:
    """Run full evaluation pipeline."""
    entries = load_jsonl(test_path)
    qa_pairs = extract_qa_pairs(entries)

    results: dict[str, Any] = {
        "test_entries": len(entries),
        "qa_pairs": len(qa_pairs),
    }

    if model is None or tokenizer is None:
        logger.warning("No model loaded. Running dataset-only metrics.")
        results["model_loaded"] = False
        return results

    results["model_loaded"] = True
    sampled = qa_pairs[:sample_size]
    questions = [q for q, _ in sampled]
    references = [a for _, a in sampled]

    logger.info("Generating predictions for %d questions...", len(questions))
    predictions = generate_predictions(model, tokenizer, questions)

    results["exact_match"] = round(compute_exact_match(predictions, references), 4)
    results["keyword_overlap"] = round(compute_keyword_overlap(predictions, references), 4)
    results["negative_rejection"] = round(compute_negative_rejection(predictions, references), 4)
    results["avg_response_length"] = round(compute_average_response_length(predictions), 2)

    logger.info("Computing perplexity...")
    texts = [f"User: {q}\nAssistant: {a}" for q, a in sampled]
    results["perplexity"] = round(compute_perplexity(model, tokenizer, texts), 4)

    results["predictions_sample"] = [
        {"question": q, "reference": r, "prediction": p}
        for q, r, p in zip(questions[:5], references[:5], predictions[:5], strict=False)
    ]

    return results


def load_model_for_eval(base_model: str, adapter_dir: Path | None = None) -> tuple[Any, Any]:
    """Load model and tokenizer for evaluation."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if adapter_dir and adapter_dir.exists():
        model = PeftModel.from_pretrained(model, adapter_dir)
        logger.info("Loaded adapter from %s", adapter_dir)

    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TMC-LM on the test dataset.")
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/processed/test.jsonl"),
    )
    parser.add_argument(
        "--base-model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("models/adapters/tmc-lm-tinyllama-lora-v1.0"),
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None, help="Path to save evaluation results JSON.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if not args.test_data.exists():
        print(f"Test data not found: {args.test_data}", file=sys.stderr)
        sys.exit(1)

    model, tokenizer = load_model_for_eval(args.base_model, args.adapter_dir)
    results = run_evaluation(args.test_data, model, tokenizer, args.sample_size)

    output_json = json.dumps(results, indent=2, ensure_ascii=False)
    print(output_json)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
