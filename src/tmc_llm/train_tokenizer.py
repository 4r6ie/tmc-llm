from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import sentencepiece as spm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_PREFIX = "You are TMC-LM, an offline assistant for Trinidad Municipal College. "


def collect_training_text(corpus_path: Path, extra_dirs: list[Path] | None = None) -> str:
    """Gather all plain-text content for tokenizer training."""
    texts: list[str] = []

    if corpus_path.exists():
        texts.append(corpus_path.read_text(encoding="utf-8"))

    for raw_dir in extra_dirs or []:
        if not raw_dir.exists():
            continue
        for ext in ("*.txt", "*.md"):
            for path in raw_dir.rglob(ext):
                try:
                    texts.append(path.read_text(encoding="utf-8"))
                except Exception:
                    logger.warning("Skipping unreadable file: %s", path)

    return "\n".join(texts)


def write_training_corpus(text: str, output_path: Path) -> None:
    """Write a single-line-per-paragraph corpus file for SentencePiece."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    output_path.write_text("\n".join(paragraphs), encoding="utf-8")


def train_sentencepiece(
    corpus_path: Path,
    model_prefix: str,
    vocab_size: int = 8000,
    model_type: str = "bpe",
) -> None:
    """Train a SentencePiece model."""
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        num_threads=4,
        split_digits=True,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    logger.info("SentencePiece model saved: %s.model", model_prefix)


def save_tokenizer_metadata(output_dir: Path, vocab_size: int, model_type: str) -> None:
    metadata = {
        "type": "sentencepiece",
        "vocab_size": vocab_size,
        "model_type": model_type,
        "special_tokens": {
            "pad": 0,
            "unk": 1,
            "bos": 2,
            "eos": 3,
        },
        "system_prompt_prefix": SYSTEM_PROMPT_PREFIX,
    }
    (output_dir / "tokenizer_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_training(
    corpus_path: Path,
    output_dir: Path,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    extra_dirs: list[Path] | None = None,
) -> Path:
    """Full pipeline: collect text -> train SentencePiece -> save metadata."""
    text = collect_training_text(corpus_path, extra_dirs)
    if not text.strip():
        raise ValueError(f"No training text found starting from {corpus_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_corpus = output_dir / "corpus_for_sp.txt"
    write_training_corpus(text, temp_corpus)

    model_prefix = str(output_dir / "tmc_tokenizer")
    train_sentencepiece(temp_corpus, model_prefix, vocab_size, model_type)
    save_tokenizer_metadata(output_dir, vocab_size, model_type)

    logger.info("Tokenizer training complete. Model at: %s.model", model_prefix)
    return Path(model_prefix + ".model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom SentencePiece tokenizer for TMC-LM.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/corpus.txt"),
        help="Path to the corpus text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/tokenizer"),
        help="Directory to save the trained tokenizer.",
    )
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--model-type", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument(
        "--extra-dirs",
        type=Path,
        nargs="*",
        default=[Path("data/raw/tmc_sources")],
        help="Additional directories with raw text for training.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    run_training(args.corpus, args.output_dir, args.vocab_size, args.model_type, args.extra_dirs)


if __name__ == "__main__":
    main()
