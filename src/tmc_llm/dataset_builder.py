from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .document_loader import LoadedDocument, discover_documents, load_documents
from .text_cleaning import compact_spaces


SYSTEM_PROMPT = (
    "You are TMC-LM, an offline assistant for Trinidad Municipal College. "
    "Answer using only the provided official TMC knowledge. "
    "If the source does not contain the answer, say that the available TMC source does not contain it."
)


@dataclass(frozen=True)
class Example:
    user: str
    assistant: str
    source: str

    def to_json(self) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ],
            "source": self.source,
        }


def make_source_prompt(kind: str, document: LoadedDocument, detail: str) -> str:
    lines = [
        "Official TMC source knowledge.",
        f"Kind: {kind}",
        f"File: {document.path.as_posix()}",
    ]
    if detail:
        lines.append(detail)
    lines.append("Store this as official institutional knowledge for future user questions.")
    return "\n".join(lines)


def format_document_for_training(document: LoadedDocument) -> str:
    return f"SOURCE FILE: {document.path.as_posix()}\n{document.text}"


def extract_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "GENERAL"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if re.fullmatch(r"[A-Z][A-Z0-9 &,./'()\-]+:", line):
            current = line.rstrip(":").strip()
            sections.setdefault(current, [])
            continue

        sections.setdefault(current, []).append(line)

    return {name: compact_spaces(" ".join(lines)) for name, lines in sections.items() if lines}


def chunk_text(text: str, max_words: int = 130) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def document_title(path: Path) -> str:
    title = path.stem.replace("_", " ").replace("-", " ").strip()
    return title or path.name


def normalize_label(label: str) -> str:
    label = compact_spaces(label.strip(" :.-").replace("_", " "))
    return label.lower()


def extract_label_values(text: str) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = re.match(r"^\s*([A-Za-z][A-Za-z0-9 &/()' .-]{1,80})\s*:\s*(.+?)\s*$", line)
        if not match:
            continue

        label = compact_spaces(match.group(1))
        value = compact_spaces(match.group(2))
        if len(value.split()) < 3:
            continue
        labels.append((label, value))

    return labels


def build_label_examples(documents: list[LoadedDocument]) -> list[Example]:
    examples: list[Example] = []
    for document in documents:
        for label, value in extract_label_values(document.text):
            label_text = normalize_label(label)
            source = f"label:{document.path.as_posix()}:{label_text}"
            examples.append(
                Example(
                    make_source_prompt("label-value", document, f"Label: {label_text}"),
                    value,
                    source,
                )
            )

    return examples


def build_section_examples(document: LoadedDocument, sections: dict[str, str]) -> list[Example]:
    examples: list[Example] = []
    for name, body in sections.items():
        if not body:
            continue
        for index, chunk in enumerate(chunk_text(body)):
            section_name = compact_spaces(name.title())
            source = f"section:{document.path.as_posix()}:{name.lower().replace(' ', '_')}:{index}"
            examples.append(
                Example(
                    make_source_prompt("section", document, f"Section: {section_name}"),
                    chunk,
                    source,
                )
            )
    return examples


def build_all_section_examples(documents: list[LoadedDocument]) -> list[Example]:
    examples: list[Example] = []
    for document in documents:
        examples.extend(build_section_examples(document, extract_sections(document.text)))
    return examples


def build_document_examples(documents: list[LoadedDocument]) -> list[Example]:
    examples: list[Example] = []
    for document in documents:
        for index, chunk in enumerate(chunk_text(document.text, max_words=150)):
            source = f"file:{document.path.as_posix()}:{index}"
            examples.append(
                Example(
                    make_source_prompt("document-chunk", document, f"Chunk: {index + 1}"),
                    chunk,
                    source,
                )
            )
    return examples


def split_examples(examples: list[Example], seed: int = 42) -> tuple[list[Example], list[Example], list[Example]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    validation_size = max(1, round(total * 0.15))
    test_size = max(1, round(total * 0.15))
    train_size = max(1, total - validation_size - test_size)

    train = shuffled[:train_size]
    validation = shuffled[train_size : train_size + validation_size]
    test = shuffled[train_size + validation_size :]
    return train, validation, test


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_corpus(path: Path, documents: list[LoadedDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n".join(format_document_for_training(document) for document in documents)
    path.write_text(content, encoding="utf-8", newline="\n")


def collect_source_paths(source: Path | None, source_dir: Path | None) -> list[Path]:
    paths: list[Path] = []
    if source and source.exists():
        paths.append(source)
    if source_dir:
        paths.extend(discover_documents(source_dir))

    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)

    return unique_paths


def build_dataset(source: Path | None, output_dir: Path, source_dir: Path | None = None) -> dict[str, int]:
    source_paths = collect_source_paths(source, source_dir)
    if not source_paths:
        raise FileNotFoundError("No source documents found. Add official TMC files to data/raw/tmc_sources.")

    documents = load_documents(source_paths)
    text = "\n\n".join(format_document_for_training(document) for document in documents)
    sections = extract_sections(text)
    examples = (
        build_label_examples(documents)
        + build_all_section_examples(documents)
        + build_document_examples(documents)
    )

    seen: set[tuple[str, str]] = set()
    unique_examples: list[Example] = []
    for example in examples:
        key = (example.user, example.assistant)
        if key not in seen:
            seen.add(key)
            unique_examples.append(example)

    train, validation, test = split_examples(unique_examples)

    write_jsonl(output_dir / "dataset.jsonl", (example.to_json() for example in unique_examples))
    write_jsonl(output_dir / "train.jsonl", (example.to_json() for example in train))
    write_jsonl(output_dir / "validation.jsonl", (example.to_json() for example in validation))
    write_jsonl(output_dir / "test.jsonl", (example.to_json() for example in test))
    write_corpus(output_dir / "corpus.txt", documents)

    metadata = {
        "source_files": [str(document.path) for document in documents],
        "total_examples": len(unique_examples),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "test_examples": len(test),
        "sections": sorted(sections),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "sources_manifest.json").write_text(
        json.dumps(
            [
                {
                    "path": str(document.path),
                    "extension": document.path.suffix.lower(),
                    "characters": len(document.text),
                }
                for document in documents
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TMC-LM chat dataset from source documents.")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/tmc_sources"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_dataset(args.source, args.output_dir, args.source_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
