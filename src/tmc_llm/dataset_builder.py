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

NEGATIVE_ANSWER = (
    "The available TMC source does not contain this information. "
    "Please contact the appropriate TMC office for assistance."
)

SCHOOL_NAME = "Trinidad Municipal College"


NEGATIVE_QUESTIONS = [
    ("What is the tuition fee for international students at TMC?",
     "The available TMC source does not contain information about tuition fees for international students. Please contact the Registrar's Office or Accounting Office for details."),
    ("What is the exact salary of a TMC professor?",
     "The available TMC source does not contain specific salary figures. TMC follows the Salary Standardization Table for LGU employees. Please contact the HRMO for details."),
    ("How many computers does the TMC computer lab have?",
     "The available TMC source does not contain the exact number of computers in the TMC computer lab. Please contact the IT Department for details."),
    ("What is the Wi-Fi password at TMC?",
     "The available TMC source does not contain the Wi-Fi password. Please ask at the IT Department or Administration Office for Wi-Fi access details."),
    ("Does TMC have a swimming pool?",
     "The available TMC source does not mention a swimming pool at TMC. Please contact the Administration Office for information about campus facilities."),
    ("What is the current enrollment deadline for next semester?",
     "The available TMC source does not contain the current enrollment deadline. Please check the TMC website or contact the Registrar's Office for the latest enrollment schedule."),
    ("Who is the current Registrar of TMC?",
     "The available TMC source does not name the current Registrar. Please contact the Registrar's Office directly for the name of the current Registrar."),
    ("What programming languages are taught in the BSIT program?",
     "The available TMC source does not specify which programming languages are taught in the BSIT program. The program covers programming, networking, database management, and web development. Please contact the College of Computer Studies for specifics."),
    ("How much is the graduation fee at TMC?",
     "The available TMC source states the graduation fee is PHP 800.00. This covers ceremony expenses."),
    ("What is the student-to-computer ratio at TMC?",
     "The available TMC source does not contain the student-to-computer ratio. Please contact the College of Computer Studies or IT Department for details."),
    ("Does TMC offer online classes?",
     "The available TMC source does not specifically mention online class offerings. TMC provides morning, afternoon, and evening class schedules. Please contact the Registrar's Office for current class modalities."),
    ("What is the address of TMC?",
     "Trinidad Municipal College is located in the Municipality of Trinidad, Province of Bohol, Philippines. Please contact the Administration Office for the exact street address."),
    ("Does TMC have a basketball court?",
     "The available TMC source mentions open grounds and sports facilities on campus but does not specifically mention a basketball court. Please contact the Administration Office for details."),
    ("What is the process for filing a grade appeal at TMC?",
     "The available TMC source does not contain a specific grade appeal process. Grades are final once recorded and signed by the Department Head. Grade changes require approval from the Department Head and Academic Affairs. Please contact the Registrar's Office for guidance."),
    ("How many sections per class does TMC have?",
     "The available TMC source does not specify the number of sections per class. Please contact the Academic Affairs Office for details."),
]


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

        header = section_header(line)
        if header:
            current = header
            sections.setdefault(current, [])
            continue

        sections.setdefault(current, []).append(line)

    return {name: compact_spaces(" ".join(lines)) for name, lines in sections.items() if lines}


_SECTION_HEADER_PATTERNS = [
    re.compile(r"^[A-Z][A-Z0-9 &,./'()%\-]+:$"),
    re.compile(r"^SECTION\s+\d+(\.\d+)?:\s+[A-Za-z0-9][A-Za-z0-9 &,.'/()%\-]*$"),
    re.compile(r"^\d+\.\d+\s+[A-Z][A-Za-z0-9 &,.'/()%\-]+$"),
]


def section_header(line: str) -> str | None:
    for pattern in _SECTION_HEADER_PATTERNS:
        if pattern.fullmatch(line):
            return line.rstrip(":").strip()
    return None


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


LABEL_QUESTION_PHRASINGS: dict[str, list[str]] = {
    "vision": [
        "What is the vision of {school}?",
        "Can you tell me {school}'s vision?",
        "What is Trinidad Municipal College's vision statement?",
        "What does {school} aspire to become?",
    ],
    "mission": [
        "What is the mission of {school}?",
        "Can you tell me {school}'s mission?",
        "What is Trinidad Municipal College's mission statement?",
        "What is {school} committed to doing?",
    ],
    "goal": [
        "What is the goal of {school}?",
        "Can you tell me {school}'s goal?",
        "What are the institutional goals of Trinidad Municipal College?",
    ],
    "philosophy": [
        "What is the philosophy of {school}?",
        "Can you explain {school}'s philosophy?",
        "What does Trinidad Municipal College believe in?",
    ],
    "slogan": [
        "What is the slogan of {school}?",
        "Can you tell me {school}'s slogan?",
        "What is Trinidad Municipal College's motto or slogan?",
    ],
    "core values": [
        "What are the core values of {school}?",
        "Can you tell me {school}'s core values?",
        "What values does Trinidad Municipal College uphold?",
    ],
    "history": [
        "What is the history of {school}?",
        "Can you tell me how {school} was founded?",
        "Tell me about Trinidad Municipal College's beginnings.",
        "When was {school} established?",
    ],
}


def label_question_templates(label: str) -> list[str]:
    normalized = normalize_label(label)
    templates = LABEL_QUESTION_PHRASINGS.get(normalized, [])
    fallback = [
        f"What can you tell me about {normalized} at {SCHOOL_NAME}?",
        f"Describe {normalized} at Trinidad Municipal College.",
    ]
    return templates or fallback


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


def build_conversational_label_examples(documents: list[LoadedDocument]) -> list[Example]:
    examples: list[Example] = []
    for document in documents:
        for label, value in extract_label_values(document.text):
            label_text = normalize_label(label)
            templates = label_question_templates(label_text)
            source = f"qa-label:{document.path.as_posix()}:{label_text}"
            for template in templates:
                question = template.replace("{school}", "TMC")
                examples.append(
                    Example(
                        user=question,
                        assistant=value,
                        source=source,
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


def build_conversational_section_examples(document: LoadedDocument, sections: dict[str, str]) -> list[Example]:
    examples: list[Example] = []
    for name, body in sections.items():
        if not body:
            continue
        section_name = compact_spaces(name.title())
        question = f"What does TMC say about {section_name}?"
        source = f"qa-section:{document.path.as_posix()}:{name.lower().replace(' ', '_')}"
        examples.append(
            Example(
                user=question,
                assistant=body,
                source=source,
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


def build_all_conversational_section_examples(documents: list[LoadedDocument]) -> list[Example]:
    examples: list[Example] = []
    for document in documents:
        examples.extend(build_conversational_section_examples(document, extract_sections(document.text)))
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


def load_qa_pairs(qa_path: Path) -> list[Example]:
    examples: list[Example] = []
    if not qa_path.exists():
        return examples
    data = json.loads(qa_path.read_text(encoding="utf-8"))
    for item in data:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if not question or not answer:
            continue
        source = f"qa:{qa_path.as_posix()}:{question[:60]}"
        examples.append(
            Example(
                user=question,
                assistant=answer,
                source=source,
            )
        )
    return examples


def build_negative_examples(qa_path: Path | None = None) -> list[Example]:
    examples: list[Example] = []
    for question, answer in NEGATIVE_QUESTIONS:
        source = f"negative:curated:{question[:60]}"
        examples.append(
            Example(
                user=question,
                assistant=answer,
                source=source,
            )
        )
    return examples


PARAPHRASE_TEMPLATES = [
    "Can you tell me {question}?",
    "I'd like to know {question}.",
    "Please explain: {question}",
    "Could you answer this for me? {question}",
]


def paraphrase_question(question: str, template: str) -> str:
    text = question.strip()
    text = text.rstrip("?.")
    sentence = template.replace("{question}", text)
    return sentence.strip()


def build_paraphrase_examples(qa_examples: list[Example], variants: int = 3) -> list[Example]:
    paraphrased: list[Example] = []
    templates = PARAPHRASE_TEMPLATES[:variants]
    for example in qa_examples:
        for index, template in enumerate(templates):
            question = paraphrase_question(example.user, template)
            paraphrased.append(
                Example(
                    user=question,
                    assistant=example.assistant,
                    source=f"{example.source}:paraphrase:{index}",
                )
            )
    return paraphrased


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

    qa_path = source_dir / "tmc_qa.json" if source_dir else Path("data/raw/tmc_sources/tmc_qa.json")
    qa_examples = load_qa_pairs(qa_path) if qa_path.exists() else []
    paraphrase_examples = build_paraphrase_examples(qa_examples)
    negative_examples = build_negative_examples()

    examples = (
        qa_examples
        + paraphrase_examples
        + negative_examples
        + build_conversational_label_examples(documents)
        + build_all_conversational_section_examples(documents)
        + build_label_examples(documents)
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
        "qa_pairs": len(qa_examples),
        "qa_paraphrases": len(paraphrase_examples),
        "negative_examples": len(negative_examples),
        "conversational_label_examples": len(build_conversational_label_examples(documents)),
        "conversational_section_examples": len(build_all_conversational_section_examples(documents)),
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
