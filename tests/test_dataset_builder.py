from pathlib import Path

from tmc_llm.dataset_builder import build_dataset, build_label_examples, extract_label_values, extract_sections
from tmc_llm.document_loader import LoadedDocument

SAMPLE_TEXT = """
BRIEF HISTORY:
It started in 1980 when Mr. Paciano Petarco initiated the plan.
Trinidad Junior College was registered with the Securities and Exchange Commission in 1985 with SEC No. 136393.

VISION, MISSION, GOAL, PHILOSOPHY & SLOGAN:
VISION: A model institution with fully developed academic, technical-vocational education and skill of manpower.
MISSION: To build well-trained, competent and employable professionals.
GOAL: TMC aims at evolving a whole individual as a child of God.
PHILOSOPHY: TMC adheres to the philosophy that education is life and growth.
SLOGAN: TMC is committed to public educational services second to none.

ACADEMIC PROGRAMS:
Bachelor of Science in Information Technology
Bachelor of Science in Criminology
Shielded Metal Arc Welding NC I
Shielded Metal Arc Welding NC II
"""


def test_extract_sections_keeps_main_headers() -> None:
    sections = extract_sections(SAMPLE_TEXT)
    assert "BRIEF HISTORY" in sections
    assert "ACADEMIC PROGRAMS" in sections


def test_extract_label_values_is_dynamic() -> None:
    labels = dict(extract_label_values(SAMPLE_TEXT))

    assert labels["VISION"].startswith("A model institution")
    assert labels["MISSION"].startswith("To build well-trained")


def test_build_label_examples_uses_source_content_without_hardcoded_facts() -> None:
    document = LoadedDocument(Path("sample_manual.txt"), SAMPLE_TEXT)
    examples = build_label_examples([document])
    prompts = {example.user for example in examples}
    answers = " ".join(example.assistant for example in examples)

    assert any("Label: vision" in prompt for prompt in prompts)
    assert not any(prompt.startswith("What is") for prompt in prompts)
    assert "employable professionals" in answers


def test_build_dataset_writes_expected_files(tmp_path: Path) -> None:
    source = tmp_path / "train.txt"
    output_dir = tmp_path / "processed"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    metadata = build_dataset(source, output_dir)

    assert metadata["total_examples"] >= 9
    assert metadata["source_files"] == [str(source)]
    assert (output_dir / "dataset.jsonl").exists()
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    assert (output_dir / "corpus.txt").exists()
    assert (output_dir / "sources_manifest.json").exists()


def test_build_dataset_from_source_folder(tmp_path: Path) -> None:
    source_dir = tmp_path / "data" / "raw" / "tmc_sources"
    output_dir = tmp_path / "processed"
    source_dir.mkdir(parents=True)
    source = source_dir / "train.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    metadata = build_dataset(None, output_dir, source_dir)

    assert metadata["source_files"] == [str(source)]
    assert metadata["total_examples"] >= 9
