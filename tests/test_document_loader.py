import json
from pathlib import Path

import fitz

from tmc_llm.document_loader import discover_documents, load_document


def test_discover_documents_finds_supported_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "manual.txt").write_text("TMC Manual", encoding="utf-8")
    (raw_dir / "ignore.exe").write_text("ignore", encoding="utf-8")

    paths = discover_documents(raw_dir)

    assert paths == [raw_dir / "manual.txt"]


def test_load_json_flattens_content(tmp_path: Path) -> None:
    source = tmp_path / "manual.json"
    source.write_text(json.dumps({"vision": "A model institution"}), encoding="utf-8")

    document = load_document(source)

    assert "vision: A model institution" in document.text


def test_load_pdf_falls_back_to_ocr_for_blank_page(tmp_path: Path) -> None:
    source = tmp_path / "scanned.pdf"
    pdf = fitz.open()
    pdf.new_page(width=612, height=792)
    pdf.save(source)
    pdf.close()

    document = load_document(source)

    assert "Page 1" in document.text
    assert ("OCR unavailable" in document.text) or ("OCR failed" in document.text)

