from __future__ import annotations

from pathlib import Path

from tmc_llm.train_tokenizer import (
    collect_training_text,
    save_tokenizer_metadata,
    write_training_corpus,
)


class TestCollectTrainingText:
    def test_collects_from_corpus(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("Hello TMC world", encoding="utf-8")
        result = collect_training_text(corpus)
        assert "Hello TMC world" in result

    def test_collects_from_extra_dirs(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("base text", encoding="utf-8")
        extra = tmp_path / "extra"
        extra.mkdir()
        (extra / "notes.txt").write_text("extra notes", encoding="utf-8")
        result = collect_training_text(corpus, [extra])
        assert "base text" in result
        assert "extra notes" in result

    def test_handles_missing_corpus(self, tmp_path: Path) -> None:
        result = collect_training_text(tmp_path / "nonexistent.txt")
        assert result == ""


class TestWriteTrainingCorpus:
    def test_writes_paragraphs(self, tmp_path: Path) -> None:
        out = tmp_path / "out.txt"
        write_training_corpus("Para one\n\nPara two\n\nPara three", out)
        content = out.read_text(encoding="utf-8")
        assert "Para one" in content
        assert "Para two" in content


class TestSaveTokenizerMetadata:
    def test_writes_metadata(self, tmp_path: Path) -> None:
        save_tokenizer_metadata(tmp_path, 32000, "bpe")
        meta = (tmp_path / "tokenizer_metadata.json").read_text(encoding="utf-8")
        assert "32000" in meta
        assert "bpe" in meta
