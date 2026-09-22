"""Tests for the CLI helpers (tmc_llm.cli)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from tmc_llm import cli


def test_run_docker_inference_prints_working_command(capsys: pytest.CaptureFixture[str]) -> None:
    cli.run_docker_inference(Path("models/gguf/tmc-lm-tinyllama-q4_k_m.gguf"), "What is TMC's vision?")

    out = capsys.readouterr().out
    assert "ghcr.io/ggml-org/llama.cpp:light" in out
    assert "-cnv" in out
    assert "-m /models/tmc-lm-tinyllama-q4_k_m.gguf" in out
    assert "What is TMC's vision?" in out
    # The old command used paths that do not exist inside the llama.cpp image
    assert "/app/llama.cpp" not in out
    assert "llama.cpp:full" not in out


def test_run_local_inference_without_llama_cpp_falls_back(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Setting a module to None in sys.modules makes `import` raise ImportError
    monkeypatch.setitem(sys.modules, "llama_cpp", None)

    result = cli.run_local_inference(tmp_path / "model.gguf", "hi")

    assert result == ""
    captured = capsys.readouterr()
    assert "llama-cpp-python is not installed" in captured.err
    assert "ghcr.io/ggml-org/llama.cpp:light" in captured.out


def test_run_local_inference_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("llama_cpp")

    class FakeLlama:
        def __init__(self, model_path: str, n_ctx: int, verbose: bool) -> None:
            pass

        def create_chat_completion(self, messages: list[dict], max_tokens: int, temperature: float) -> dict:
            return {"choices": [{"message": {"content": "  hello there  "}}]}

    fake.Llama = FakeLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)

    result = cli.run_local_inference(Path("model.gguf"), "hi")

    assert result == "hello there"


def test_find_model_prefers_existing_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = tmp_path / "custom.gguf"
    fake_model.write_text("gguf", encoding="utf-8")
    monkeypatch.setattr(cli, "MODEL_CANDIDATES", [tmp_path / "missing.gguf", fake_model])

    assert cli.find_model() == fake_model
    assert cli.find_model(fake_model) == fake_model


def test_find_model_returns_none_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "MODEL_CANDIDATES", [tmp_path / "missing.gguf"])

    assert cli.find_model() is None
