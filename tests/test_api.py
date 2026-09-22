"""Tests for the HTTP API (tmc_llm.api)."""

from __future__ import annotations

import concurrent.futures
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tmc_llm import api


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    """TestClient wired to a fake model path with a reset model cache."""
    monkeypatch.setattr(api, "find_model_path", lambda *_: tmp_path / "fake.gguf")
    monkeypatch.setattr(api, "_MODEL", None)
    monkeypatch.setattr(api, "_MODEL_PATH", None)
    return TestClient(api.app)


def _install_fake_llama(monkeypatch: pytest.MonkeyPatch, answer: str = "hello") -> list[Any]:
    """Install a fake llama_cpp module; returns the list of created instances."""
    created: list[Any] = []

    class FakeLlama:
        def __init__(self, model_path: str, n_ctx: int, verbose: bool) -> None:
            self.model_path = model_path
            created.append(self)

        def create_chat_completion(self, messages: list[dict], max_tokens: int, temperature: float) -> dict:
            return {"choices": [{"message": {"content": f"  {answer}  "}}]}

    fake = types.ModuleType("llama_cpp")
    fake.Llama = FakeLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    return created


def test_root_reports_ok(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_chat_serves_web_ui(client: TestClient) -> None:
    response = client.get("/chat")
    assert response.status_code == 200
    assert "TMC-LM" in response.text


def test_query_returns_answer(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, float]] = []

    def fake_generate(model_path: Path, prompt: str, ctx_size: int, temp: float) -> str:
        calls.append((prompt, ctx_size, temp))
        return "The vision of TMC is service."

    monkeypatch.setattr(api, "_generate_answer", fake_generate)

    response = client.post("/query", json={"prompt": "What is the vision of TMC?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "The vision of TMC is service."}
    assert calls == [("What is the vision of TMC?", 2048, 0.2)]


def test_query_uses_default_prompt_when_blank(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    def fake_generate(model_path: Path, prompt: str, ctx_size: int, temp: float) -> str:
        prompts.append(prompt)
        return "ok"

    monkeypatch.setattr(api, "_generate_answer", fake_generate)

    client.post("/query", json={"prompt": "   "})

    assert prompts == ["What is TMC's vision?"]


def test_query_rejects_oversized_ctx_size(client: TestClient) -> None:
    response = client.post("/query", json={"prompt": "hi", "ctx_size": 10**9})
    assert response.status_code == 422


def test_query_rejects_out_of_range_temp(client: TestClient) -> None:
    response = client.post("/query", json={"prompt": "hi", "temp": 99.0})
    assert response.status_code == 422


def test_query_rejects_overlong_prompt(client: TestClient) -> None:
    response = client.post("/query", json={"prompt": "x" * 3000})
    assert response.status_code == 422


def test_query_masks_inference_errors(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(model_path: Path, prompt: str, ctx_size: int, temp: float) -> str:
        raise RuntimeError("secret detail: C:/models/gguf/hidden.gguf")

    monkeypatch.setattr(api, "_generate_answer", boom)

    response = client.post("/query", json={"prompt": "hi"})

    assert response.status_code == 500
    assert "secret detail" not in response.text
    assert response.json()["detail"] == "Inference failed. Check the server logs."


def test_query_reports_missing_llama_cpp(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_bindings(model_path: Path, prompt: str, ctx_size: int, temp: float) -> str:
        raise ImportError("No module named 'llama_cpp'")

    monkeypatch.setattr(api, "_generate_answer", no_bindings)

    response = client.post("/query", json={"prompt": "hi"})

    assert response.status_code == 500
    assert "llama-cpp-python is not installed" in response.json()["detail"]


def test_query_without_model_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_model() -> Path:
        raise HTTPException(status_code=500, detail="No GGUF model found.")

    monkeypatch.setattr(api, "find_model_path", no_model)
    unpatched_client = TestClient(api.app)

    response = unpatched_client.post("/query", json={"prompt": "hi"})

    assert response.status_code == 500
    assert response.json()["detail"] == "No GGUF model found."


def test_local_inference_instructions_contain_valid_docker_command(client: TestClient) -> None:
    response = client.get("/api/local-inference")

    assert response.status_code == 200
    assert "ghcr.io/ggml-org/llama.cpp:light" in response.text
    assert "-cnv" in response.text
    assert "-m /models/fake.gguf" in response.text
    # The old, broken command referenced paths that do not exist in the image
    assert "/app/llama.cpp" not in response.text


def test_get_loaded_model_caches_instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created = _install_fake_llama(monkeypatch)
    model_path = tmp_path / "m.gguf"

    first = api.get_loaded_model(model_path, 2048)
    second = api.get_loaded_model(model_path, 2048)

    assert first is second
    assert len(created) == 1


def test_generate_answer_uses_cached_model_and_strips_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created = _install_fake_llama(monkeypatch, answer="the answer")
    monkeypatch.setattr(api, "_MODEL", None)
    monkeypatch.setattr(api, "_MODEL_PATH", None)

    answer = api._generate_answer(tmp_path / "m.gguf", "hi", 2048, 0.2)

    assert answer == "the answer"
    assert len(created) == 1


def test_generate_answer_serializes_concurrent_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """llama-cpp is not thread-safe: inference must never overlap."""
    _install_fake_llama(monkeypatch)
    monkeypatch.setattr(api, "_MODEL", None)
    monkeypatch.setattr(api, "_MODEL_PATH", None)

    guard = threading.Lock()
    active = 0
    max_active = 0
    fake_llama_cls = sys.modules["llama_cpp"].Llama
    original_method = fake_llama_cls.create_chat_completion

    def slow_chat(self: Any, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            return original_method(self, messages, max_tokens=max_tokens, temperature=temperature)
        finally:
            with guard:
                active -= 1

    fake_llama_cls.create_chat_completion = slow_chat

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(api._generate_answer, tmp_path / "m.gguf", "hi", 2048, 0.2) for _ in range(3)]
        results = [future.result(timeout=15) for future in futures]

    assert results == ["hello", "hello", "hello"]
    assert max_active == 1
