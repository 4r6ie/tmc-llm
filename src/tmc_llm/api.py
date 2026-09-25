from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from tmc_llm.cli import find_model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Log a warning if no GGUF model is found; the API still starts."""
    if find_model_path_if_exists() is None:
        print(
            "WARNING: No GGUF model found in models/gguf/. "
            "The /query endpoint will fail until you run the training pipeline "
            "or place a GGUF model in models/gguf/. Use /api/local-inference for the Docker command.",
        )
    yield


app = FastAPI(title="TMC-LM API", description="Offline TMC knowledge assistant API", lifespan=lifespan)

WEB_UI_PATH = Path(__file__).resolve().parent.parent.parent / "web_chat.html"


def find_model_path(model_path: Path | None = None) -> Path:
    found = find_model(model_path)
    if found is None:
        raise HTTPException(
            status_code=500,
            detail="No GGUF model found. Please run the training pipeline first or provide a valid --model path.",
        )
    return found


# Cached model + inference lock. llama-cpp is not thread-safe and reloading a
# multi-GB GGUF per request is far too slow, so the model is loaded once and
# inference is serialized across requests.
_MODEL: Any = None
_MODEL_PATH: Path | None = None
_MODEL_CTX_SIZE: int | None = None
_INFERENCE_LOCK = threading.Lock()


def get_loaded_model(model_path: Path, ctx_size: int) -> Any:
    """Return the cached Llama instance, loading the GGUF only on first use."""
    global _MODEL, _MODEL_PATH, _MODEL_CTX_SIZE
    resolved = model_path.resolve()
    if _MODEL is None or resolved != _MODEL_PATH or ctx_size != _MODEL_CTX_SIZE:
        from llama_cpp import Llama

        logger.info("Loading GGUF model: %s (n_ctx=%d)", model_path, ctx_size)
        _MODEL = Llama(model_path=str(model_path), n_ctx=ctx_size, verbose=False)
        _MODEL_PATH = resolved
        _MODEL_CTX_SIZE = ctx_size
    return _MODEL


def _build_messages(prompt: str, messages: list[ChatMessage] | None) -> list[dict[str, str]]:
    """Build the OpenAI-style message list, preferring full chat history."""
    if not messages:
        return [{"role": "user", "content": prompt}]
    return [{"role": message.role, "content": message.content} for message in messages]


def _generate_answer_stream(
    model_path: Path,
    messages: list[dict[str, str]],
    ctx_size: int,
    temp: float,
) -> Iterator[str]:
    """Yield answer tokens as the model generates them, serialized across requests.

    Shares the same ``_INFERENCE_LOCK`` and cached-model rules as
    ``_generate_answer`` so streaming never overlaps with non-streaming calls.
    """
    with _INFERENCE_LOCK:
        llm = get_loaded_model(model_path, ctx_size)
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=temp,
            stream=True,
        )
        for chunk in cast(Any, output):
            delta = cast(Any, chunk)["choices"][0].get("delta") or {}
            content = delta.get("content") or ""
            if content:
                yield content


def _llama_cpp_available() -> bool:
    """Return True when the optional llama-cpp-python runtime is importable."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True


def _sse_event(payload: dict[str, Any]) -> str:
    """Encode a payload as a single Server-Sent-Events ``data`` event."""
    return f"data: {json.dumps(payload)}\n\n"


def _sse_events(tokens: Iterator[str]) -> Iterator[str]:
    """Wrap a token stream into SSE events, always ending with a done/error marker."""
    try:
        for token in tokens:
            yield _sse_event({"type": "token", "content": token})
    except ImportError:
        yield _sse_event(
            {
                "type": "error",
                "detail": (
                    "llama-cpp-python is not installed on the server. "
                    "See /api/local-inference for the Docker command instead."
                ),
            }
        )
    except Exception:
        logger.exception("Inference failed")
        yield _sse_event({"type": "error", "detail": "Inference failed. Check the server logs."})
    else:
        yield _sse_event({"type": "done"})


def _generate_answer(model_path: Path, messages: list[dict[str, str]], ctx_size: int, temp: float) -> str:
    """Run one synchronous chat completion, serialized across requests."""
    with _INFERENCE_LOCK:
        llm = get_loaded_model(model_path, ctx_size)
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=temp,
        )
    response = cast(Any, output)
    content = response["choices"][0]["message"]["content"]
    return (content or "").strip()


class ChatMessage(BaseModel):
    """One message in a multi-turn conversation history."""

    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(default="", max_length=2000)


class Query(BaseModel):
    prompt: str = Field(default="What is TMC's vision?", max_length=2000)
    # Optional multi-turn history (lowercase roles, OpenAI-style). When present
    # it replaces the single ``prompt`` as the chat payload; the last message
    # must come from the user.
    messages: list[ChatMessage] | None = Field(default=None, max_length=20)
    # ctx_size allocates KV-cache memory and temp only accepts [0, 2]; reject
    # out-of-range values (422) instead of clamping silently.
    ctx_size: int = Field(default=2048, ge=256, le=4096)
    temp: float = Field(default=0.2, ge=0.0, le=2.0)

    @field_validator("messages")
    @classmethod
    def messages_must_end_with_user(cls, value: list[ChatMessage] | None) -> list[ChatMessage] | None:
        """Reject histories that do not end with the user's latest prompt."""
        if value and value[-1].role != "user":
            raise ValueError("the last message must have role 'user'")
        return value


class Answer(BaseModel):
    answer: str


@app.get("/")
async def root() -> dict[str, str | bool]:
    """Health check endpoint (never 500: reports model availability)."""
    model_path = find_model_path_if_exists()
    if model_path is None:
        return {
            "status": "ok",
            "message": "TMC-LM API is running (no GGUF model found)",
            "model": "",
            "found": False,
        }
    return {
        "status": "ok",
        "message": "TMC-LM API is running",
        "model": str(model_path),
        "found": True,
    }


@app.post("/query", response_model=Answer)
async def query(request: Query) -> Answer | JSONResponse:
    """Run inference with the GGUF model and return the answer."""
    try:
        model_path = find_model_path()
    except HTTPException as e:
        return JSONResponse(status_code=500, content={"detail": e.detail})

    prompt = request.prompt if request.prompt.strip() else "What is TMC's vision?"
    messages = _build_messages(prompt, request.messages)

    # Blocking CPU inference must not run on the event loop thread.
    loop = asyncio.get_running_loop()
    try:
        answer = await loop.run_in_executor(
            None, _generate_answer, model_path, messages, request.ctx_size, request.temp
        )
    except ImportError:
        return JSONResponse(
            status_code=500,
            content={
                "detail": "llama-cpp-python is not installed on the server. "
                "See /api/local-inference for the Docker command instead."
            },
        )
    except Exception:
        logger.exception("Inference failed")
        return JSONResponse(status_code=500, content={"detail": "Inference failed. Check the server logs."})
    return Answer(answer=answer)


@app.post("/query/stream", response_model=None)
async def query_stream(request: Query) -> StreamingResponse | JSONResponse:
    """Stream the answer as SSE token events for a GPT-style typewriter UI.

    Events are ``data:`` JSON lines of ``{"type": "token", "content": ...}``
    followed by a final ``{"type": "done"}`` (or ``{"type": "error"}``).
    """
    try:
        model_path = find_model_path()
    except HTTPException as e:
        return JSONResponse(status_code=500, content={"detail": e.detail})

    prompt = request.prompt if request.prompt.strip() else "What is TMC's vision?"
    messages = _build_messages(prompt, request.messages)

    if not _llama_cpp_available():
        return JSONResponse(
            status_code=500,
            content={
                "detail": "llama-cpp-python is not installed on the server. "
                "See /api/local-inference for the Docker command instead."
            },
        )

    tokens = _generate_answer_stream(model_path, messages, request.ctx_size, request.temp)
    return StreamingResponse(_sse_events(tokens), media_type="text/event-stream")


@app.get("/chat")
async def chat() -> FileResponse:
    """Serve the web-based chat interface."""
    return FileResponse(WEB_UI_PATH)


@app.get("/favicon.ico")
async def favicon() -> Response:
    """Suppress favicon 404 noise in the logs."""
    return Response(status_code=204)


@app.get("/api/model-status")
async def model_status() -> dict[str, bool]:
    """Report whether a GGUF model is available for inference."""
    return {"found": find_model_path_if_exists() is not None}


@app.get("/api/local-inference")
async def local_inference_instructions() -> PlainTextResponse:
    """Print the docker command needed to run local inference."""
    model_path = find_model_path_if_exists()
    if model_path is None:
        return PlainTextResponse(
            "No GGUF model found. Please run the training pipeline first. See the README for the full Docker workflow.",
            media_type="text/plain",
        )
    # The llama.cpp "light" image exposes llama-cli as its entrypoint (there is
    # no /app/llama.cpp inside the image) and conversation mode (-cnv) applies
    # the GGUF's embedded TinyLlama chat template, which matches how the model
    # was trained (see configs/train_lora_qa.yaml).
    # Mount the model file's own directory so the command works for any model
    # path, not just files under ./models/gguf.
    models_dir = model_path.resolve().parent
    cmd = (
        "docker run --rm -it "
        f'-v "{models_dir}:/models" '
        "ghcr.io/ggml-org/llama.cpp:light "
        f"-m /models/{model_path.name} "
        "-c 2048 --temp 0.2 --repeat-penalty 1.12 -cnv"
    )
    return PlainTextResponse(cmd, media_type="text/plain")


def find_model_path_if_exists() -> Path | None:
    try:
        return find_model_path()
    except HTTPException:
        return None


if __name__ == "__main__":
    import uvicorn

    model_path = find_model_path_if_exists()
    if model_path is None:
        print(
            "WARNING: No GGUF model found in models/gguf/. The UI will still load, "
            "but chat won't work until a model is available (see /api/local-inference).",
        )
    else:
        print(f"TMC-LM API starting with model: {model_path}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
