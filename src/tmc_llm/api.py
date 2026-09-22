from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from tmc_llm.cli import find_model, run_local_inference


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


class Query(BaseModel):
    prompt: str = "What is TMC's vision?"
    ctx_size: int = 2048
    temp: float = 0.2


class Answer(BaseModel):
    answer: str


@app.get("/")
async def root() -> dict[str, str]:
    """Health check endpoint."""
    model_path = find_model_path()
    return {
        "status": "ok",
        "message": "TMC-LM API is running",
        "model": str(model_path),
    }


@app.post("/query", response_model=Answer)
async def query(request: Query) -> Answer | JSONResponse:
    """Run inference with the GGUF model and return the answer."""
    try:
        model_path = find_model_path()
    except HTTPException as e:
        return JSONResponse(status_code=500, content={"detail": e.detail})

    prompt = request.prompt if request.prompt.strip() else "What is TMC's vision?"

    try:
        answer = run_local_inference(model_path, prompt, request.ctx_size, request.temp)
        if not answer.strip():
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Local inference could not load the model (llama-cpp missing). "
                    "See /api/local-inference for the Docker command instead."
                },
            )
        return Answer(answer=answer)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


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
