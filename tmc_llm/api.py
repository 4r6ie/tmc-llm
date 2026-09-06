from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tmc_llm.cli import find_model, run_local_inference

app = FastAPI(title="TMC-LM API", description="Offline TMC knowledge assistant API")

MODEL_CANDIDATES = [
    Path("./models/gguf/tmc-lm-tinyllama-q4_k_m.gguf"),
    Path("./models/gguf/tmc-lm-tinyllama-f16.gguf"),
    Path("./models/gguf/tmc-lm-tinyllama-f32.gguf"),
]


def find_model_path(model_path: Optional[Path] = None) -> Path:
    if model_path:
        p = Path(model_path)
        if p.exists():
            return p
    for cand in MODEL_CANDIDATES:
        if cand.exists():
            return cand
    raise HTTPException(
        status_code=500,
        detail="No GGUF model found. Please run the training pipeline first or provide a valid --model path.",
    )


class Query(BaseModel):
    prompt: str = "What is TMC's vision?"
    ctx_size: int = 2048
    temp: float = 0.2


class Answer(BaseModel):
    answer: str


@app.on_event("startup")
async def startup_check():
    """Verify model exists on startup; raise helpful error if missing."""
    try:
        find_model_path()
    except HTTPException:
        # Re-raise so FastAPI shows a clear startup error
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    model_path = find_model_path()
    return {
        "status": "ok",
        "message": "TMC-LM API is running",
        "model": str(model_path),
    }


@app.post("/query", response_model=Answer)
async def query(request: Query):
    """Run inference with the GGUF model and return the answer."""
    try:
        model_path = find_model_path()
    except HTTPException as e:
        return JSONResponse(status_code=500, content={"detail": e.detail})

    prompt = request.prompt if request.prompt.strip() else "What is TMC's vision?"

    try:
        answer = run_local_inference(model_path, prompt, request.ctx_size, request.temp)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


if __name__ == "__main__":
    import uvicorn

    model_path = find_model_path()
    print(f"TMC-LM API starting with model: {model_path}")
    uvicorn.run(app, host="0.0.0.0", port=8000)