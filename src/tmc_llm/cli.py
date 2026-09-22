from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any, cast

MODEL_CANDIDATES = [
    Path("./models/gguf/tmc-lm-tinyllama-q4_k_m.gguf"),
    Path("./models/gguf/tmc-lm-tinyllama-f16.gguf"),
    Path("./models/gguf/tmc-lm-tinyllama-f32.gguf"),
]


def find_model(model_path: Path | None = None) -> Path | None:
    if model_path:
        p = Path(model_path)
        if p.exists():
            return p
    for cand in MODEL_CANDIDATES:
        if cand.exists():
            return cand
    return None


def run_docker_inference(model_path: Path, prompt: str, ctx_size: int = 2048, temp: float = 0.2) -> None:
    """Print the docker command needed to run inference with the GGUF model.

    Uses the llama.cpp "light" image, where llama-cli is the container
    entrypoint, and conversation mode so the GGUF's embedded chat template
    is applied. Passing a raw --completion prompt would skip the template
    the model was trained with.
    """
    # Mount the model file's own directory so the command works for any model
    # path, not just files under ./models/gguf.
    models_dir = model_path.resolve().parent
    cmd = (
        "docker run --rm -it "
        f'-v "{models_dir}:/models" '
        "ghcr.io/ggml-org/llama.cpp:light "
        f"-m /models/{model_path.name} "
        f"-c {ctx_size} --temp {temp} --repeat-penalty 1.12 "
        f'-cnv -p "{prompt}"'
    )
    print("Run this command (Docker must be running):")
    print(cmd)


def run_local_inference(model_path: Path, prompt: str, ctx_size: int = 2048, temp: float = 0.2) -> str:
    """Run inference with llama-cpp Python bindings, falling back to docker if unavailable."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python is not installed. Install it with: pip install llama-cpp-python", file=sys.stderr)
        run_docker_inference(model_path, prompt, ctx_size, temp)
        return ""

    llm = Llama(model_path=str(model_path), n_ctx=ctx_size, verbose=False)
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=temp,
    )
    response = cast(Any, output)
    content = response["choices"][0]["message"]["content"]
    return content.strip() if content else ""


def main() -> None:
    with contextlib.suppress(Exception):
        cast(Any, sys.stdout).reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="TMC-LM offline inference with GGUF model")
    parser.add_argument("--model", type=Path, default=None, help="Path to GGUF model file")
    parser.add_argument("--prompt", type=str, default="", help="Prompt to send to the model")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context window size")
    parser.add_argument("--temp", type=float, default=0.2, help="Sampling temperature")
    args = parser.parse_args()

    model_path = find_model(args.model)
    if not model_path:
        print("No GGUF model found. Please run the training pipeline first or provide --model path.", file=sys.stderr)
        sys.exit(1)

    prompt = args.prompt if args.prompt.strip() else "What is TMC's vision?"

    result = run_local_inference(model_path, prompt, args.ctx_size, args.temp)
    if result:
        print(result)


if __name__ == "__main__":
    main()
