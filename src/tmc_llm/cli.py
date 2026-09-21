from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

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

    Uses llama-cli conversation mode so the GGUF's embedded chat template
    (TinyLlama's "<|user|>/<|assistant|>" format) is applied. Passing a raw
    --completion prompt would skip the template the model was trained with.
    """
    cmd = (
        "docker run --rm -it -v ${PWD}:/workspace "
        f"ghcr.io/ggml-org/llama.cpp:full "
        f"/app/llama.cpp/build/bin/llama-cli -m /app/{model_path.as_posix()} "
        f"-c {ctx_size} --temp {temp} --repeat-penalty 1.12 "
        f"-cnv -p \"{prompt}\""
    )
    print("Run this command (Docker must be running):")
    print(cmd)


def run_local_inference(model_path: Path, prompt: str, ctx_size: int = 2048, temp: float = 0.2) -> str:
    """Run inference with llama-cpp Python bindings, falling back to docker if unavailable."""
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=str(model_path), n_ctx=ctx_size, verbose=False)

        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=temp,
        )
        return output["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"llama-cpp not available ({e}), falling back to docker command.", file=sys.stderr)
        run_docker_inference(model_path, prompt, ctx_size, temp)
        return ""


def main() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
