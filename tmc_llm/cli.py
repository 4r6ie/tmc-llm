from __future__ import annotations

import argparse
import json
import os
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
    """Print the docker command needed to run inference with the GGUF model."""
    cmd = (
        "docker run --rm -it -v ${PWD}:/app "
        f"ghcr.io/ggml-org/llama.cpp:full "
        f"/app/llama.cpp/build/bin/llama-cli -m {model_path} "
        f"-c {ctx_size} --temp {temp} --repeat-penalty 1.12 -p \"{prompt}\""
    )
    print("Run this command (Docker must be running):")
    print(cmd)


def run_local_inference(model_path: Path, prompt: str, ctx_size: int = 2048, temp: float = 0.2) -> str:
    """Try to use llama-cpp Python bindings if available, otherwise fall back to docker command."""
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=str(model_path), n_ctx=ctx_size, verbose=False)
        system_prompt = "You are TMC-LM, an offline assistant for Trinidad Municipal College. "
        system_prompt += "Answer using only the provided official TMC knowledge. "
        system_prompt += "If the source does not contain the answer, say that the available TMC source does not contain it."
        full_prompt = f"<</SYS>>{system_prompt}User: {prompt}</SYS>>"

        output = llm(full_prompt, max_tokens=512, temperature=temp, stop=[ "<</SYS>>" ])
        return output["choices"][0]["text"].strip()
    except Exception as e:
        print(f"llama-cpp not available ({e}), falling back to docker command.", file=sys.stderr)
        run_docker_inference(model_path, prompt, ctx_size, temp)
        return ""


def main() -> None:
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