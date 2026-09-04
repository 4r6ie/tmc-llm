from __future__ import annotations

import argparse
import torch
from pathlib import Path

from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with TinyLlama + LoRA adapter.")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter-dir", type=Path, default=Path("models/adapters/tmc-lm-tinyllama-lora-v1.0"))
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    adapter_dir = args.adapter_dir.resolve()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    print("Chat with AI! Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        messages = [
            {"role": "user", "content": user_input},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, truncation=True, max_length=args.max_length, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        print(f"AI: {response.strip()}\n")


if __name__ == "__main__":
    main()