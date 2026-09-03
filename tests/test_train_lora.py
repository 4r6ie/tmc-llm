from pathlib import Path

import torch

from tmc_llm.train_lora import (
    PromptAwareDataCollator,
    format_messages,
    load_jsonl,
    pick_dtype,
    tokenize_dataset,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.chat_template = None
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.pad_token_id = 99

    def __call__(self, text: str, truncation: bool = False, max_length: int | None = None) -> dict:
        ids = [ord(char) for char in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = False
    ) -> str:
        return " ".join(f"{m['role']}:{m['content']}" for m in messages)


def make_conversation(assistant_content: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the vision of TMC?"},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def test_load_jsonl_reads_non_empty_lines(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"messages": [{"role": "user", "content": "a"}]}\n\n{"messages": [{"role": "user", "content": "b"}]}\n',
        encoding="utf-8",
    )

    rows = load_jsonl(path)

    assert len(rows) == 2
    assert rows[0]["messages"][0]["content"] == "a"


def test_load_jsonl_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text("\n\n\n", encoding="utf-8")

    assert load_jsonl(path) == []


def test_format_messages_without_chat_template_uses_roles() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    result = format_messages(tokenizer, messages)

    assert result == "USER: Hello\nASSISTANT: Hi there<eos>"


def test_format_messages_with_chat_template_delegates() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.chat_template = "custom"
    messages = [{"role": "user", "content": "Hello"}]

    result = format_messages(tokenizer, messages)

    assert result == "user:Hello"


def test_tokenize_dataset_marks_labels_on_assistant_tokens() -> None:
    tokenizer = FakeTokenizer()
    rows = [make_conversation("The college's vision is a model institution.")]

    dataset = tokenize_dataset(tokenizer, rows, max_length=512)

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["input_ids"]
    assert sample["attention_mask"] == [1] * len(sample["input_ids"])
    assert -100 in sample["labels"]
    assert sample["labels"][-1] == sample["input_ids"][-1]


def test_tokenize_dataset_skips_rows_without_assistant() -> None:
    tokenizer = FakeTokenizer()
    rows = [{"messages": [{"role": "user", "content": "only a user"}]}]

    dataset = tokenize_dataset(tokenizer, rows, max_length=512)

    assert len(dataset) == 0


def test_tokenize_dataset_skips_too_short_assistant_replies() -> None:
    class ChatJoinTokenizer(FakeTokenizer):
        def __init__(self) -> None:
            super().__init__()
            self.chat_template = "custom"

        def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = False) -> str:
            return "".join(m["content"] for m in messages)

    tokenizer = ChatJoinTokenizer()
    rows = [make_conversation("ok")]

    dataset = tokenize_dataset(tokenizer, rows, max_length=512)

    assert len(dataset) == 0


def test_tokenize_dataset_truncates_to_max_length() -> None:
    tokenizer = FakeTokenizer()
    rows = [make_conversation("This is a long response that should be truncated to the limit.")]
    max_length = 100

    dataset = tokenize_dataset(tokenizer, rows, max_length=max_length)

    assert len(dataset) == 1
    assert len(dataset[0]["input_ids"]) == max_length


def test_prompt_aware_collator_pads_batch_to_max_length() -> None:
    tokenizer = FakeTokenizer()
    collator = PromptAwareDataCollator(tokenizer=tokenizer)
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "labels": [-100, 2, 3, 4]},
    ]

    batch = collator(features)

    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)
    assert batch["input_ids"][0][3] == tokenizer.pad_token_id
    assert batch["attention_mask"][0][3] == 0
    assert batch["labels"][0][3] == -100
    assert batch["labels"][0][1] == 2


def test_prompt_aware_collator_pad_to_multiple_of() -> None:
    tokenizer = FakeTokenizer()
    collator = PromptAwareDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    features = [{"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1] * 5, "labels": [-100, 2, 3, 4, 5]}]

    batch = collator(features)

    assert batch["input_ids"].shape[1] == 8


def test_prompt_aware_collator_falls_back_to_pad_id_zero() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = None
    collator = PromptAwareDataCollator(tokenizer=tokenizer)
    features = [
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [2, 2]},
        {"input_ids": [1], "attention_mask": [1], "labels": [2]},
    ]

    batch = collator(features)

    assert batch["input_ids"][1][1] == 0


def test_pick_dtype_returns_fp32_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert pick_dtype({"bf16": True, "fp16": True}) == torch.float32


def test_pick_dtype_returns_bf16_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert pick_dtype({"bf16": True}) == torch.bfloat16
    assert pick_dtype({"bf16": False, "fp16": True}) == torch.float16
