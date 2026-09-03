from pathlib import Path
from unittest.mock import Mock

from tmc_llm.merge_lora import merge_lora


class FakeTokenizer:
    def save_pretrained(self, output_dir: Path) -> None:
        pass


def test_merge_lora_saves_merged_model_and_tokenizer(tmp_path: Path, monkeypatch) -> None:
    base_model = "fake/base-model"
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "merged"
    adapter_dir.mkdir()

    merged_model = Mock()
    merged_model.save_pretrained = Mock()

    peft_model = Mock()
    peft_model.merge_and_unload = Mock(return_value=merged_model)

    base_model_instance = Mock()
    base_model_instance.config = Mock()

    tokenizer = FakeTokenizer()
    tokenizer.save_pretrained = Mock()

    monkeypatch.setattr(
        "tmc_llm.merge_lora.AutoModelForCausalLM.from_pretrained",
        Mock(return_value=base_model_instance),
    )
    monkeypatch.setattr(
        "tmc_llm.merge_lora.PeftModel.from_pretrained",
        Mock(return_value=peft_model),
    )
    monkeypatch.setattr(
        "tmc_llm.merge_lora.AutoTokenizer.from_pretrained",
        Mock(return_value=tokenizer),
    )

    merge_lora(base_model, adapter_dir, output_dir)

    merged_model.save_pretrained.assert_called_once_with(output_dir, safe_serialization=True)
    tokenizer.save_pretrained.assert_called_once_with(output_dir)
    assert output_dir.is_dir()


def test_merge_lora_requires_adapter_via_peft(monkeypatch, tmp_path: Path) -> None:
    base_model = "fake/base-model"
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "merged"
    adapter_dir.mkdir()

    merged_model = Mock()
    peft_model = Mock()
    peft_model.merge_and_unload = Mock(return_value=merged_model)

    peft_from_pretrained = Mock()
    monkeypatch.setattr("tmc_llm.merge_lora.PeftModel.from_pretrained", peft_from_pretrained)
    monkeypatch.setattr(
        "tmc_llm.merge_lora.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: Mock(),
    )
    monkeypatch.setattr(
        "tmc_llm.merge_lora.AutoTokenizer.from_pretrained",
        lambda *a, **k: FakeTokenizer(),
    )

    merge_lora(base_model, adapter_dir, output_dir)

    peft_from_pretrained.assert_called_once()
    assert peft_from_pretrained.call_args.args[1] == adapter_dir
