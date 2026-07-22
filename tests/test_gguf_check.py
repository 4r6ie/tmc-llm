import struct
from pathlib import Path

import pytest

from tmc_llm.gguf_check import GgufCheckError, inspect_gguf


def test_inspect_gguf_accepts_valid_header(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"\x00" * 16)

    info = inspect_gguf(gguf)

    assert info["valid"] is True
    assert info["magic"] == "GGUF"
    assert info["version"] == 3


def test_inspect_gguf_rejects_invalid_header(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.gguf"
    bad_file.write_bytes(b"NOPE" + b"\x00" * 16)

    with pytest.raises(GgufCheckError):
        inspect_gguf(bad_file)

