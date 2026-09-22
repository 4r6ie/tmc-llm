from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmc_llm.versioning import (
    delete_version,
    get_current_gguf,
    get_current_version,
    get_version_info,
    list_versions,
    promote_version,
    register_version,
    set_current_version,
)


class TestVersionRegistry:
    def test_register_and_list(self, tmp_path: Path) -> None:
        vp = tmp_path / "versions.json"
        register_version("1.0", tmp_path / "adapter", "Initial", versions_path=vp)
        data = list_versions(vp)
        assert "1.0" in data["versions"]
        assert data["current"] == "1.0"

    def test_set_current(self, tmp_path: Path) -> None:
        vp = tmp_path / "versions.json"
        register_version("1.0", tmp_path / "a", versions_path=vp)
        register_version("1.1", tmp_path / "b", versions_path=vp)
        set_current_version("1.0", vp)
        assert get_current_version(vp) == "1.0"

    def test_set_current_invalid(self, tmp_path: Path) -> None:
        vp = tmp_path / "versions.json"
        with pytest.raises(ValueError):
            set_current_version("9.9", vp)

    def test_get_version_info(self, tmp_path: Path) -> None:
        vp = tmp_path / "versions.json"
        register_version("1.0", tmp_path / "a", "desc", versions_path=vp)
        info = get_version_info("1.0", vp)
        assert info is not None
        assert info["description"] == "desc"

    def test_delete_version(self, tmp_path: Path) -> None:
        vp = tmp_path / "versions.json"
        register_version("1.0", tmp_path / "a", versions_path=vp)
        register_version("1.1", tmp_path / "b", versions_path=vp)
        set_current_version("1.1", vp)
        assert delete_version("1.0", vp) is True
        assert get_current_version(vp) == "1.1"

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        vp = tmp_path / "versions.json"
        assert delete_version("9.9", vp) is False

    def test_duplicate_register_raises(self, tmp_path: Path) -> None:
        vp = tmp_path / "versions.json"
        register_version("1.0", tmp_path / "a", versions_path=vp)
        with pytest.raises(ValueError):
            register_version("1.0", tmp_path / "b", versions_path=vp)


class TestGetCurrentGguf:
    @staticmethod
    def _register_with_gguf(vp: Path, tmp_path: Path, gguf_target: Path) -> None:
        register_version("1.0", tmp_path / "adapter", versions_path=vp)
        data = json.loads(vp.read_text(encoding="utf-8"))
        data["versions"]["1.0"]["gguf_dir"] = str(gguf_target)
        vp.write_text(json.dumps(data), encoding="utf-8")

    def test_returns_gguf_file_of_current_version(self, tmp_path: Path) -> None:
        vp = tmp_path / "versions.json"
        gguf = tmp_path / "model.gguf"
        gguf.write_text("gguf", encoding="utf-8")
        self._register_with_gguf(vp, tmp_path, gguf)

        assert get_current_gguf(vp) == gguf

    def test_prefers_q4_k_m_inside_directory(self, tmp_path: Path) -> None:
        vp = tmp_path / "versions.json"
        gguf_dir = tmp_path / "gguf"
        gguf_dir.mkdir()
        (gguf_dir / "tmc-lm-f16.gguf").write_text("x", encoding="utf-8")
        (gguf_dir / "tmc-lm-q4_k_m.gguf").write_text("x", encoding="utf-8")
        self._register_with_gguf(vp, tmp_path, gguf_dir)

        assert get_current_gguf(vp) == gguf_dir / "tmc-lm-q4_k_m.gguf"

    def test_returns_none_when_registered_gguf_missing(self, tmp_path: Path) -> None:
        vp = tmp_path / "versions.json"
        self._register_with_gguf(vp, tmp_path, tmp_path / "nope.gguf")

        assert get_current_gguf(vp) is None

    def test_returns_none_without_current_version(self, tmp_path: Path) -> None:
        vp = tmp_path / "versions.json"

        assert get_current_gguf(vp) is None

    def test_survives_corrupt_registry(self, tmp_path: Path) -> None:
        vp = tmp_path / "versions.json"
        vp.write_text("{not json", encoding="utf-8")

        assert get_current_gguf(vp) is None


class TestPromoteCli:
    def test_promote_registers_gguf_dir(self, tmp_path: Path) -> None:
        """promote_version with gguf_dir makes the version resolvable for serving."""
        vp = tmp_path / "versions.json"
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        register_version("1.0", adapter, versions_path=vp)
        gguf = tmp_path / "gguf" / "model.gguf"
        gguf.parent.mkdir()
        gguf.write_text("gguf", encoding="utf-8")

        promote_version("1.0", adapter, versions_path=vp, gguf_dir=gguf)

        assert get_current_gguf(vp) == gguf
