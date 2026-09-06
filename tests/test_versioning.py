from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmc_llm.versioning import (
    delete_version,
    get_current_version,
    get_version_info,
    list_versions,
    register_version,
    save_versions,
    load_versions,
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
