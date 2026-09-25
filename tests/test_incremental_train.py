from __future__ import annotations

from pathlib import Path

from tmc_llm.incremental_train import get_next_version


class TestGetNextVersion:
    def testIncrementsMinor(self, tmp_path: Path) -> None:
        assert get_next_version(tmp_path, "1.0") == "1.1"

    def testIncrementsFromHigherMinor(self, tmp_path: Path) -> None:
        assert get_next_version(tmp_path, "1.5") == "1.6"

    def testHandlesSinglePart(self, tmp_path: Path) -> None:
        result = get_next_version(tmp_path, "1")
        assert result == "1.1"

    def testPreservesSuffix(self, tmp_path: Path) -> None:
        assert get_next_version(tmp_path, "1.0-qa") == "1.1-qa"

    def testPreservesSuffixSinglePart(self, tmp_path: Path) -> None:
        assert get_next_version(tmp_path, "1-qa") == "1.1-qa"

    def testHandlesNonNumericGracefully(self, tmp_path: Path) -> None:
        assert get_next_version(tmp_path, "abc") == "abc.1"
