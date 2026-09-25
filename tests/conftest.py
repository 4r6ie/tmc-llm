"""Session-level fixtures.

``data/processed/*.jsonl`` is gitignored, so a fresh clone has an empty
``data/processed/`` directory. Some tests (e.g. ``test_evaluate.py``) read those
files directly, so we rebuild them from the tracked raw sources once per session
when they are missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tmc_llm.dataset_builder import build_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "tmc_sources"
PROCESSED_DIR = ROOT / "data" / "processed"


@pytest.fixture(scope="session", autouse=True)
def ensure_processed_dataset() -> None:
    """Rebuild ``data/processed/`` from the raw sources when it is missing."""
    if (PROCESSED_DIR / "train.jsonl").exists():
        return
    if not RAW_DIR.exists():
        return
    sources = [path for path in RAW_DIR.iterdir() if path.is_file()]
    if not sources:
        return
    build_dataset(None, PROCESSED_DIR, RAW_DIR)
