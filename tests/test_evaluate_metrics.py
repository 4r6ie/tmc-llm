from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmc_llm.evaluate import (
    compute_average_response_length,
    compute_exact_match,
    compute_keyword_overlap,
    compute_negative_rejection,
    extract_qa_pairs,
    load_jsonl,
)


class TestExtractQaPairs:
    def test_extracts_pairs(self) -> None:
        entries = [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "What is TMC?"},
                    {"role": "assistant", "content": "Trinidad Municipal College."},
                ]
            }
        ]
        pairs = extract_qa_pairs(entries)
        assert len(pairs) == 1
        assert pairs[0] == ("What is TMC?", "Trinidad Municipal College.")

    def test_skips_entries_without_both(self) -> None:
        entries = [{"messages": [{"role": "user", "content": "hi"}]}]
        pairs = extract_qa_pairs(entries)
        assert len(pairs) == 0


class TestComputeExactMatch:
    def test_perfect_match(self) -> None:
        assert compute_exact_match(["hello", "world"], ["hello", "world"]) == 1.0

    def test_no_match(self) -> None:
        assert compute_exact_match(["a", "b"], ["c", "d"]) == 0.0

    def test_case_insensitive(self) -> None:
        assert compute_exact_match(["Hello"], ["hello"]) == 1.0

    def test_empty(self) -> None:
        assert compute_exact_match([], []) == 0.0


class TestComputeKeywordOverlap:
    def test_high_overlap(self) -> None:
        score = compute_keyword_overlap(["the cat sat"], ["the cat sat on a mat"])
        assert score == pytest.approx(0.5)

    def test_no_overlap(self) -> None:
        score = compute_keyword_overlap(["apple"], ["banana"])
        assert score == 0.0


class TestComputeNegativeRejection:
    def test_rejects_negative(self) -> None:
        pred = ["The available TMC source does not contain this information."]
        ref = ["The available TMC source does not contain this information."]
        assert compute_negative_rejection(pred, ref) == 1.0

    def test_no_rejection(self) -> None:
        pred = ["TMC is great"]
        ref = ["The available TMC source does not contain this information."]
        assert compute_negative_rejection(pred, ref) == 0.0


class TestComputeAverageResponseLength:
    def test_average(self) -> None:
        result = compute_average_response_length(["hi", "hello world"])
        assert result == 6.5

    def test_empty(self) -> None:
        assert compute_average_response_length([]) == 0.0
