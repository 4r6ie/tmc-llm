from __future__ import annotations

import re

MOJIBAKE_REPLACEMENTS = {
    "â€œ": '"',
    "â€": '"',
    "â€˜": "'",
    "â€™": "'",
    "â€“": "-",
    "â€”": "-",
    "â†“": "->",
    "âœ”": "-",
    "âŒ": "x",
    "â”‚": "|",
    "â–¼": "v",
}


def clean_text(text: str) -> str:
    """Normalize common copied-document artifacts without changing meaning."""
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
