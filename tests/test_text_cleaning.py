from tmc_llm.text_cleaning import MOJIBAKE_REPLACEMENTS, clean_text, compact_spaces


def test_clean_text_replaces_mojibake_quotes() -> None:
    assert clean_text("â€œquotedâ€") == '"quoted"'


def test_clean_text_normalizes_newlines() -> None:
    assert clean_text("one\r\ntwo\rthree") == "one\ntwo\nthree"


def test_clean_text_collapses_spaces_and_keeps_single_newlines() -> None:
    assert clean_text("a   b\t\tc") == "a b c"
    assert clean_text("line1\n\n\nline2") == "line1\n\nline2"


def test_clean_text_strips_surrounding_whitespace() -> None:
    assert clean_text("  padded  ") == "padded"


def test_clean_text_allows_all_mojibake_keys() -> None:
    representative = "".join(MOJIBAKE_REPLACEMENTS)
    result = clean_text(representative)
    for original in MOJIBAKE_REPLACEMENTS:
        assert original not in result


def test_clean_text_empty_input() -> None:
    assert clean_text("") == ""


def test_compact_spaces_collapses_all_whitespace() -> None:
    assert compact_spaces("hello\n  world\t\t!\r\n") == "hello world !"
