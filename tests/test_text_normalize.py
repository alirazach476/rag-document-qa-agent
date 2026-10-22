from rag_qa.text_normalize import normalize_text


def test_normalize_ellipsis_and_dashes() -> None:
    raw = "Objectives:\u2026 scope \u2014 phase \u2013 one"
    assert normalize_text(raw) == "Objectives:... scope - phase - one"


def test_normalize_is_ascii() -> None:
    out = normalize_text("café • “quote”")
    out.encode("ascii")  # must not raise
