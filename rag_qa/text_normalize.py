"""Normalize text so ASCII-only locales (Streamlit Cloud) do not crash."""

from __future__ import annotations

# Common PDF / Word punctuation that triggers UnicodeEncodeError under ASCII.
_REPLACEMENTS = str.maketrans(
    {
        "\u2026": "...",  # …
        "\u2013": "-",  # –
        "\u2014": "-",  # —
        "\u2018": "'",  # ‘
        "\u2019": "'",  # ’
        "\u201c": '"',  # “
        "\u201d": '"',  # ”
        "\u00a0": " ",  # nbsp
        "\u2022": "*",  # •
        "\u00b7": "*",  # ·
        "\ufeff": "",  # BOM
    }
)


def normalize_text(text: str) -> str:
    """Replace fancy punctuation; drop any remaining non-ASCII safely."""
    if not text:
        return text
    cleaned = text.translate(_REPLACEMENTS)
    return cleaned.encode("ascii", errors="replace").decode("ascii")
