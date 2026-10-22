"""Replace fancy Unicode punctuation in runtime Python files (Streamlit Cloud ASCII locale)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ROOT / "app.py",
    ROOT / "rag_qa" / "store.py",
    ROOT / "rag_qa" / "ingest.py",
    ROOT / "rag_qa" / "chain.py",
    ROOT / "rag_qa" / "cli.py",
    ROOT / "rag_qa" / "providers.py",
    ROOT / "rag_qa" / "ui_theme.py",
    ROOT / "rag_qa" / "__init__.py",
]
REPLACEMENTS = {
    "\u2026": "...",
    "\u2014": "-",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}


def main() -> None:
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        updated = text
        for src, dst in REPLACEMENTS.items():
            updated = updated.replace(src, dst)
        if updated != text:
            path.write_text(updated, encoding="utf-8", newline="\n")
            print(f"updated {path.relative_to(ROOT)}")
        else:
            print(f"ok {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
