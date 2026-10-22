"""Tests for document loading (no OpenAI calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_qa.chain import format_docs
from rag_qa.config import BASE_DIR, Settings
from rag_qa.ingest import list_documents, load_documents, safe_save_upload


def test_load_sample_company_knowledge() -> None:
    docs = load_documents(BASE_DIR / "docs")
    assert len(docs) >= 1
    joined = "\n".join(d.page_content for d in docs)
    assert "Aether Analytics" in joined
    assert "SOC 2" in joined


def test_list_documents_filters_unsupported(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("# Hello", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    paths = list_documents(tmp_path)
    assert [p.name for p in paths] == ["keep.md"]


def test_safe_save_upload_rejects_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        safe_save_upload("notes.exe", b"bad", docs_path=tmp_path)


def test_safe_save_upload_uses_basename_only(tmp_path: Path) -> None:
    target = safe_save_upload("../nested/notes.md", b"# ok", docs_path=tmp_path)
    assert target.resolve() == (tmp_path / "notes.md").resolve()
    assert target.read_text(encoding="utf-8") == "# ok"


def test_safe_save_upload_writes_basename(tmp_path: Path) -> None:
    target = safe_save_upload("notes.txt", b"hello", docs_path=tmp_path)
    assert target == tmp_path / "notes.txt"
    assert target.read_text(encoding="utf-8") == "hello"


def test_format_docs_includes_source_labels() -> None:
    docs = [
        Document(page_content="alpha", metadata={"source": "a.md"}),
        Document(page_content="beta", metadata={"source": "b.md"}),
    ]
    text = format_docs(docs)
    assert "[Source 1: a.md]" in text
    assert "[Source 2: b.md]" in text
    assert "alpha" in text and "beta" in text


def test_load_documents_empty_dir(tmp_path: Path) -> None:
    settings = Settings(openai_api_key="sk-test", docs_dir=tmp_path)
    assert load_documents(settings=settings) == []
