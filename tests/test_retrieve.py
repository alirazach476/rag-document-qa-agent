"""Tests for hybrid retrieval helpers."""

from __future__ import annotations

from langchain_core.documents import Document

from rag_qa.retrieve import extract_query_terms, keyword_match_documents


def test_extract_what_is_term() -> None:
    assert extract_query_terms("what is Prudent") == ["Prudent"]
    assert extract_query_terms("Define resilience") == ["resilience"]


def test_keyword_match_finds_vocab_entry() -> None:
    docs = [
        Document(page_content="Apple is a fruit.", metadata={"page": 1}),
        Document(
            page_content="Prudent: careful and sensible; acting with care.",
            metadata={"page": 2},
        ),
        Document(page_content="Banana is yellow.", metadata={"page": 3}),
    ]
    hits = keyword_match_documents(docs, ["Prudent"])
    assert len(hits) == 1
    assert "Prudent" in hits[0].page_content
