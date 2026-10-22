"""Hybrid retrieval: exact/keyword matches + vector similarity."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag_qa.config import Settings

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "what",
        "who",
        "whom",
        "which",
        "where",
        "when",
        "why",
        "how",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "about",
        "define",
        "definition",
        "meaning",
        "mean",
        "of",
        "in",
        "on",
        "to",
        "for",
        "and",
        "or",
        "please",
        "tell",
        "me",
        "explain",
    }
)


def _all_store_documents(store: Any) -> list[Document]:
    """Return every document currently held in a FAISS store."""
    docs: list[Document] = []
    docstore = getattr(store, "docstore", None)
    if docstore is None:
        return docs
    data = getattr(docstore, "_dict", None)
    if isinstance(data, dict):
        return list(data.values())
    return docs


def extract_query_terms(question: str) -> list[str]:
    """Pull meaningful terms from a question (keeps Capitalized / long words)."""
    # Prefer the target of "what is X" / "define X" patterns.
    m = re.search(
        r"(?i)\b(?:what\s+is|what's|whats|define|definition\s+of|meaning\s+of)\s+(.+?)\s*$",
        question.strip(),
    )
    if m:
        tail = m.group(1).strip(" ?.!\"'")
        if tail:
            return [tail]

    words = re.findall(r"[A-Za-z][A-Za-z\-']{1,}", question)
    terms: list[str] = []
    for w in words:
        if w.lower() in _STOPWORDS:
            continue
        if len(w) < 3 and not w[:1].isupper():
            continue
        terms.append(w)
    # Dedupe case-insensitively, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def keyword_match_documents(
    documents: list[Document],
    terms: list[str],
) -> list[Document]:
    """Return docs that contain any query term (case-insensitive whole-word when possible)."""
    if not terms or not documents:
        return []

    matched: list[Document] = []
    seen_ids: set[int] = set()
    for term in terms:
        # Whole-word first, then substring fallback for hyphenated/partial OCR.
        whole = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        partial = re.compile(re.escape(term), re.IGNORECASE)
        for doc in documents:
            text = doc.page_content or ""
            if not (whole.search(text) or partial.search(text)):
                continue
            doc_id = id(doc)
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            matched.append(doc)
    return matched


def retrieve_documents(
    store: Any,
    question: str,
    settings: Settings,
) -> list[Document]:
    """Hybrid retrieve: keyword hits first, then BM25 + vector fill."""
    k = max(settings.top_k, 4)
    all_docs = _all_store_documents(store)
    terms = extract_query_terms(question)
    logger.info("Query terms: %s", terms)

    keyword_hits = keyword_match_documents(all_docs, terms)

    vector_hits = store.similarity_search(question, k=k)

    bm25_hits: list[Document] = []
    if all_docs:
        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = k
        bm25_hits = bm25.invoke(question)

    # Merge: keyword first (exact definitions), then BM25, then vector.
    merged: list[Document] = []
    seen: set[str] = set()

    def _add(docs: list[Document]) -> None:
        for doc in docs:
            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", "")
            key = f"{source}|{page}|{hash(doc.page_content)}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)

    _add(keyword_hits)
    _add(bm25_hits)
    _add(vector_hits)
    return merged[:k]
