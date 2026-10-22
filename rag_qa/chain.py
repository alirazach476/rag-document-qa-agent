"""RAG retrieval + generation."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag_qa.config import Settings, get_settings
from rag_qa.providers import get_chat_model
from rag_qa.retrieve import retrieve_documents
from rag_qa.store import load_or_build_vectorstore
from rag_qa.text_normalize import normalize_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise RAG assistant. Answer ONLY using the provided context. "
    "If the user asks what a word means and that word (or a close form) appears in the "
    "context-especially in a vocabulary/dictionary style entry-explain it from the context. "
    "If the answer is truly not in the context, say you do not know. "
    "Cite sources like [Source N] when relevant.\n\nContext:\n{context}"
)


def format_docs(docs: list[Document]) -> str:
    """Format retrieved chunks for the LLM prompt."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = normalize_text(str(doc.metadata.get("source", "unknown")))
        parts.append(f"[Source {i}: {source}]\n{normalize_text(doc.page_content or '')}")
    return "\n\n".join(parts)


def create_rag_chain(
    vectorstore: Any | None = None,
    settings: Settings | None = None,
) -> tuple[Any, Any]:
    """Build a LangChain RAG runnable and its retriever."""
    cfg = settings or get_settings()
    store = vectorstore or load_or_build_vectorstore(settings=cfg)
    retriever = store.as_retriever(search_kwargs={"k": cfg.top_k})
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_chat_model(cfg)
        | StrOutputParser()
    )
    return chain, retriever


def ask(
    question: str,
    *,
    settings: Settings | None = None,
) -> tuple[str, list[Document]]:
    """Answer a question using hybrid retrieval (keyword + BM25 + vector)."""
    cfg = settings or get_settings()
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    from rag_qa.encoding_fix import apply as apply_encoding_fix

    apply_encoding_fix()
    store = load_or_build_vectorstore(settings=cfg)
    question = normalize_text(question)
    sources = retrieve_documents(store, question, cfg)
    for doc in sources:
        doc.page_content = normalize_text(doc.page_content or "")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    answer = (prompt | get_chat_model(cfg) | StrOutputParser()).invoke(
        {"context": format_docs(sources), "question": question}
    )
    answer = normalize_text(answer or "")
    logger.info("Answered question with %d source chunk(s)", len(sources))
    return answer, sources
