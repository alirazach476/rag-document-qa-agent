"""RAG Document Q&A Agent — grounded answers over your documents."""

from rag_qa.chain import ask, create_rag_chain, format_docs
from rag_qa.config import Settings, get_settings
from rag_qa.ingest import list_documents, load_documents
from rag_qa.store import build_vectorstore, load_or_build_vectorstore

__all__ = [
    "Settings",
    "ask",
    "build_vectorstore",
    "create_rag_chain",
    "format_docs",
    "get_settings",
    "list_documents",
    "load_documents",
    "load_or_build_vectorstore",
]

__version__ = "1.0.0"
