"""Backward-compatible entry point for the RAG CLI.

Prefer: ``python -m rag_qa.cli`` or the ``rag-qa`` console script after install.
"""

from rag_qa.chain import ask, create_rag_chain, format_docs
from rag_qa.cli import main
from rag_qa.config import BASE_DIR, get_settings
from rag_qa.ingest import load_documents
from rag_qa.store import build_vectorstore, load_or_build_vectorstore

_settings = get_settings()
DOCS_DIR = _settings.docs_dir
VECTOR_DIR = _settings.vector_dir

__all__ = [
    "BASE_DIR",
    "DOCS_DIR",
    "VECTOR_DIR",
    "ask",
    "build_vectorstore",
    "create_rag_chain",
    "format_docs",
    "load_documents",
    "load_or_build_vectorstore",
    "main",
]

if __name__ == "__main__":
    main()
