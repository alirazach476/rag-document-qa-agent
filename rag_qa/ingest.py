"""Document loading helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from rag_qa.config import Settings, get_settings

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
ProgressCallback = Callable[[str], None]


def list_documents(docs_path: Path | None = None) -> list[Path]:
    """Return supported document paths under docs/."""
    settings = get_settings()
    root = docs_path or settings.docs_dir
    root.mkdir(parents=True, exist_ok=True)
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _load_pdf(path: Path) -> list[Document]:
    """Load a PDF with PyMuPDF (fast). Falls back to pypdf if needed."""
    try:
        import pymupdf

        documents: list[Document] = []
        with pymupdf.open(path) as pdf:
            for i, page in enumerate(pdf):
                text = page.get_text("text").strip()
                if not text:
                    continue
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "page": i},
                    )
                )
        return documents
    except ImportError:
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(path)).load()


def load_documents(
    docs_path: Path | None = None,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> list[Document]:
    """Load .pdf / .txt / .md files into LangChain documents."""
    cfg = settings or get_settings()
    root = docs_path or cfg.docs_dir
    root.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    files = list_documents(root)
    total = len(files)

    for idx, path in enumerate(files, start=1):
        if progress:
            progress(f"Reading {path.name} ({idx}/{total})…")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(_load_pdf(path))
        elif suffix in {".txt", ".md"}:
            documents.extend(TextLoader(str(path), encoding="utf-8").load())

    logger.info("Loaded %d document page(s) from %s", len(documents), root)
    return documents


def safe_save_upload(filename: str, data: bytes, docs_path: Path | None = None) -> Path:
    """Save an uploaded file into docs/, rejecting path traversal."""
    settings = get_settings()
    root = (docs_path or settings.docs_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    name = Path(filename).name
    if not name or Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported or empty filename: {filename!r}")

    target = (root / name).resolve()
    if not str(target).startswith(str(root)):
        raise ValueError("Invalid upload path.")

    target.write_bytes(data)
    return target
