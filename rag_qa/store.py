"""FAISS vector store build and load."""

from __future__ import annotations

import logging
from collections.abc import Callable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_qa.config import Settings, get_settings
from rag_qa.ingest import load_documents
from rag_qa.providers import get_embeddings
from rag_qa.text_normalize import normalize_text

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


def split_documents(
    documents: list[Document],
    settings: Settings | None = None,
) -> list[Document]:
    cfg = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(
    documents: list[Document] | None = None,
    *,
    persist: bool = True,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
    batch_size: int | None = None,
) -> FAISS:
    """Chunk documents, embed them in batches, and optionally persist FAISS."""
    cfg = settings or get_settings()
    batch = batch_size or cfg.embed_batch_size

    if documents is None:
        if progress:
            progress(0, 1, "Loading documents...")
        docs = load_documents(
            settings=cfg,
            progress=(lambda msg: progress(0, 1, msg)) if progress else None,
        )
    else:
        docs = documents

    if not docs:
        raise FileNotFoundError(
            f"No documents found in {cfg.docs_dir}. Add .txt, .md, or .pdf files."
        )

    if progress:
        progress(0, 1, f"Loaded {len(docs)} page(s); splitting...")

    chunks = split_documents(docs, settings=cfg)
    total = len(chunks)
    logger.info(
        "Split into %d chunk(s); chat=%s embed=%s",
        total,
        cfg.provider,
        cfg.resolved_embedding_backend,
    )

    if progress:
        progress(0, total, "Loading embedding model...")

    embeddings = get_embeddings(cfg)
    texts = [normalize_text(c.page_content or "") for c in chunks]
    metadatas = [c.metadata for c in chunks]

    if progress:
        progress(0, total, f"Embedding 0/{total} chunks...")

    vectors: list[list[float]] = []
    for start in range(0, total, batch):
        end = min(start + batch, total)
        vectors.extend(embeddings.embed_documents(texts[start:end]))
        if progress:
            progress(end, total, f"Embedding {end}/{total} chunks...")

    text_embeddings = list(zip(texts, vectors, strict=True))
    store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings,
        metadatas=metadatas,
    )

    if persist:
        if progress:
            progress(total, total, "Saving FAISS index...")
        cfg.vector_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(cfg.vector_dir))
        logger.info("Saved FAISS index to %s", cfg.vector_dir)

    if progress:
        progress(total, total, f"Done - indexed {total} chunks.")

    return store


def load_or_build_vectorstore(
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
) -> FAISS:
    """Load a persisted FAISS index, or build one if missing."""
    cfg = settings or get_settings()
    embeddings = get_embeddings(cfg)

    if cfg.index_path.exists():
        logger.info("Loading existing FAISS index from %s", cfg.vector_dir)
        return FAISS.load_local(
            str(cfg.vector_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    logger.info("No index found; building a new one")
    return build_vectorstore(settings=cfg, progress=progress)


def index_exists(settings: Settings | None = None) -> bool:
    cfg = settings or get_settings()
    return cfg.index_path.exists()


def get_index_mtime(settings: Settings | None = None) -> float | None:
    cfg = settings or get_settings()
    if not cfg.index_path.exists():
        return None
    return cfg.index_path.stat().st_mtime
