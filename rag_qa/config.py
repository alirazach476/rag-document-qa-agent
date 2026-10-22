"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

PROVIDERS = frozenset({"ollama", "openai", "groq"})
EMBEDDING_BACKENDS = frozenset(
    {"auto", "local", "ollama", "openai", "fastembed", "hash"}
)


def _env(name: str, default: str = "") -> str:
    """Read from process env, then Streamlit secrets (for cloud deploy)."""
    value = os.getenv(name)
    if value is not None and value != "":
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:  # noqa: BLE001
        pass
    return default


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the RAG pipeline."""

    provider: str = "ollama"
    embedding_backend: str = "auto"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "auto"
    embed_batch_size: int = 64
    chunk_size: int = 1200
    chunk_overlap: int = 100
    top_k: int = 4
    temperature: float = 0.1
    docs_dir: Path = BASE_DIR / "docs"
    vector_dir: Path = BASE_DIR / "vectorstore" / "local"

    @property
    def index_path(self) -> Path:
        return self.vector_dir / "index.faiss"

    @property
    def chat_model_name(self) -> str:
        if self.provider == "ollama":
            return self.ollama_model
        if self.provider == "groq":
            return self.groq_model
        return self.openai_model

    @property
    def resolved_embedding_backend(self) -> str:
        # Groq/cloud: never use FastEmbed/HF (ASCII locale crashes on "Downloading…").
        if self.provider == "groq" and self.embedding_backend in {
            "auto",
            "fastembed",
            "hash",
        }:
            return "hash"
        if self.embedding_backend != "auto":
            return self.embedding_backend
        if self.provider == "openai":
            return "openai"
        return "local"

    def require_provider_ready(self) -> None:
        """Fail fast when the selected provider is not configured."""
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER={self.provider!r}. "
                "Use 'ollama', 'groq' (free), or 'openai'."
            )
        if self.embedding_backend not in EMBEDDING_BACKENDS:
            raise ValueError(
                f"Unknown EMBEDDING_BACKEND={self.embedding_backend!r}. "
                "Use auto, local, fastembed, ollama, or openai."
            )
        if self.provider == "openai" or self.resolved_embedding_backend == "openai":
            if not self.openai_api_key or self.openai_api_key.startswith("sk-your-key"):
                raise ValueError(
                    "OPENAI_API_KEY is missing. For a free cloud deploy use "
                    "LLM_PROVIDER=groq with a free key from https://console.groq.com/keys"
                )
        if self.provider == "groq":
            if not self.groq_api_key or self.groq_api_key.startswith("gsk_your"):
                raise ValueError(
                    "GROQ_API_KEY is missing. Get a free key at "
                    "https://console.groq.com/keys (no OpenAI credits needed)."
                )

    def require_api_key(self) -> None:
        self.require_provider_ready()


def _default_embedding_model(backend: str) -> str:
    override = _env("EMBEDDING_MODEL")
    if override:
        return override.strip()
    if backend == "openai":
        return "text-embedding-3-small"
    if backend == "ollama":
        return "nomic-embed-text"
    if backend == "fastembed":
        return "BAAI/bge-small-en-v1.5"
    if backend == "hash":
        return "hashing-trick-384"
    return "sentence-transformers/all-MiniLM-L6-v2"


def _default_vector_dir(backend: str) -> Path:
    override = _env("VECTOR_DIR")
    if override:
        return Path(override)
    return BASE_DIR / "vectorstore" / backend


def get_settings() -> Settings:
    """Build settings from environment variables / Streamlit secrets."""
    provider = _env("LLM_PROVIDER", "ollama").strip().lower() or "ollama"
    embedding_backend = _env("EMBEDDING_BACKEND", "auto").strip().lower() or "auto"
    resolved = embedding_backend
    if resolved == "auto":
        if provider == "openai":
            resolved = "openai"
        elif provider == "groq":
            resolved = "hash"
        else:
            resolved = "local"

    return Settings(
        provider=provider,
        embedding_backend=embedding_backend,
        openai_api_key=_env("OPENAI_API_KEY").strip(),
        openai_model=_env("OPENAI_MODEL", "gpt-4o-mini").strip(),
        groq_api_key=_env("GROQ_API_KEY").strip(),
        groq_model=_env("GROQ_MODEL", "openai/gpt-oss-20b").strip(),
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip(),
        ollama_model=_env("OLLAMA_MODEL", "llama3.2").strip(),
        embedding_model=_default_embedding_model(resolved),
        embedding_device=_env("EMBEDDING_DEVICE", "auto").strip().lower() or "auto",
        embed_batch_size=int(_env("EMBED_BATCH_SIZE", "64")),
        chunk_size=int(_env("CHUNK_SIZE", "1200")),
        chunk_overlap=int(_env("CHUNK_OVERLAP", "100")),
        top_k=int(_env("TOP_K", "4")),
        temperature=float(_env("TEMPERATURE", "0.1")),
        docs_dir=Path(_env("DOCS_DIR", str(BASE_DIR / "docs"))),
        vector_dir=_default_vector_dir(resolved),
    )
