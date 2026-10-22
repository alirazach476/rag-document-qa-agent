"""Tests for settings loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_qa.config import BASE_DIR, Settings, get_settings


def test_require_api_key_rejects_placeholder_for_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-your-key-here")
    settings = get_settings()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.require_provider_ready()


def test_ollama_with_local_embeddings_skips_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    settings = get_settings()
    settings.require_provider_ready()
    assert settings.provider == "ollama"
    assert settings.resolved_embedding_backend == "local"
    assert "MiniLM" in settings.embedding_model
    assert settings.vector_dir == BASE_DIR / "vectorstore" / "local"


def test_require_api_key_accepts_real_looking_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-real-looking-key")
    settings = get_settings()
    settings.require_api_key()


def test_get_settings_reads_numeric_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CHUNK_SIZE", "512")
    monkeypatch.setenv("CHUNK_OVERLAP", "64")
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("VECTOR_DIR", str(tmp_path / "vs"))

    settings = get_settings()
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.top_k == 3
    assert settings.docs_dir == tmp_path / "docs"
    assert settings.index_path == tmp_path / "vs" / "index.faiss"


def test_settings_frozen() -> None:
    settings = Settings(openai_api_key="sk-test")
    with pytest.raises(AttributeError):
        settings.top_k = 99  # type: ignore[misc]


def test_unknown_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    settings = get_settings()
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        settings.require_provider_ready()
