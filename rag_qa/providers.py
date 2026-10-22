"""LLM and embedding providers (local GPU, Ollama, Groq free, or OpenAI)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import rag_qa.encoding_fix  # noqa: F401
from rag_qa.config import Settings

logger = logging.getLogger(__name__)


def check_ollama_reachable(base_url: str) -> None:
    """Raise a clear error if the Ollama server is not running."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urlopen(url, timeout=3) as resp:  # noqa: S310 - local Ollama only
            if resp.status >= 400:
                raise ConnectionError(f"Ollama returned HTTP {resp.status} at {url}")
    except URLError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}. "
            "Install from https://ollama.com, start the app, then run:\n"
            "  ollama pull llama3.2"
        ) from exc


@lru_cache(maxsize=4)
def resolve_torch_device(requested: str) -> str:
    """Pick cuda when available unless the user forced cpu."""
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu"
    if requested in {"cuda", "gpu", "auto"}:
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                logger.info("Using CUDA GPU for embeddings: %s", name)
                return "cuda"
        except Exception:  # noqa: BLE001
            logger.warning("CUDA check failed; falling back to CPU embeddings")
        if requested in {"cuda", "gpu"}:
            logger.warning("CUDA requested but unavailable; using CPU")
            return "cpu"
    return "cpu"


def display_device(requested: str) -> str:
    """Label for the UI without importing torch (uses env preference)."""
    requested = (requested or "auto").lower()
    if requested in {"cuda", "gpu"}:
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "auto"


@lru_cache(maxsize=4)
def _cached_local_embeddings(model_name: str, device: str) -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 128},
        show_progress=False,
    )


@lru_cache(maxsize=4)
def _cached_fastembed(model_name: str) -> Any:
    """Lightweight CPU embeddings for free cloud deploy (no torch GPU needed)."""
    from rag_qa.encoding_fix import apply as apply_encoding_fix

    apply_encoding_fix()
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=model_name)


def get_embeddings(settings: Settings) -> Any:
    """Return the embedding model for the configured backend."""
    settings.require_provider_ready()
    backend = settings.resolved_embedding_backend

    if backend == "local":
        device = resolve_torch_device(settings.embedding_device)
        return _cached_local_embeddings(settings.embedding_model, device)

    if backend == "fastembed":
        return _cached_fastembed(settings.embedding_model)

    if backend == "ollama":
        check_ollama_reachable(settings.ollama_base_url)
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)

    raise ValueError(f"Unsupported embedding backend: {backend}")


def get_chat_model(settings: Settings) -> Any:
    """Return the chat model for the configured provider."""
    settings.require_provider_ready()
    if settings.provider == "ollama":
        check_ollama_reachable(settings.ollama_base_url)
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
        )

    if settings.provider == "groq":
        from langchain_openai import ChatOpenAI

        # Groq is OpenAI-compatible and has a free API tier.
        return ChatOpenAI(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=settings.temperature,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.openai_model, temperature=settings.temperature)
