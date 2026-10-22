"""Download-free embeddings for Streamlit Cloud (no HuggingFace / FastEmbed)."""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

_TOKEN = re.compile(r"[a-z0-9]+", re.I)


class HashEmbeddings(Embeddings):
    """Signed hashing-trick embeddings. Good enough for demo RAG, zero network."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8", errors="replace")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t or "") for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text or "")
