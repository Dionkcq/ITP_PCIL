"""Embedding cache for PostgreSQL RAG retrieval (default: all-MiniLM-L6-v2)."""

from __future__ import annotations

import os
import threading
from typing import Iterable


# Default to a small, CPU/offline-friendly model for the NUC. all-MiniLM-L6-v2
# is ~88MB and emits 384-dim dense vectors. RAG_EMBEDDING_MODEL can point at a
# different model (e.g. BAAI/bge-m3, 1024-dim, ~2.27GB), but the pgvector column
# width is fixed to EMBEDDING_DIM by migration 001 — switching to a model with a
# different dimension means updating both EMBEDDING_DIM and that migration.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingModelCache:
    """Lazy singleton around the embedding model with an explicit warm-up hook."""

    _lock = threading.Lock()
    _model = None
    _model_name: str | None = None

    @classmethod
    def model_name(cls) -> str:
        return os.environ.get("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    @classmethod
    def get(cls):
        model_name = cls.model_name()
        with cls._lock:
            if cls._model is not None and cls._model_name == model_name:
                return cls._model
            try:
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer(model_name)
            except Exception as exc:
                # Could be a missing package or a dependency conflict (e.g. a
                # transformers version mismatch). The traceback shows which.
                raise RuntimeError(
                    f"Failed to load the embedding model {model_name!r}. "
                    f"Check that sentence-transformers and transformers are "
                    f"installed and compatible. Error: {exc}"
                ) from exc

            cls._model_name = model_name
            return cls._model

    @classmethod
    def encode(cls, texts: str | Iterable[str]) -> list[float] | list[list[float]]:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return [] if not single else [0.0] * EMBEDDING_DIM
        model = cls.get()
        # get() always returns a sentence-transformers model; .encode() returns
        # a numpy array of dense vectors (one row per input text).
        output = model.encode(
            batch,
            batch_size=int(os.environ.get("RAG_EMBED_BATCH_SIZE", "8")),
        )
        vectors = [list(map(float, v)) for v in output]
        for vector in vectors:
            if len(vector) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"embedding model returned {len(vector)} dimensions; "
                    f"expected {EMBEDDING_DIM}"
                )
        return vectors[0] if single else vectors

    @classmethod
    def warm_up(cls) -> None:
        cls.encode("warm up retrieval model")
