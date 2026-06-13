"""BGE-M3 embedding cache for PostgreSQL RAG retrieval."""

from __future__ import annotations

import os
import threading
from typing import Iterable


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024


class EmbeddingModelCache:
    """Lazy singleton around BGE-M3 with an explicit warm-up hook."""

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
                # Try to use sentence_transformers for BGE-M3 to avoid FlagEmbedding compatibility issues
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer(model_name)
            except Exception as exc:
                # Re-raise with context: could be missing package OR a dependency conflict
                # (e.g., transformers version mismatch). The traceback will show which.
                raise RuntimeError(
                    f"Failed to import embedding model dependencies. "
                    f"Check that sentence-transformers (or FlagEmbedding) and transformers are installed and compatible. "
                    f"Error: {exc}"
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
        
        # Check if using SentenceTransformer (simpler API) or BGEM3FlagModel (advanced API)
        model_type = type(model).__name__
        if model_type == "SentenceTransformer":
            # SentenceTransformer.encode() returns numpy array directly
            output = model.encode(
                batch,
                batch_size=int(os.environ.get("RAG_EMBED_BATCH_SIZE", "8")),
            )
            # Convert numpy array to list of lists
            vectors = [list(map(float, v)) for v in output]
        else:
            # BGEM3FlagModel returns dict with "dense_vecs" key
            output = model.encode(
                batch,
                batch_size=int(os.environ.get("RAG_EMBED_BATCH_SIZE", "8")),
                max_length=int(os.environ.get("RAG_EMBED_MAX_LENGTH", "2048")),
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dense = output["dense_vecs"]
            vectors = [list(map(float, v)) for v in dense]
        for vector in vectors:
            if len(vector) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"BGE-M3 returned {len(vector)} dimensions; expected {EMBEDDING_DIM}"
                )
        return vectors[0] if single else vectors

    @classmethod
    def warm_up(cls) -> None:
        cls.encode("warm up retrieval model")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False
