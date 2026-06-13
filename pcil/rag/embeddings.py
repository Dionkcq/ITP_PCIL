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
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise RuntimeError(
                    "FlagEmbedding is not installed; install FlagEmbedding "
                    "to use BGE-M3 retrieval"
                ) from exc

            use_fp16 = os.environ.get("RAG_BGE_USE_FP16", "auto").lower()
            if use_fp16 == "auto":
                use_fp16_bool = _cuda_available()
            else:
                use_fp16_bool = use_fp16 in {"1", "true", "yes"}
            cls._model = BGEM3FlagModel(model_name, use_fp16=use_fp16_bool)
            cls._model_name = model_name
            return cls._model

    @classmethod
    def encode(cls, texts: str | Iterable[str]) -> list[float] | list[list[float]]:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return [] if not single else [0.0] * EMBEDDING_DIM
        output = cls.get().encode(
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
