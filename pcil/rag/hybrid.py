"""Hybrid BM25 + pgvector RAG retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import os
import re
import threading
from typing import Any

from pcil.rag.db import connect, fetch_all_records, vector_literal
from pcil.rag.embeddings import EmbeddingModelCache


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    rrf_k: int = 60,
    limit: int = 3,
) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, record_id in enumerate(ranking, start=1):
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (rrf_k + rank)
    return [
        record_id
        for record_id, _score in sorted(
            scores.items(), key=lambda kv: (-kv[1], kv[0])
        )[:limit]
    ]


class BM25IndexCache:
    _lock = threading.Lock()
    _records: dict[int, dict[str, Any]] = {}
    _record_ids: list[int] = []
    _index = None

    @classmethod
    def rebuild(cls) -> int:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "rank-bm25 is not installed; install rank-bm25 to use "
                "hybrid RAG retrieval"
            ) from exc

        rows = fetch_all_records()
        tokenized = [tokenize(row["retrieval_text"]) for row in rows]
        with cls._lock:
            cls._records = {int(row["id"]): dict(row) for row in rows}
            cls._record_ids = [int(row["id"]) for row in rows]
            cls._index = BM25Okapi(tokenized) if rows else None
        return len(rows)

    @classmethod
    def search(cls, query: str, *, limit: int) -> list[int]:
        with cls._lock:
            index = cls._index
            record_ids = list(cls._record_ids)
        if index is None or not record_ids:
            return []
        scores = index.get_scores(tokenize(query))
        ranked = sorted(
            zip(record_ids, scores),
            key=lambda kv: (-float(kv[1]), kv[0]),
        )
        return [record_id for record_id, score in ranked[:limit] if score > 0]

    @classmethod
    def hydrate(cls, record_ids: list[int]) -> list[dict[str, Any]]:
        with cls._lock:
            records = dict(cls._records)
        return [records[i] for i in record_ids if i in records]


def vector_search(query: str, *, limit: int) -> list[int]:
    embedding = EmbeddingModelCache.encode(query)
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id
            FROM rag_recovery_records
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_literal(embedding), limit),
        ).fetchall()
    return [int(row["id"]) for row in rows]


def hybrid_lookup(
    query: str,
    *,
    top_k: int = 3,
    candidate_k: int | None = None,
    rrf_k: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not query:
        return [], {
            "bm25_record_ids": [],
            "vector_record_ids": [],
            "fused_record_ids": [],
        }

    candidate_k = candidate_k or int(os.environ.get("RAG_CANDIDATE_K", "20"))
    rrf_k = rrf_k or int(os.environ.get("RAG_RRF_K", "60"))
    bm25_ids = BM25IndexCache.search(query, limit=candidate_k)
    vector_ids = vector_search(query, limit=candidate_k)
    fused_ids = reciprocal_rank_fusion(
        [bm25_ids, vector_ids],
        rrf_k=rrf_k,
        limit=top_k,
    )
    records = [
        {
            "id": row["id"],
            "error": row["error"],
            "cause": row["cause"],
            "recovery": row["recovery"],
            "source_doc": row["source_doc"],
        }
        for row in BM25IndexCache.hydrate(fused_ids)
    ]
    meta = {
        "bm25_record_ids": bm25_ids,
        "vector_record_ids": vector_ids,
        "fused_record_ids": fused_ids,
        "rrf_k": rrf_k,
        "top_k": top_k,
        "result_count": len(records),
    }
    return records, meta


def insert_search_event(query: str, meta: dict[str, Any]) -> int | None:
    with connect() as conn:
        row = conn.execute(
            """
            INSERT INTO rag_search_events (
                query, retrieval_mode, bm25_record_ids, vector_record_ids,
                fused_record_ids, rrf_k, top_k, result_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                query,
                "bm25_vector_rrf",
                meta.get("bm25_record_ids", []),
                meta.get("vector_record_ids", []),
                meta.get("fused_record_ids", []),
                meta.get("rrf_k", 60),
                meta.get("top_k", 3),
                meta.get("result_count", 0),
            ),
        ).fetchone()
        conn.commit()
    return int(row["id"]) if row else None


def insert_recommendation_event(
    *,
    search_event_id: int | None,
    impacts: dict[str, Any],
    signal_summary: dict[str, Any],
    baseline_comparison: dict[str, Any],
    recommendation_text: str,
    recommendation_source: str,
    recommendation_warnings: list[str],
) -> int | None:
    try:
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise RuntimeError("psycopg is not installed") from exc

    with connect() as conn:
        row = conn.execute(
            """
            INSERT INTO rag_recommendations (
                search_event_id, impacts, signal_summary, baseline_comparison,
                recommendation_text, recommendation_source,
                recommendation_warnings
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                search_event_id,
                Jsonb(impacts),
                Jsonb(signal_summary),
                Jsonb(baseline_comparison),
                recommendation_text,
                recommendation_source,
                Jsonb(recommendation_warnings),
            ),
        ).fetchone()
        conn.commit()
    return int(row["id"]) if row else None
