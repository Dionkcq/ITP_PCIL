"""Tests for the PostgreSQL hybrid RAG scaffolding.

These stay offline by mocking the BGE-M3 model, BM25 dependency, and DB
access. Real pgvector integration is a Docker-level verification step.
"""

from __future__ import annotations

import sys
import types


def test_rrf_is_deterministic():
    from pcil.rag.hybrid import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(
        [[10, 20, 30], [30, 40, 10]],
        rrf_k=60,
        limit=3,
    )

    assert fused == [10, 30, 20]


def test_bm25_cache_ranks_lexical_matches(monkeypatch):
    from pcil.rag.hybrid import BM25IndexCache
    import pcil.rag.hybrid as hybrid

    class FakeBM25:
        def __init__(self, docs):
            self.docs = docs

        def get_scores(self, query_tokens):
            query = set(query_tokens)
            return [len(query & set(doc)) for doc in self.docs]

    monkeypatch.setitem(
        sys.modules,
        "rank_bm25",
        types.SimpleNamespace(BM25Okapi=FakeBM25),
    )
    monkeypatch.setattr(
        hybrid,
        "fetch_all_records",
        lambda: [
            {
                "id": 1,
                "retrieval_text": "air pressure low",
                "error": "Air pressure low",
                "cause": "Supply pressure below limit",
                "recovery": "Adjust regulator",
                "source_doc": "Inkjet.docx",
            },
            {
                "id": 2,
                "retrieval_text": "nozzle clog ink",
                "error": "Nozzle clog",
                "cause": "Dried ink",
                "recovery": "Clean nozzle",
                "source_doc": "Inkjet.docx",
            },
        ],
    )

    assert BM25IndexCache.rebuild() == 2
    assert BM25IndexCache.search("pressure low", limit=2) == [1]
    assert BM25IndexCache.hydrate([1])[0]["error"] == "Air pressure low"


def test_embedding_wrapper_validates_dense_dimension(monkeypatch):
    from pcil.rag.embeddings import EMBEDDING_DIM, EmbeddingModelCache

    # encode() dispatches on the model class being named "SentenceTransformer",
    # so the fake must carry that exact name to exercise the real code path.
    class SentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, **kwargs):
            return [[0.0] * EMBEDDING_DIM for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=SentenceTransformer),
    )
    monkeypatch.setattr(EmbeddingModelCache, "_model", None)
    monkeypatch.setattr(EmbeddingModelCache, "_model_name", None)

    vector = EmbeddingModelCache.encode("warm up retrieval model")

    assert len(vector) == EMBEDDING_DIM


def test_rag_reindex_endpoint_uses_ingest_when_postgres(
    client, monkeypatch, tmp_path,
):
    import pcil.orchestrator as orch
    import pcil.rag.ingest as ingest

    rag_dir = tmp_path / "RAG"
    rag_dir.mkdir()
    monkeypatch.setattr(orch, "RAG_DIR", rag_dir)
    monkeypatch.setattr(orch, "_rag_backend", lambda: "postgres")
    monkeypatch.setattr(
        ingest,
        "ingest_rag_dir",
        lambda path: {
            "status": "ok",
            "rag_dir": str(path),
            "documents_seen": 1,
            "documents_changed": 1,
            "records_loaded": 2,
            "bm25_cached_records": 2,
        },
    )

    r = client.post("/rag/reindex")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rag_backend"] == "postgres"
    assert body["records_loaded"] == 2


def test_postgres_retrieval_branch_returns_search_event(monkeypatch):
    import pcil.orchestrator as orch
    import pcil.rag.hybrid as hybrid

    monkeypatch.setattr(orch, "_rag_backend", lambda: "postgres")
    monkeypatch.setattr(
        hybrid,
        "hybrid_lookup",
        lambda query, top_k=3: (
            [{"error": "Air pressure low", "cause": "", "recovery": "Adjust", "source_doc": "x.docx"}],
            {
                "bm25_record_ids": [1],
                "vector_record_ids": [1],
                "fused_record_ids": [1],
                "rrf_k": 60,
                "top_k": top_k,
                "result_count": 1,
            },
        ),
    )
    monkeypatch.setattr(hybrid, "insert_search_event", lambda query, meta: 42)

    records, search_event_id = orch._retrieve_recovery_records("air pressure low")

    assert records[0]["error"] == "Air pressure low"
    assert search_event_id == 42
