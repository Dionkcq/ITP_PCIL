CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_documents (
    id BIGSERIAL PRIMARY KEY,
    source_doc TEXT NOT NULL UNIQUE,
    source_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    file_mtime TIMESTAMPTZ,
    parsed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    parser_version TEXT NOT NULL DEFAULT 'docx_recovery_v1',
    record_count INTEGER NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE IF NOT EXISTS rag_recovery_records (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    record_index INTEGER NOT NULL,
    error TEXT NOT NULL,
    cause TEXT NOT NULL DEFAULT '',
    recovery TEXT NOT NULL,
    retrieval_text TEXT NOT NULL,
    embedding vector(1024),
    embedding_model TEXT NOT NULL DEFAULT 'BAAI/bge-m3',
    embedding_dim INTEGER NOT NULL DEFAULT 1024,
    embedding_updated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, record_index)
);

CREATE INDEX IF NOT EXISTS idx_rag_records_embedding
ON rag_recovery_records
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_rag_records_document_id
ON rag_recovery_records(document_id);

CREATE TABLE IF NOT EXISTS rag_search_events (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    query TEXT NOT NULL,
    retrieval_mode TEXT NOT NULL,
    bm25_record_ids BIGINT[] NOT NULL DEFAULT '{}',
    vector_record_ids BIGINT[] NOT NULL DEFAULT '{}',
    fused_record_ids BIGINT[] NOT NULL DEFAULT '{}',
    rrf_k INTEGER NOT NULL DEFAULT 60,
    top_k INTEGER NOT NULL,
    result_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS rag_recommendations (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    search_event_id BIGINT REFERENCES rag_search_events(id) ON DELETE SET NULL,
    impacts JSONB,
    signal_summary JSONB,
    baseline_comparison JSONB,
    recommendation_text TEXT NOT NULL,
    recommendation_source TEXT NOT NULL,
    recommendation_warnings JSONB NOT NULL DEFAULT '[]'::jsonb
);
