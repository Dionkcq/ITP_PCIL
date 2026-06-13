"""DOCX -> PostgreSQL ingestion for RAG recovery records."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pcil.rag.db import connect, run_migrations, vector_literal
from pcil.rag.embeddings import EMBEDDING_DIM, EmbeddingModelCache
from pcil.rag.hybrid import BM25IndexCache
from pcil.rag.loader import load_docx


def retrieval_text(record: dict[str, str]) -> str:
    return (
        f"Error: {record.get('error', '')}\n"
        f"Cause: {record.get('cause', '')}\n"
        f"Recovery: {record.get('recovery', '')}"
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_rag_dir(rag_dir: Path) -> dict[str, Any]:
    run_migrations()
    if not rag_dir.is_dir():
        raise RuntimeError(f"RAG directory not found: {rag_dir}")

    docs_seen = 0
    docs_changed = 0
    records_loaded = 0

    for docx_path in sorted(rag_dir.glob("*.docx")):
        if "e-scentz" in docx_path.name.lower():
            continue
        docs_seen += 1
        sha = file_sha256(docx_path)
        with connect() as conn:
            existing = conn.execute(
                """
                SELECT id, sha256
                FROM rag_documents
                WHERE source_doc = %s
                """,
                (docx_path.name,),
            ).fetchone()
            if existing and existing["sha256"] == sha:
                continue

        records = load_docx(docx_path)
        texts = [retrieval_text(record) for record in records]
        embeddings = (
            EmbeddingModelCache.encode(texts)
            if texts
            else []
        )

        with connect() as conn:
            if existing:
                document_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE rag_documents
                    SET source_path = %s,
                        sha256 = %s,
                        file_mtime = %s,
                        parsed_at = now(),
                        record_count = %s,
                        is_active = true
                    WHERE id = %s
                    """,
                    (
                        str(docx_path),
                        sha,
                        datetime.fromtimestamp(
                            docx_path.stat().st_mtime,
                            tz=timezone.utc,
                        ),
                        len(records),
                        document_id,
                    ),
                )
                conn.execute(
                    "DELETE FROM rag_recovery_records WHERE document_id = %s",
                    (document_id,),
                )
            else:
                row = conn.execute(
                    """
                    INSERT INTO rag_documents (
                        source_doc, source_path, sha256, file_mtime,
                        record_count, is_active
                    )
                    VALUES (%s, %s, %s, %s, %s, true)
                    RETURNING id
                    """,
                    (
                        docx_path.name,
                        str(docx_path),
                        sha,
                        datetime.fromtimestamp(
                            docx_path.stat().st_mtime,
                            tz=timezone.utc,
                        ),
                        len(records),
                    ),
                ).fetchone()
                document_id = int(row["id"])

            for idx, (record, text, embedding) in enumerate(
                zip(records, texts, embeddings),
                start=1,
            ):
                conn.execute(
                    """
                    INSERT INTO rag_recovery_records (
                        document_id, record_index, error, cause, recovery,
                        retrieval_text, embedding, embedding_model,
                        embedding_dim, embedding_updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, now())
                    """,
                    (
                        document_id,
                        idx,
                        record["error"],
                        record.get("cause", ""),
                        record["recovery"],
                        text,
                        vector_literal(embedding),
                        EmbeddingModelCache.model_name(),
                        EMBEDDING_DIM,
                    ),
                )
            conn.commit()

        docs_changed += 1
        records_loaded += len(records)

    cached_records = BM25IndexCache.rebuild()
    return {
        "status": "ok",
        "rag_dir": str(rag_dir),
        "documents_seen": docs_seen,
        "documents_changed": docs_changed,
        "records_loaded": records_loaded,
        "bm25_cached_records": cached_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest RAG DOCX files into PostgreSQL.")
    parser.add_argument("--rag-dir", required=True, type=Path)
    args = parser.parse_args()
    print(ingest_rag_dir(args.rag_dir))


if __name__ == "__main__":
    main()
