"""PostgreSQL helpers for the RAG store.

Imports psycopg lazily so the file-backed RAG path and unit tests can run
without a PostgreSQL driver installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def database_url() -> str | None:
    return os.environ.get("DATABASE_URL")


def connect(*, autocommit: bool = False):
    url = database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is not installed; install psycopg[binary] to use "
            "RAG_BACKEND=postgres"
        ) from exc
    return psycopg.connect(url, autocommit=autocommit, row_factory=dict_row)


def run_migrations() -> list[str]:
    applied: list[str] = []
    with connect(autocommit=True) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = path.name
            exists = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s",
                (version,),
            ).fetchone()
            if exists:
                continue
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)",
                (version,),
            )
            applied.append(version)
    return applied


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def fetch_all_records() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                r.id,
                r.error,
                r.cause,
                r.recovery,
                r.retrieval_text,
                d.source_doc
            FROM rag_recovery_records r
            JOIN rag_documents d ON d.id = r.document_id
            WHERE d.is_active = true
            ORDER BY d.source_doc, r.record_index
            """
        ).fetchall()
    return list(rows)
