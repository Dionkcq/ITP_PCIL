"""
RAG document loader
====================
Parse a DOCX recovery document into structured records:
    {error, cause, recovery}

Six of the 7 docs in `data/RAG/` follow a similar structure with
"Error Message" / "Root Cause" / "Recovery Steps" headings. The seventh
(`E-Scentz.docx`) is product overview only — skip it.

Dependency: python-docx
    pip install python-docx
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TypedDict

# Per-file parse cache: path string -> list of records parsed from that file.
_RECORD_CACHE: dict[str, list["RecoveryRecord"]] = {}

# Aggregate cache: str(rag_dir) -> full concatenated list across all docs.
# Populated by load_all_recovery_docs on first call per directory.
# TODO: when containerising, replace in-process cache with pgvector on PostgreSQL
_ALL_RECORDS_CACHE: dict[str, list["RecoveryRecord"]] = {}


class RecoveryRecord(TypedDict):
    error: str
    cause: str
    recovery: str
    source_doc: str        # the DOCX filename, for traceability


def load_docx(docx_path: Path) -> list[RecoveryRecord]:
    """
    Parse one DOCX into a list of RecoveryRecord dicts.

    TODO (teammate):
      1. Open the DOCX with `from docx import Document`.
      2. Walk paragraphs/tables and detect the heading pattern
         (Error Message / Root Cause / Recovery Steps). Headings vary —
         inspect the doc structure first.
      3. Group each (error, cause, recovery) trio into one record.
      4. Return list[RecoveryRecord].

    Heads up:
      - Some docs use tables, some use paragraphs.
      - Pick ONE doc to start with (e.g. Screen Printer.docx — Dion read
        it earlier and confirmed it has 19 structured error blocks).
    """
    cache_key = str(docx_path)
    if cache_key in _RECORD_CACHE:
      return _RECORD_CACHE[cache_key]

    from docx import Document  # noqa: PLC0415 - keep docx as optional dep

    doc = Document(docx_path)
    records: list[RecoveryRecord] = []

    current: dict[str, str] = {}
    heading_map = {
        "error message": "error",
        "root cause": "cause",
        "recovery steps": "recovery",
    }

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        key = heading_map.get(text.lower())
        if key:
            current[key] = ""
        elif current:
            # Accumulate body text under the most recently seen heading.
            last_key = list(current)[-1]
            current[last_key] = (current[last_key] + " " + text).strip()

        if all(k in current for k in ("error", "cause", "recovery")):
            records.append(RecoveryRecord(
                error=current["error"],
                cause=current["cause"],
                recovery=current["recovery"],
                source_doc=docx_path.name,
            ))
            current = {}

    if len(records) < 5:
        print(
            f"[loader] WARNING: only {len(records)} records found in "
            f"{docx_path.name}; expected >= 5.",
            file=sys.stderr,
        )

    _RECORD_CACHE[cache_key] = records
    return records


def load_all_recovery_docs(rag_dir: Path) -> list[RecoveryRecord]:
    """
    Convenience wrapper: load every *.docx in `rag_dir` (skipping
    E-Scentz.docx) and concatenate the records.

    TODO (teammate):
      1. Iterate rag_dir.glob("*.docx").
      2. Skip "E-Scentz.docx".
      3. Call load_docx on each, extend the result list.
      4. Return.
    """
    cache_key = str(rag_dir)
    if cache_key in _ALL_RECORDS_CACHE:
        return _ALL_RECORDS_CACHE[cache_key]

    if not rag_dir.is_dir():
        return []

    all_records: list[RecoveryRecord] = []
    for docx_path in sorted(rag_dir.glob("*.docx")):
        if "e-scentz" in docx_path.name.lower():
            continue
        all_records.extend(load_docx(docx_path))

    _ALL_RECORDS_CACHE[cache_key] = all_records
    return all_records
