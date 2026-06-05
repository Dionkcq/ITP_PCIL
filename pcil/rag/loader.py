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

import re
import sys
from pathlib import Path
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
    """Parse one DOCX into a list of RecoveryRecord dicts.

    The Model Factory machine docs put each field on its own paragraph in
    "Label: value" form, e.g.::

        Error 1:
        Error Message: Air pressure low.
        Root Cause: Air supply pressure is below operational value; ...
        Critical: Yes
        Recovery Steps: Increase air supply by adjusting regulator.
        Preventive Actions: ...

    Each block begins with an "Error Message:" line; "Root Cause:" and
    "Recovery Steps:" follow. Other fields (Critical, Resolvable,
    Preventive Actions, ...) are ignored. A record is kept only when it has
    both an error message and recovery steps.
    """
    cache_key = str(docx_path)
    if cache_key in _RECORD_CACHE:
        return _RECORD_CACHE[cache_key]

    from docx import Document  # noqa: PLC0415 - keep docx as optional dep

    doc = Document(docx_path)
    records: list[RecoveryRecord] = []

    error_re = re.compile(r"^error\s*message\s*:\s*(.*)$", re.IGNORECASE)
    cause_re = re.compile(r"^root\s*cause\s*:\s*(.*)$", re.IGNORECASE)
    recovery_re = re.compile(r"^recovery\s*steps?\s*:\s*(.*)$", re.IGNORECASE)

    def _flush(cur: dict[str, str]) -> None:
        if cur.get("error") and cur.get("recovery"):
            records.append(RecoveryRecord(
                error=cur.get("error", ""),
                cause=cur.get("cause", ""),
                recovery=cur.get("recovery", ""),
                source_doc=docx_path.name,
            ))

    current: dict[str, str] = {}
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        m = error_re.match(text)
        if m:
            _flush(current)                       # new block boundary
            current = {"error": m.group(1).strip()}
            continue
        m = cause_re.match(text)
        if m:
            current["cause"] = m.group(1).strip()
            continue
        m = recovery_re.match(text)
        if m:
            current["recovery"] = m.group(1).strip()
            continue

    _flush(current)                               # final block

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
