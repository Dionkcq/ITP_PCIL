"""Tests for pcil.rag.loader against the REAL document format.

The 2026-06-05 incident: the loader originally scanned for standalone
"Error Message" headings, but the actual Model Factory docs use inline
"Label: value" paragraphs — so it silently parsed 0 records from every
file. These tests build a small DOCX in that real format so any future
format drift in the parser fails loudly instead of degrading to an
empty RAG corpus.

Skipped automatically when python-docx is not installed.
"""

import pytest

docx = pytest.importorskip("docx")

from pcil.rag.loader import load_all_recovery_docs, load_docx  # noqa: E402


def _write_docx(path, paragraphs):
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    document.save(path)


REAL_FORMAT_PARAGRAPHS = [
    "2.1 Error Handling",
    "Error 1:",
    "Error Message: Air pressure low.",
    "Root Cause: Air supply pressure is below operational value.",
    "Critical: Yes",
    "Resolvable: Yes",
    "Recovery Steps: Increase air supply by adjusting the regulator.",
    "Preventive Actions: Inspect the air supply line weekly.",
    "Error 2:",
    "Error Message: Conveyor jam detected.",
    "Root Cause: Product misaligned on the belt.",
    "Recovery Steps: Remove the jammed product and restart the conveyor.",
    # Incomplete block — has an error but no recovery steps; must be dropped.
    "Error 3:",
    "Error Message: Sensor offline.",
    "Root Cause: Cable disconnected.",
]


def test_load_docx_parses_real_label_value_format(tmp_path):
    doc_path = tmp_path / "Test_Machine.docx"
    _write_docx(doc_path, REAL_FORMAT_PARAGRAPHS)

    records = load_docx(doc_path)

    assert len(records) == 2  # third block lacks Recovery Steps
    assert records[0]["error"] == "Air pressure low."
    assert records[0]["cause"] == "Air supply pressure is below operational value."
    assert records[0]["recovery"] == "Increase air supply by adjusting the regulator."
    assert records[0]["source_doc"] == "Test_Machine.docx"
    assert records[1]["error"] == "Conveyor jam detected."


def test_load_all_recovery_docs_skips_escentz(tmp_path):
    _write_docx(tmp_path / "Machine_A.docx", REAL_FORMAT_PARAGRAPHS)
    _write_docx(tmp_path / "E-Scentz.docx", REAL_FORMAT_PARAGRAPHS)

    records = load_all_recovery_docs(tmp_path)

    assert len(records) == 2
    assert all(r["source_doc"] == "Machine_A.docx" for r in records)
