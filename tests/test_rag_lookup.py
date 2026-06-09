"""Tests for pcil.rag.lookup (TF-IDF retrieval).

These run fully offline — no DOCX parsing, no Gemini. They lock in the
two behaviours that motivated the v1 keyword-count -> v2 TF-IDF upgrade:
word-boundary tokenisation (no substring false-positives) and
relevance-ranked ordering.
"""

from pcil.rag.lookup import lookup_keywords


def _rec(error: str, cause: str, source_doc: str = "Test.docx") -> dict:
    return {
        "error": error,
        "cause": cause,
        "recovery": "Do the recovery steps.",
        "source_doc": source_doc,
    }


RECORDS = [
    _rec("Air pressure low", "Air supply pressure is under the operational value"),
    _rec("Nozzle clogged", "Dried ink residue blocking the nozzle"),
    _rec("Belt misalignment", "Conveyor belt tension drifted out of spec"),
]


def test_most_relevant_record_ranks_first():
    result = lookup_keywords("air pressure low supply", RECORDS, top_k=3)
    assert result, "expected at least one match"
    assert result[0]["error"] == "Air pressure low"


def test_no_substring_false_positives():
    """v1 matched 'low' inside 'flow' via substring search. TF-IDF
    tokenises on word boundaries, so 'low' must not match a record
    that only contains 'flow'."""
    records = [_rec("Ink flow interrupted", "Restricted flow in the supply tube")]
    assert lookup_keywords("low", records, top_k=3) == []


def test_zero_similarity_records_excluded():
    """top_k=3 but only one record shares vocabulary with the query —
    the other two must not be padded into the result."""
    result = lookup_keywords("nozzle ink clogged", RECORDS, top_k=3)
    assert len(result) == 1
    assert result[0]["error"] == "Nozzle clogged"


def test_top_k_limits_results():
    # 'pressure nozzle belt' touches all three records.
    result = lookup_keywords("pressure nozzle belt", RECORDS, top_k=2)
    assert len(result) == 2


def test_empty_query_and_empty_records_return_empty():
    assert lookup_keywords("", RECORDS) == []
    assert lookup_keywords("air pressure", []) == []


def test_stopword_only_query_returns_empty():
    assert lookup_keywords("the of and which", RECORDS) == []
