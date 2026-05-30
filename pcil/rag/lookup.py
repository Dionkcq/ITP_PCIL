"""
RAG keyword lookup
===================
Given a query string and a list of RecoveryRecord, return matching records.

v1: plain substring match against `error` / `cause` fields.
v2 (stretch): vector embeddings + cosine similarity.
"""

from __future__ import annotations

from pcil.rag.loader import RecoveryRecord

_STOPWORDS: frozenset[str] = frozenset({
    "the", "is", "a", "an", "of", "in", "and", "or", "to", "for",
    "with", "on", "at", "from", "by", "this", "that", "it", "be",
    "was", "has", "have", "not", "are", "as", "but", "if", "so",
    "defined", "below", "which", "during", "time", "proportion",
})


def lookup_keywords(
    query: str,
    records: list[RecoveryRecord],
    *,
    top_k: int = 3,
) -> list[RecoveryRecord]:
    """
    Return the top_k records whose `error` or `cause` field contains
    the most matching tokens from `query`.

    TODO (teammate):
      1. Lowercase + tokenise the query (split on whitespace, drop
         common stopwords like "the", "is", etc.).
      2. For each record, count how many query tokens appear in
         record["error"] + record["cause"].
      3. Sort records by count descending, return top_k.
      4. If nothing matches, return an empty list.

    Stretch:
      - Use sentence-transformers + cosine similarity instead of a
        bag-of-words count.
      - Re-rank with the LLM after retrieval.
    """
    if not query or not records:
      return []

    tokens = [
        t for t in query.lower().split()
        if t not in _STOPWORDS and len(t) > 2
    ]
    if not tokens:
        return []

    def score(record: RecoveryRecord) -> int:
      haystack = (record["error"] + " " + record["cause"]).lower()
      return sum(1 for t in tokens if t in haystack)

    scored = [(score(r), r) for r in records]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for count, r in scored[:top_k] if count > 0]
