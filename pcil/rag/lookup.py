"""
RAG lookup
===========
Given a query string and a list of RecoveryRecord, return the most
relevant records.

v1: keyword bag-of-words count (substring match against error/cause).
v2 (current): TF-IDF + cosine similarity over the error/cause text.

Why TF-IDF over the v1 keyword count
-------------------------------------
- Word-boundary tokenisation: the v1 substring check matched "low"
  inside "flow"/"below", inflating scores for unrelated records.
- Term weighting: a word that appears in nearly every record (e.g.
  "machine") barely discriminates; TF-IDF down-weights it, while a
  rare, specific word ("misalignment") counts for more.
- Still fully local and deterministic — no embeddings API, no network.
  Uses scikit-learn, which is already a core project dependency.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

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
    Return the top_k records ranked by TF-IDF cosine similarity between
    `query` and each record's `error` + `cause` text.

    Records with zero similarity (no shared vocabulary) are excluded,
    so the result can be shorter than top_k — or empty when nothing
    matches at all.

    The name says "keywords" for backwards compatibility with v1; the
    query is still a plain space-separated keyword string (built by the
    orchestrator's _build_rag_query()).
    """
    if not query or not records:
        return []

    corpus = [f"{r['error']} {r['cause']}" for r in records]

    vectoriser = TfidfVectorizer(
        lowercase=True,
        stop_words=list(_STOPWORDS),
        token_pattern=r"(?u)\b\w\w\w+\b",  # 3+ chars, mirrors v1's len > 2 filter
    )
    try:
        doc_matrix = vectoriser.fit_transform(corpus)
        query_vec = vectoriser.transform([query])
    except ValueError:
        # Corpus or query reduced to an empty vocabulary (e.g. all
        # stopwords) — nothing meaningful to rank against.
        return []

    # TfidfVectorizer output is L2-normalised, so the dot product IS the
    # cosine similarity.
    similarities = (doc_matrix @ query_vec.T).toarray().ravel()

    ranked = sorted(range(len(records)), key=lambda i: similarities[i], reverse=True)
    return [records[i] for i in ranked[:top_k] if similarities[i] > 0.0]
