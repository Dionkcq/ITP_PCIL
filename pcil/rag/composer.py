"""
RAG LLM composer - turns impacts + recovery records into an operator
recommendation using the Gemini API.

# TODO: when containerising, replace in-process cache with pgvector on PostgreSQL
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pcil.rag.loader import RecoveryRecord


def compose_recommendation(
    impacts: dict,
    records: list["RecoveryRecord"],
    *,
    model: str = "gemini-2.5-flash",
) -> str:
    """Generate a plain-English operator recommendation.

    Parameters
    ----------
    impacts:
        The live-generated impacts dict from train_context_model_from_df().
        Must use the current schema: top-level "context" key with
        "ranked_feature_impacts" lists (not the legacy "blocks" schema).
    records:
        Recovery records retrieved by lookup_keywords(). May be empty.
    model:
        Gemini model name. Defaults to "gemini-2.5-flash".

    Returns
    -------
    str
        One concise paragraph for the operator, or a fallback string
        if records are empty or the API call fails.
    """
    if not records:
        return (
            "No matching recovery records found. "
            "Review the feature impacts data manually."
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it before starting the orchestrator."
        )

    prompt = _build_prompt(impacts, records)

    try:
        from google import genai  # noqa: PLC0415

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text.strip()
    except Exception as exc:  # noqa: BLE001
        return (
            f"LLM composition failed ({type(exc).__name__}): {exc}. "
            "Review the recovery records below manually."
        )


def _build_prompt(impacts: dict, records: list["RecoveryRecord"]) -> str:
    """Construct the Gemini prompt from impacts and retrieved records.

    Keeps total length under ~1 800 characters by truncating long
    recovery texts.
    """
    system_name = impacts.get("system", "unknown system")

    # Identify the two worst-performing targets by lowest intercept.
    # In this linear model the intercept is the baseline predicted value
    # when all normalised features are 0 - a lower intercept indicates
    # a weaker performance baseline for that target.
    context_blocks = impacts.get("context", [])
    sorted_blocks = sorted(context_blocks, key=lambda b: b["intercept"])
    worst_two = sorted_blocks[:2]
    target_lines = "\n".join(
        f"  - {b['target']} (baseline: {b['intercept']:.3f})"
        for b in worst_two
    )

    # Collect top-ranked feature impact descriptions (deduplicated).
    seen: set[str] = set()
    impact_lines: list[str] = []
    for block in context_blocks:
        for fi in block.get("ranked_feature_impacts", [])[:1]:
            feat = fi["feature"]
            if feat not in seen:
                seen.add(feat)
                desc = fi.get("description") or feat.replace("_", " ")
                impact_lines.append(
                    f"  - {feat} (score {fi['raw_impact_score']:+.3f}): {desc}"
                )
            if len(impact_lines) >= 3:
                break
        if len(impact_lines) >= 3:
            break

    # Format recovery records; truncate long recovery text.
    record_blocks: list[str] = []
    for i, rec in enumerate(records, 1):
        recovery_text = rec["recovery"]
        if len(recovery_text) > 300:
            recovery_text = recovery_text[:297] + "..."
        record_blocks.append(
            f"Record {i} (source: {rec['source_doc']})\n"
            f"  Error:    {rec['error']}\n"
            f"  Cause:    {rec['cause']}\n"
            f"  Recovery: {recovery_text}"
        )

    records_text = "\n\n".join(record_blocks)
    impacts_text = "\n".join(impact_lines) or "  (none ranked)"

    return (
        f"Machine: {system_name}\n\n"
        f"Worst-performing targets:\n{target_lines}\n\n"
        f"Top contributing features:\n{impacts_text}\n\n"
        f"Relevant recovery records:\n{records_text}\n\n"
        "Task: Write one concise paragraph (3-5 sentences) for a factory "
        "floor operator. State what the data suggests is wrong, which "
        "physical component to inspect first, and the most important "
        "recovery step from the records above. Use plain language; avoid "
        "technical jargon or variable names."
    )
