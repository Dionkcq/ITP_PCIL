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

# Cap the Gemini HTTP round-trip (milliseconds). On a firewalled network
# the connection can be blackholed rather than refused; without a timeout
# the request hangs instead of falling back to the records-only response.
_LLM_TIMEOUT_MS: int = 30_000


def compose_recommendation(
    impacts: dict,
    records: list["RecoveryRecord"],
    *,
    model: str = "gemini-2.5-flash",
    target_summary: dict[str, float] | None = None,
) -> str:
    """Generate a plain-English operator recommendation.

    Parameters
    ----------
    impacts:
        The live-generated impacts dict from train_context_model_from_df().
        Must use the current schema: top-level "context" key with
        "ranked_feature_impacts" lists (not the legacy "blocks" schema).
    records:
        Recovery records retrieved by lookup_keywords(). Must be non-empty
        (the orchestrator handles the empty case before calling this).
    model:
        Gemini model name. Defaults to "gemini-2.5-flash".
    target_summary:
        Measured window-mean of each target (0-1). When provided, the worst
        targets are chosen by these real values and the values are shown to
        the model, so the recommendation is grounded in measured performance
        rather than the regression intercept.

    Returns
    -------
    str
        One concise paragraph for the operator.

    Raises
    ------
    ValueError
        If `records` is empty.
    RuntimeError
        If GEMINI_API_KEY is unset, the API call fails, or the model returns
        an empty response. The orchestrator catches these and degrades to a
        records-only response tagged recommendation_status="llm_unavailable",
        keeping failures distinguishable from real recommendations.
    """
    if not records:
        # The orchestrator short-circuits the no-records case (and tags it
        # recommendation_status="no_records"); this guard is for any direct
        # caller. Raise rather than return a string so "no recommendation"
        # is never mistaken for a real one.
        raise ValueError("no recovery records to compose from")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it before starting the orchestrator."
        )

    prompt = _build_prompt(impacts, records, target_summary=target_summary)

    # No try/except here: failures (missing key, no internet, timeout,
    # empty/safety-filtered response) propagate so the orchestrator can
    # degrade with a machine-readable recommendation_status instead of the
    # caller having to sniff the returned text for marker words.
    from google import genai  # noqa: PLC0415

    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": _LLM_TIMEOUT_MS},
    )
    response = client.models.generate_content(model=model, contents=prompt)
    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response (possibly safety-filtered)."
        )
    return response.text.strip()


def _build_prompt(
    impacts: dict,
    records: list["RecoveryRecord"],
    target_summary: dict[str, float] | None = None,
) -> str:
    """Construct the Gemini prompt from impacts and retrieved records.

    Worst-performing targets are chosen by their actual window-mean value
    (`target_summary`) when available - the real "how is this window doing"
    signal - instead of the regression intercept, which is only the predicted
    value when all scaled features are 0 and does not reflect current
    performance. The features shown are the largest LIVE CONTRIBUTORS to those
    worst targets (contribution = current normalised value x model weight), not
    the features with the largest weight regardless of their current value. The
    prompt feeds those measured values to the model and forbids inventing
    numbers it was not given, so it cannot fabricate severity. Total length is
    kept under ~1 800 characters by truncating long recovery texts.
    """
    system_name = impacts.get("system", "unknown system")
    context_blocks = impacts.get("context", [])

    # --- Worst-performing targets -----------------------------------
    if target_summary:
        worst_two = sorted(
            context_blocks,
            key=lambda b: target_summary.get(b["target"], float("inf")),
        )[:2]
        target_lines = "\n".join(
            f"  - {b['target']}: {target_summary.get(b['target'], float('nan')):.3f}"
            for b in worst_two
        )
        targets_header = (
            "Worst-performing targets this window (measured 0-1, lower = worse)"
        )
    else:
        # Defensive fallback: the orchestrator normally passes target_summary.
        worst_two = sorted(context_blocks, key=lambda b: b["intercept"])[:2]
        target_lines = "\n".join(
            f"  - {b['target']} (baseline {b['intercept']:.3f})"
            for b in worst_two
        )
        targets_header = "Worst-performing targets (by regression baseline)"

    # --- Top live contributors to the worst targets ----------------
    # Pull features from the worst-performing blocks only and keep the three
    # with the largest absolute LIVE CONTRIBUTION (current normalised value x
    # model weight). Contribution, not the weight alone, is what actually moved
    # the target this window: a feature with a big weight but a near-zero value
    # contributes almost nothing. ranked_feature_impacts already arrives sorted
    # by |contribution|, but we re-sort defensively across the pooled blocks.
    candidates: list[dict] = []
    for block in worst_two:
        candidates.extend(block.get("ranked_feature_impacts", []))
    if not candidates:  # worst-two carried no impacts; fall back to all blocks
        for block in context_blocks:
            candidates.extend(block.get("ranked_feature_impacts", []))

    def _abs_contribution(fi: dict) -> float:
        # New schema carries "contribution"; fall back to the coefficient for
        # archived impacts produced before live contribution existed.
        return abs(fi.get("contribution", fi.get("raw_impact_score", 0.0)))

    candidates.sort(key=_abs_contribution, reverse=True)

    seen: set[str] = set()
    impact_lines: list[str] = []
    for fi in candidates:
        feat = fi["feature"]
        if feat in seen:
            continue
        seen.add(feat)
        desc = fi.get("description") or feat.replace("_", " ")
        if "contribution" in fi:
            impact_lines.append(
                f"  - {feat} (live contribution {fi['contribution']:+.3f} "
                f"= value {fi.get('feature_value', float('nan')):.2f} "
                f"x weight {fi.get('raw_impact_score', float('nan')):+.3f}): {desc}"
            )
        else:  # archived run without contribution: fall back to the weight
            impact_lines.append(
                f"  - {feat} (impact {fi.get('raw_impact_score', 0.0):+.3f}): {desc}"
            )
        if len(impact_lines) >= 3:
            break

    # --- Recovery records (truncate long recovery text) -------------
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
        f"{targets_header}:\n{target_lines}\n\n"
        f"Features contributing most to the worst target(s) this window "
        f"(live contribution = current normalised value x model weight; these "
        f"are model quantities, not raw sensor readings; positive pushed the "
        f"target up, negative pulled it down):\n{impacts_text}\n\n"
        f"Relevant recovery records from the maintenance documents:\n"
        f"{records_text}\n\n"
        "Task: Using ONLY the information above, write a short paragraph "
        "(2-4 sentences) for a factory-floor operator. Recommend which "
        "recovery step from the records to try first and which component to "
        "inspect, and refer to the worst-performing target(s) by name. Do "
        "NOT invent sensor readings, numeric thresholds, severity levels, or "
        "causes that are not stated above; if the records only partially "
        "match, say the match is approximate. Use plain language and avoid "
        "variable names."
    )
