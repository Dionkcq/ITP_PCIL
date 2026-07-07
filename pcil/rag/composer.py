"""
RAG LLM composer - turns impacts + recovery records into an operator
recommendation.

Two LLM providers are supported, tried in PRIORITY ORDER (no code change to
switch, no beta headers):
  * Google Gemini (GEMINI_API_KEY)  - PRIMARY: the cheap, validated default.
  * OpenRouter (OPENROUTER_API_KEY) - FALLBACK: OpenAI-compatible; used only
    when Gemini is configured but fails, or when Gemini isn't configured.
Only ONE provider is called on success, so a working Gemini call never also
spends OpenRouter tokens. Override the model per provider with
GEMINI_MODEL / OPENROUTER_MODEL.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pcil.rag.loader import RecoveryRecord

logger = logging.getLogger(__name__)

# Cap the LLM HTTP round-trip (milliseconds). On a firewalled network the
# connection can be blackholed rather than refused; without a timeout the
# request hangs instead of falling back to the records-only response.
_LLM_TIMEOUT_MS: int = 30_000

_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def compose_recommendation(
    impacts: dict,
    records: list["RecoveryRecord"],
    *,
    target_summary: dict[str, float] | None = None,
    signal_summary: dict | None = None,
    baseline_comparison: dict | None = None,
    model: str | None = None,
) -> str:
    """Generate a plain-English operator recommendation.

    Parameters
    ----------
    impacts:
        The live-generated impacts dict from train_context_model_from_df().
        Current schema: top-level "context" with per-target
        "ranked_feature_impacts" (ranked by live contribution).
    records:
        Recovery records retrieved by the RAG layer (file lookup or the
        Postgres hybrid store). Must be non-empty (the orchestrator handles
        the empty case before calling this).
    target_summary:
        Measured window-mean of each target (0-1). Used to pick the worst
        targets and shown to the model, grounding the recommendation in
        measured performance rather than the regression intercept.
    signal_summary:
        Optional current RAW feature/target statistics for this window
        (mean/min/max/...). When present, feature lines also show the live
        raw readings, not just the normalised contribution.
    baseline_comparison:
        Optional per-feature deviation vs the stored normal-operation
        baseline ({"features": {feat: {z_score, direction}}}). When present,
        feature lines note how far each feature sits from normal.
    model:
        Explicit model name. When None (the default) each provider resolves
        its own: GEMINI_MODEL or gemini-2.5-flash for Gemini, OPENROUTER_MODEL
        or google/gemini-2.5-flash for OpenRouter.

    Returns
    -------
    str
        One concise paragraph for the operator.

    Raises
    ------
    ValueError
        If `records` is empty.
    RuntimeError
        If no LLM key is set, or every configured provider fails / returns
        empty (Gemini is tried first, OpenRouter is the fallback). The
        orchestrator catches these and degrades to a records-only response
        tagged recommendation_status="llm_unavailable", keeping failures
        distinguishable from real recommendations.
    """
    if not records:
        # The orchestrator short-circuits the no-records case (and tags it
        # recommendation_status="no_records"); this guard is for any direct
        # caller. Raise rather than return a string so "no recommendation"
        # is never mistaken for a real one.
        raise ValueError("no recovery records to compose from")

    prompt = _build_prompt(
        impacts,
        records,
        target_summary=target_summary,
        signal_summary=signal_summary,
        baseline_comparison=baseline_comparison,
    )

    # Provider selection: try Gemini FIRST, fall back to OpenRouter only if
    # Gemini is configured but fails. Only one provider is called on success,
    # so a working Gemini call never also spends OpenRouter tokens. .strip()
    # defends against a stray space/newline in the .env (e.g. "KEY = value")
    # slipping through as a non-empty-but-unusable key.
    gemini_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()

    if not gemini_key and not openrouter_key:
        raise RuntimeError(
            "No LLM API key set. Set GEMINI_API_KEY (primary) or "
            "OPENROUTER_API_KEY (fallback) before starting the orchestrator."
        )

    if gemini_key:
        try:
            return _compose_gemini(prompt, gemini_key, model)
        except Exception as gemini_error:  # noqa: BLE001
            if not openrouter_key:
                # No fallback configured: let the exception propagate so the
                # orchestrator degrades to recommendation_status=llm_unavailable.
                raise
            logger.warning(
                "Gemini composition failed (%s); falling back to OpenRouter. "
                "Fix the Gemini key/billing to stop spending OpenRouter "
                "tokens. %s",
                type(gemini_error).__name__, gemini_error,
            )

    # Either Gemini isn't configured, or it failed and OpenRouter is the
    # fallback. An OpenRouter failure propagates so the orchestrator degrades
    # to recommendation_status="llm_unavailable".
    return _compose_openrouter(prompt, openrouter_key, model)


def _compose_gemini(prompt: str, api_key: str, model: str | None) -> str:
    """Call Google Gemini via the native google-genai SDK."""
    from google import genai  # noqa: PLC0415

    model = model or os.environ.get("GEMINI_MODEL") or _DEFAULT_GEMINI_MODEL
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


def _compose_openrouter(prompt: str, api_key: str, model: str | None) -> str:
    """Call OpenRouter's OpenAI-compatible chat-completions endpoint.

    OpenRouter can route to many models; pick one with OPENROUTER_MODEL
    (default google/gemini-2.5-flash, so behaviour matches the Gemini path).
    Uses httpx, already a dependency, so no extra package is needed.
    """
    import httpx  # noqa: PLC0415

    model = model or os.environ.get("OPENROUTER_MODEL") or _DEFAULT_OPENROUTER_MODEL
    response = httpx.post(
        _OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=_LLM_TIMEOUT_MS / 1000,
    )
    if response.status_code != 200:
        # Surface OpenRouter's JSON error body (invalid key, no credits, bad
        # model id, ...) so the reason reaches the logs, not just a status.
        raise RuntimeError(
            f"OpenRouter HTTP {response.status_code}: {response.text[:200]}"
        )
    choices = response.json().get("choices") or []
    text = (choices[0].get("message", {}).get("content") if choices else "") or ""
    text = text.strip()
    if not text:
        raise RuntimeError("OpenRouter returned an empty response.")
    return text


def _build_prompt(
    impacts: dict,
    records: list["RecoveryRecord"],
    *,
    target_summary: dict[str, float] | None = None,
    signal_summary: dict | None = None,
    baseline_comparison: dict | None = None,
) -> str:
    """Construct the Gemini prompt from impacts and retrieved records.

    Worst-performing targets are chosen by their measured window mean
    (`target_summary`) - the real "how is this window doing" signal - not the
    regression intercept. The features shown are the largest LIVE CONTRIBUTORS
    (contribution = current normalised value x model weight) to those worst
    targets; each is enriched with its current RAW reading (`signal_summary`)
    and its deviation from the stored normal baseline (`baseline_comparison`)
    when those are available. The prompt forbids inventing numbers it was not
    given. Total length is kept under ~1800 characters by truncating long
    recovery texts.
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
    # model weight). Contribution, not weight alone, is what actually moved the
    # target this window: a feature with a big weight but a near-zero value
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

    sig_features = (signal_summary or {}).get("features", {})
    base_features = (baseline_comparison or {}).get("features", {})

    seen: set[str] = set()
    impact_lines: list[str] = []
    for fi in candidates:
        feat = fi["feature"]
        if feat in seen:
            continue
        seen.add(feat)
        desc = fi.get("description") or feat.replace("_", " ")

        if "contribution" in fi:
            parts = [
                f"live contribution {fi['contribution']:+.3f} "
                f"= value {fi.get('feature_value', float('nan')):.2f} "
                f"x weight {fi.get('raw_impact_score', float('nan')):+.3f}"
            ]
        else:  # archived run without contribution: fall back to the weight
            parts = [f"impact {fi.get('raw_impact_score', 0.0):+.3f}"]

        current = sig_features.get(feat, {})
        if current.get("mean") is not None:
            cmin, cmax = current.get("min"), current.get("max")
            if cmin is not None and cmax is not None:
                parts.append(
                    f"current raw mean {current['mean']:.3f}, "
                    f"range {cmin:.3f}-{cmax:.3f}"
                )
            else:
                parts.append(f"current raw mean {current['mean']:.3f}")

        baseline = base_features.get(feat, {})
        if baseline.get("z_score") is not None:
            direction = str(baseline.get("direction", "off-baseline")).replace("_", " ")
            parts.append(f"{direction} (z {baseline['z_score']:+.2f})")

        impact_lines.append(f"  - {feat} ({'; '.join(parts)}): {desc}")
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
        f"(live contribution = current normalised value x model weight; raw "
        f"means and baseline z-scores, when shown, are the actual readings; "
        f"positive contribution pushed the target up, negative pulled it "
        f"down):\n{impacts_text}\n\n"
        f"Relevant recovery records from the maintenance documents:\n"
        f"{records_text}\n\n"
        "Task: Using ONLY the information above, write a short paragraph "
        "(2-4 sentences) for a factory-floor operator. Recommend which "
        "recovery step from the records to try first and which component to "
        "inspect, and refer to the worst-performing target(s) by name. If the "
        "evidence is only correlational or no baseline comparison is shown, "
        "say so plainly. Do NOT invent sensor readings, numeric thresholds, "
        "severity levels, or causes that are not stated above; if the records "
        "only partially match, say the match is approximate. Use plain "
        "language and avoid variable names."
    )
