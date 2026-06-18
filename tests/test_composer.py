"""Unit tests for the RAG composer's prompt construction (pcil.rag.composer).

These lock in the grounding fixes from the 2026-06-13 review:
  - worst-performing targets are chosen by MEASURED window means
    (target_summary), not by the regression intercept;
  - the prompt forbids the model from inventing numbers it was not given.

They exercise _build_prompt directly so no Gemini API call is made.
"""

from pcil.rag.composer import _build_prompt


def _impacts():
    """Impacts where intercept order and measured-mean order DISAGREE.

    By intercept, 'quality' (0.10) looks worst. By measured mean (supplied
    via target_summary in the tests), 'oee'/'performance' are worst. The
    prompt must follow the measured means.
    """
    return {
        "system": "inkjet_printer",
        "context": [
            {"target": "availability", "intercept": 0.90,
             "ranked_feature_impacts": [
                 {"feature": "vibration", "raw_impact_score": 0.5,
                  "description": "machine vibration"}]},
            {"target": "performance", "intercept": 0.80,
             "ranked_feature_impacts": []},
            {"target": "quality", "intercept": 0.10,
             "ranked_feature_impacts": []},
            {"target": "oee", "intercept": 0.95,
             "ranked_feature_impacts": []},
        ],
    }


_RECORDS = [{
    "error": "Print head clog",
    "cause": "Dried ink in nozzle",
    "recovery": "Run cleaning cycle and prime the head.",
    "source_doc": "Inkjet.docx",
}]


def test_prompt_picks_worst_targets_by_measured_value_not_intercept():
    target_summary = {
        "availability": 0.95,
        "performance": 0.40,
        "quality": 0.85,
        "oee": 0.20,
    }
    prompt = _build_prompt(_impacts(), _RECORDS, target_summary=target_summary)
    # Everything before the features section is the worst-targets block.
    head = prompt.split("Features most associated")[0]
    # Worst two by MEASURED mean are oee (0.20) and performance (0.40).
    assert "oee" in head
    assert "performance" in head
    # 'quality' has the LOWEST intercept (0.10) but a healthy measured mean
    # (0.85), so the old intercept logic would have flagged it. It must NOT
    # appear as worst-performing now.
    assert "quality" not in head


def test_prompt_forbids_fabrication():
    prompt = _build_prompt(_impacts(), _RECORDS, target_summary={"oee": 0.2})
    low = prompt.lower()
    assert "do not invent" in low
    assert "only the information above" in low


def test_prompt_falls_back_to_intercept_without_summary():
    # Defensive path: no target_summary -> intercept ordering.
    prompt = _build_prompt(_impacts(), _RECORDS, target_summary=None)
    head = prompt.split("Features")[0]
    # Lowest intercepts are quality (0.10) and performance (0.80).
    assert "quality" in head
    assert "performance" in head


def _impacts_with_contribution():
    """Worst target (oee) carries two features. air_pressure has the SMALLER
    weight but a maxed current value, so its live contribution is larger;
    setvelo has the larger weight but a near-zero value. Live-contribution
    ranking must surface air_pressure first."""
    return {
        "system": "inkjet_printer",
        "context": [
            {"target": "oee", "intercept": 0.5, "ranked_feature_impacts": [
                {"feature": "air_pressure_low_ratio", "raw_impact_score": -0.5,
                 "feature_value": 0.98, "contribution": -0.49,
                 "standardized_contribution": -0.86,
                 "description": "air pressure below threshold ratio"},
                {"feature": "setvelo_mean", "raw_impact_score": 0.8,
                 "feature_value": 0.10, "contribution": 0.08,
                 "standardized_contribution": 0.14,
                 "description": "mean commanded velocity"},
            ]},
            {"target": "availability", "intercept": 0.9,
             "ranked_feature_impacts": []},
        ],
    }


def test_prompt_features_ranked_by_live_contribution_not_weight():
    prompt = _build_prompt(
        _impacts_with_contribution(), _RECORDS,
        target_summary={"oee": 0.20, "availability": 0.95},
    )
    low = prompt.lower()
    # air_pressure has the smaller weight (0.5 < 0.8) but the larger live
    # contribution (|-0.49| > |0.08|) because its value is maxed - it must lead.
    assert "air_pressure_low_ratio" in prompt
    assert "live contribution" in low
    # The signed contribution value is shown to the model, not just the weight.
    assert "-0.490" in prompt
    # ...and it is ranked ahead of the higher-weight but dormant feature.
    assert prompt.index("air_pressure_low_ratio") < prompt.index("setvelo_mean")
