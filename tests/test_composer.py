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
    head = prompt.split("Features most associated")[0]
    # Lowest intercepts are quality (0.10) and performance (0.80).
    assert "quality" in head
    assert "performance" in head
