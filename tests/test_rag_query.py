"""Tests for signal-aware RAG query construction."""

from pcil.orchestrator import _build_rag_query


def test_rag_query_uses_live_signal_direction_over_static_description():
    impacts = {
        "context": [
            {
                "target": "oee",
                "intercept": 0.9,
                "ranked_feature_impacts": [
                    {
                        "feature": "air_pressure_low_ratio",
                        "description": (
                            "Proportion of operating time during which air "
                            "pressure falls below the defined threshold."
                        ),
                        "raw_impact_score": -0.4,
                    },
                    {
                        "feature": "vibration",
                        "description": "Aggregated vibration signal.",
                        "raw_impact_score": -0.2,
                    },
                ],
            }
        ],
    }
    signal_summary = {
        "features": {
            "air_pressure_low_ratio": {"mean": 0.3, "active_count": 3},
            "vibration": {"mean": 0.8, "active_count": 5},
        },
        "targets": {
            "oee": {"mean": 0.55, "status": "degraded"},
        },
    }
    baseline_comparison = {
        "features": {
            "vibration": {"direction": "above_baseline", "z_score": 2.1},
        },
    }

    query = _build_rag_query(
        impacts,
        signal_summary=signal_summary,
        baseline_comparison=baseline_comparison,
    )

    assert "oee degraded" in query
    assert "air pressure low ratio low" in query
    assert "vibration high" in query
    assert "proportion" not in query
    assert "defined threshold" not in query
