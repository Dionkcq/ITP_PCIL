"""Unit tests for the context model's live-contribution ranking.

These lock in the 2026-06-18 fix: ranked_feature_impacts is ordered by LIVE
CONTRIBUTION (current normalised feature value x model weight), not by the
regression weight alone. A feature the model is very sensitive to but which
sits near zero this window must NOT outrank a feature with a smaller weight
that is currently maxed out.
"""

import pandas as pd
import pytest

from pcil.train_context_model import train_context_model_from_df


def _cfg() -> dict:
    return {
        "system": "test_system",
        "input": {
            "timestamp_column": "timestamp",
            "targets": ["performance"],
            "numerical_features": ["dormant_feature", "maxed_feature"],
            "categorical_features": [],
        },
        "feature_descriptions": {
            "dormant_feature": "high weight, near-zero current value",
            "maxed_feature": "low weight, near-maxed current value",
        },
    }


def _golden_df() -> pd.DataFrame:
    """Golden DataFrame with features already MinMax-scaled to [0, 1].

    dormant_feature varies but sits near 0 (mean 0.05); maxed_feature varies
    but sits near 1 (mean 0.95). performance = 1.0*dormant + 0.2*maxed, so
    LinearRegression recovers the weights exactly: dormant gets the 5x larger
    weight, maxed the larger current value. The two columns vary independently
    so the fit is identifiable.
    """
    rows = [
        ("2026-01-01T00:00:00+00:00", 0.00, 0.90),
        ("2026-01-01T00:00:01+00:00", 0.10, 0.90),
        ("2026-01-01T00:00:02+00:00", 0.00, 1.00),
        ("2026-01-01T00:00:03+00:00", 0.10, 1.00),
        ("2026-01-01T00:00:04+00:00", 0.05, 0.95),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "dormant_feature", "maxed_feature"])
    df["performance"] = 1.0 * df["dormant_feature"] + 0.2 * df["maxed_feature"]
    # Golden DataFrame column order is timestamp + targets + features; the
    # adapter derives features as everything that isn't the timestamp or target.
    return df[["timestamp", "performance", "dormant_feature", "maxed_feature"]]


def test_impacts_rank_by_live_contribution_not_weight():
    impacts, _model = train_context_model_from_df(_golden_df(), _cfg())
    block = impacts["context"][0]
    assert block["target"] == "performance"

    by_feature = {fi["feature"]: fi for fi in block["ranked_feature_impacts"]}
    dormant = by_feature["dormant_feature"]
    maxed = by_feature["maxed_feature"]

    # The regression weight is larger for the dormant feature...
    assert abs(dormant["raw_impact_score"]) > abs(maxed["raw_impact_score"])
    # ...but it sits near zero this window, so its live contribution is smaller.
    assert abs(maxed["contribution"]) > abs(dormant["contribution"])
    # Ranking must follow live contribution: the maxed feature leads.
    assert maxed["rank"] == 1
    assert dormant["rank"] == 2


def test_contribution_equals_value_times_weight():
    impacts, _model = train_context_model_from_df(_golden_df(), _cfg())
    for fi in impacts["context"][0]["ranked_feature_impacts"]:
        assert fi["contribution"] == pytest.approx(
            fi["feature_value"] * fi["raw_impact_score"], abs=1e-9
        )
        # feature_value is the mean normalised value, so it stays in [0, 1].
        assert 0.0 <= fi["feature_value"] <= 1.0


def test_standardized_contribution_sums_to_one_in_magnitude():
    impacts, _model = train_context_model_from_df(_golden_df(), _cfg())
    for block in impacts["context"]:
        shares = [
            abs(fi["standardized_contribution"])
            for fi in block["ranked_feature_impacts"]
        ]
        # Shares are contribution / sum(|contribution|), so |shares| sum to 1
        # (unless every contribution is zero, which is not the case here).
        assert sum(shares) == pytest.approx(1.0, abs=1e-9)
