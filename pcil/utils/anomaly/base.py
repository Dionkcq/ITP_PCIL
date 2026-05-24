"""
Shared base classes + utilities for anomaly detection.
=======================================================

Lives at the top of `pcil/utils/anomaly/` so both `cyclical/` and
`non_cyclical/` subpackages can import from it without depending on
each other.

Two exports:

  - `AnomalyModel` — abstract base class. Cyclical / non-cyclical model
    classes inherit from this so the orchestrator can dispatch
    /anomaly/score uniformly. Supervised models use the optional `y`
    in fit(); unsupervised models ignore it.

  - `PerMachineNormaliser` — z-score normaliser that maintains a separate
    mean/std per machine_id. Fit on clean training data only; apply with
    transform() at inference time. Shared by both cyclical and
    non-cyclical pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# AnomalyModel — common interface every per-data-type model implements
# ─────────────────────────────────────────────────────────────────────

class AnomalyModel(ABC):
    """Common interface for anomaly detection models.

    Cyclical (Daniel/Jaymon) and non-cyclical (Zi Hin) pipelines both
    expose model classes that satisfy this contract. The orchestrator
    dispatches `/anomaly/score` to whichever subpackage matches the
    requested `model_type`, and that subpackage's model implements this
    interface.

    Two methods:
      - `fit(X, y=None)`  — supervised models use y, unsupervised ignore.
      - `score(X)`        — returns per-row anomaly scores (higher = more
                            anomalous). Float in [0, 1] for probabilistic
                            models; arbitrary range for distance-based models.

    Persistence is bundle-level (joblib.dump({"model": model, ...})), not
    per-class — keeps the model object simple and lets each pipeline carry
    its own normaliser / metadata in the same .pkl.
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "AnomalyModel":
        """Fit the model on training data. Return self for chaining."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return per-row anomaly scores. Higher = more anomalous."""


# ─────────────────────────────────────────────────────────────────────
# PerMachineNormaliser — shared z-score normaliser keyed by machine_id
# ─────────────────────────────────────────────────────────────────────

class PerMachineNormaliser:
    """Z-score normaliser that maintains a separate mean/std per machine_id.

    Fit on clean training data only (never on test or anomaly data, or you
    leak the anomaly distribution into the baseline). Apply with transform()
    at inference time.
    """

    def __init__(self) -> None:
        self.stats_: dict[str, dict[str, np.ndarray]] = {}
        self.feature_columns_: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        *,
        machine_id_column: str,
        feature_columns: list[str],
    ) -> "PerMachineNormaliser":
        """Compute mean and std for each feature, separately per machine."""
        self.feature_columns_ = list(feature_columns)
        self.stats_ = {}

        for machine_id, group in df.groupby(machine_id_column):
            X = group[feature_columns].to_numpy(dtype=float)
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            # Avoid division by zero on constant features.
            std = np.where(std == 0, 1.0, std)
            self.stats_[machine_id] = {"mean": mean, "std": std}

        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        machine_id_column: str,
    ) -> pd.DataFrame:
        """Apply the fitted z-score transform to a new DataFrame."""
        if not self.stats_:
            raise RuntimeError("Call fit() or fit_transform() before transform().")

        result = df.copy()

        for machine_id, group_idx in df.groupby(machine_id_column).groups.items():
            if machine_id not in self.stats_:
                raise KeyError(
                    f"Machine '{machine_id}' was not seen during fit(). "
                    f"Known machines: {list(self.stats_)}"
                )
            mean = self.stats_[machine_id]["mean"]
            std = self.stats_[machine_id]["std"]
            X = df.loc[group_idx, self.feature_columns_].to_numpy(dtype=float)
            result.loc[group_idx, self.feature_columns_] = (X - mean) / std

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        *,
        machine_id_column: str,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        return self.fit(
            df,
            machine_id_column=machine_id_column,
            feature_columns=feature_columns,
        ).transform(df, machine_id_column=machine_id_column)
