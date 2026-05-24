"""
Cyclical anomaly pipeline — Step 4: model architecture + training
==================================================================
Pick ONE candidate. All four implement the same `.fit(X)` / `.score(X)`
interface so train.py and score.py work the same way regardless of
which one is chosen.

Candidates
----------
  ZScoreModel          — pure statistics, no training. Good baseline.
  IsolationForestModel — sklearn-native, light. Good first NN-free choice.
  OneClassSVMModel     — classic anomaly detector.
  AutoencoderModel     — heaviest, most flexible.

Design choice: Isolation Forest
---------------------------------
  - Trains unsupervised on normal cycles only — no anomaly labels needed.
  - Randomly partitions the feature space into isolation trees. Anomalous
    cycles sit in sparse regions and get isolated in fewer splits, giving
    them a higher anomaly score.
  - Sklearn-native, no neural net, no hyperparameter tuning required for
    a first pass.
  - Loss function : n/a (tree-based, not gradient-based)
  - Optimiser     : n/a
  - Stopping criterion : fixed n_estimators (100 trees by default);
    no iterative training loop.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class AnomalyModel(Protocol):
    """The interface every candidate satisfies."""

    def fit(self, X: np.ndarray) -> "AnomalyModel": ...
    def score(self, X: np.ndarray) -> np.ndarray: ...


# ─────────────────────────────────────────────────────────────
# Candidate 1: Z-score baseline (no training, pure statistics)
# ─────────────────────────────────────────────────────────────

class ZScoreModel:
    """
    Flag rows whose maximum absolute z-score across features exceeds
    `threshold`. "Trains" by storing the training set's per-feature
    mean and std.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mu_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZScoreModel":
        raise NotImplementedError("ZScoreModel not selected — use IsolationForestModel.")

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ZScoreModel not selected — use IsolationForestModel.")


# ─────────────────────────────────────────────────────────────
# Candidate 2: Isolation Forest (sklearn)  ← SELECTED
# ─────────────────────────────────────────────────────────────

class IsolationForestModel:
    """
    Wraps sklearn.ensemble.IsolationForest. Higher `score` = more anomalous.

    Default kwargs:
      n_estimators=100  — number of isolation trees
      contamination=0.05 — expected fraction of anomalies in training data
      random_state=42   — reproducibility
    """

    def __init__(self, **kwargs):
        from sklearn.ensemble import IsolationForest

        # Sensible defaults — caller can override via kwargs
        params = {"n_estimators": 100, "contamination": 0.05, "random_state": 42}
        params.update(kwargs)
        self._model = IsolationForest(**params)

    def fit(self, X: np.ndarray) -> "IsolationForestModel":
        """Fit the isolation forest on normal training cycles."""
        self._model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return per-row anomaly score where higher = more anomalous.
        sklearn's score_samples returns negative values (more negative = more anomalous),
        so we negate to make the direction intuitive.
        """
        return -self._model.score_samples(X)


# ─────────────────────────────────────────────────────────────
# Candidate 3: One-class SVM (sklearn)
# ─────────────────────────────────────────────────────────────

class OneClassSVMModel:
    """Wraps sklearn.svm.OneClassSVM. Higher `score` = more anomalous."""

    def __init__(self, **kwargs):
        from sklearn.svm import OneClassSVM
        self._model = OneClassSVM(**kwargs)

    def fit(self, X: np.ndarray) -> "OneClassSVMModel":
        raise NotImplementedError("OneClassSVMModel not selected — use IsolationForestModel.")

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("OneClassSVMModel not selected — use IsolationForestModel.")


# ─────────────────────────────────────────────────────────────
# Candidate 4: Autoencoder (sketch — pick a NN library)
# ─────────────────────────────────────────────────────────────

class AutoencoderModel:
    """
    Reconstruction-error anomaly detector.
    Not selected for Week 2 — Isolation Forest is sufficient.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def fit(self, X: np.ndarray) -> "AutoencoderModel":
        raise NotImplementedError("AutoencoderModel not selected — use IsolationForestModel.")

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("AutoencoderModel not selected — use IsolationForestModel.")