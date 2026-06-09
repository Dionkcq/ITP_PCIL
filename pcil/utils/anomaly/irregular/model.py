"""
Irregular anomaly pipeline — Step 4: model
===========================================
One model, sharing the .fit(X) / .score(X) interface from base.py.

Selected: IsolationForestModel
-------------------------------
Unsupervised — event-log data almost never comes with anomaly labels,
so a supervised model (like the non-cyclical Random Forest) is not an
option here. Isolation Forest works directly on the small per-window
feature vectors, needs no distributional assumptions about gap lengths
(which are heavy-tailed for event data), and is cheap enough to retrain
per machine.

The class is intentionally a sibling of cyclical/model.py's
IsolationForestModel rather than an import from it — subpackages stay
independent (only base.py is shared) so one team's refactor cannot
break another's pipeline.
"""

from __future__ import annotations

import numpy as np

from pcil.utils.anomaly.base import AnomalyModel


class IsolationForestModel(AnomalyModel):
    """sklearn IsolationForest. Higher score = more anomalous."""

    def __init__(self, **kwargs):
        from sklearn.ensemble import IsolationForest
        params = {"n_estimators": 100, "contamination": 0.05, "random_state": 42}
        params.update(kwargs)
        self._model = IsolationForest(**params)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IsolationForestModel":
        self._model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(X)
