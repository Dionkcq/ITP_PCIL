import numpy as np

class RandomForestModel:
    """
    Supervised anomaly classifier.
    Trains on clean (label=0) and anomaly (label=1) windows.
    Returns anomaly probability as the score (higher = more anomalous).
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self._model.fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]