"""
Non-cyclical anomaly model training (reusable).
=================================================
Importable training function for the non-cyclical pipeline. Two callers:

  - The /anomaly/train API endpoint (engineer uploads CSVs at runtime).
  - run.py CLI (preserves Zi Hin's original dev workflow with the
    config-driven file paths + threshold sweep evaluation).

The function takes pandas DataFrames (not file paths) so callers can
provide data from any source: HTTP upload, file, in-memory generator.

The output bundle dict has the same shape as the one previously produced
by run.py, so score.py / `/anomaly/score` consume it unchanged.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from pcil.utils.anomaly.base import PerMachineNormaliser
from pcil.utils.anomaly.non_cyclical.features import (
    CHANNEL_COLUMNS as DEFAULT_CHANNEL_COLUMNS,
    DEFAULT_FEATURE_NAMES,
    extract_features,
    stack_features,
)
from pcil.utils.anomaly.non_cyclical.model import RandomForestModel
from pcil.utils.anomaly.non_cyclical.slice import detect_windows


def _extract_all_features(
    df: pd.DataFrame,
    *,
    window_size_rows: int,
    channel_columns: Sequence[str],
    feature_names: Sequence[str],
    machine_id: str,
    machine_id_column: str,
) -> pd.DataFrame:
    """Slice into fixed windows, extract features per window, stack into a DF.

    Returns an empty DataFrame if no full windows fit in df (e.g. df is
    shorter than window_size_rows). Callers are responsible for handling
    that case.
    """
    rows = []
    for start, end in detect_windows(df, window_size_rows=window_size_rows):
        feats = extract_features(
            df.iloc[start:end],
            channel_columns=list(channel_columns),
            feature_names=list(feature_names),
        )
        feats[machine_id_column] = machine_id
        rows.append(feats)
    if not rows:
        return pd.DataFrame()
    return stack_features(rows)


def train_from_clean_and_anomaly(
    clean_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    *,
    machine_id: str = "inkjet_01",
    window_size_rows: int = 12800,
    train_ratio: float = 0.8,
    channel_columns: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
    machine_id_column: str = "machine_id",
) -> dict:
    """Train a non-cyclical Random Forest anomaly model from clean + anomaly
    recordings. Return a bundle dict ready for joblib.dump().

    Parameters
    ----------
    clean_df, anomaly_df : pd.DataFrame
        Two labelled recordings. Clean = normal operation; anomaly =
        misbehaving operation. Both should contain the channel columns
        the model uses (default: Acceleration 0/1/2 + AE).
    machine_id : str
        Identifier stored on each per-window feature row so the
        per-machine normaliser can group correctly.
    window_size_rows : int
        Number of raw sensor rows per analysis window. Default 12,800 =
        0.5 s at 25.6 kHz. Lower this for small test fixtures.
    train_ratio : float
        Fraction of each recording used for training. The rest is held
        out for evaluation by the caller (e.g. run.py's threshold sweep).
    channel_columns, feature_names : optional
        Override the defaults from features.py.

    Returns
    -------
    bundle : dict
        {model, normaliser, feature_columns, machine_id,
         machine_id_column, window_size_rows, channel_columns,
         trained_window_counts}
        Ready for joblib.dump(); consumed by non_cyclical.score().

    Raises
    ------
    ValueError
        If either recording produces zero windows after slicing
        (recording too short for window_size_rows).
    """
    channel_columns = list(channel_columns or DEFAULT_CHANNEL_COLUMNS)
    feature_names = list(feature_names or DEFAULT_FEATURE_NAMES)

    # Time-series split: preserve order, no shuffle.
    n_clean_train = int(len(clean_df) * train_ratio)
    n_anomaly_train = int(len(anomaly_df) * train_ratio)
    clean_train = clean_df.iloc[:n_clean_train]
    anomaly_train = anomaly_df.iloc[:n_anomaly_train]

    common = dict(
        window_size_rows=window_size_rows,
        channel_columns=channel_columns,
        feature_names=feature_names,
        machine_id=machine_id,
        machine_id_column=machine_id_column,
    )
    feat_clean_train = _extract_all_features(clean_train, **common)
    feat_anomaly_train = _extract_all_features(anomaly_train, **common)

    if feat_clean_train.empty or feat_anomaly_train.empty:
        raise ValueError(
            "train_from_clean_and_anomaly: no windows produced. "
            f"clean_train has {len(clean_train)} rows, "
            f"anomaly_train has {len(anomaly_train)} rows, "
            f"window_size_rows={window_size_rows}. "
            "Either provide more data or lower window_size_rows."
        )

    feature_cols = [c for c in feat_clean_train.columns if c != machine_id_column]

    # Per-machine z-score normalisation — fit on CLEAN training only.
    normaliser = PerMachineNormaliser()
    feat_clean_train_n = normaliser.fit_transform(
        feat_clean_train,
        machine_id_column=machine_id_column,
        feature_columns=feature_cols,
    )
    feat_anomaly_train_n = normaliser.transform(
        feat_anomaly_train,
        machine_id_column=machine_id_column,
    )

    X_train = pd.concat(
        [feat_clean_train_n[feature_cols], feat_anomaly_train_n[feature_cols]],
        ignore_index=True,
    ).to_numpy()
    y_train = np.array(
        [0] * len(feat_clean_train_n) + [1] * len(feat_anomaly_train_n)
    )

    model = RandomForestModel()
    model.fit(X_train, y_train)

    return {
        "model":              model,
        "normaliser":         normaliser,
        "feature_columns":    feature_cols,
        "machine_id":         machine_id,
        "machine_id_column":  machine_id_column,
        "window_size_rows":   window_size_rows,
        "channel_columns":    channel_columns,
        "trained_window_counts": {
            "clean":   int(len(feat_clean_train_n)),
            "anomaly": int(len(feat_anomaly_train_n)),
        },
    }
