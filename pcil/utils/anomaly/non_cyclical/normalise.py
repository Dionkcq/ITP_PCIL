"""
Shared utility — Per-machine z-score normalisation
====================================================
Fits a baseline mean and std per machine from clean training data,
then applies the same fitted baseline when scoring new data from
the same machine.
"""
import numpy as np
import pandas as pd


class PerMachineNormaliser:
    # Z-score normaliser that maintains a separate mean/std per machine ID.

    def __init__(self):
        self.stats_: dict[str, dict[str, np.ndarray]] = {}
        self.feature_columns_: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        *,
        machine_id_column: str,
        feature_columns: list[str],
    ) -> "PerMachineNormaliser":
        # Compute mean and std for each feature, separately per machine.
        self.feature_columns_ = list(feature_columns)
        self.stats_ = {}

        for machine_id, group in df.groupby(machine_id_column):
            X = group[feature_columns].to_numpy(dtype=float)
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            # Avoid division by zero for constant features
            std = np.where(std == 0, 1.0, std)
            self.stats_[machine_id] = {"mean": mean, "std": std}

        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        machine_id_column: str,
    ) -> pd.DataFrame:
        # Apply the fitted z-score transform to a new DataFrame.

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