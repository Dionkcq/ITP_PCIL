import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

# np.trapz was removed in NumPy 2.0 (renamed np.trapezoid). Resolve once
# here so the registry lambda works on both 1.x and 2.x.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
 
# ---------------------------------------------------------------------------
# Default channel columns for the acoustic sensor CSV.
# Override at call-time for different sensor configurations.
# ---------------------------------------------------------------------------
CHANNEL_COLUMNS: list[str] = [
    "Acceleration 0 (g)",
    "Acceleration 1 (g)",
    "Acceleration 2 (g)",
    "AE (V) (V)",
]
 
# ---------------------------------------------------------------------------
# Feature registry — add new features here without touching extract_features.
# ---------------------------------------------------------------------------
FEATURE_REGISTRY: dict[str, callable] = {
    # Core vibration/acoustic features
    "rms": lambda x: float(np.sqrt(np.mean(x ** 2))),
    "peak": lambda x: float(np.max(np.abs(x))),
    "std": lambda x: float(np.std(x)),
    "kurtosis": lambda x: float(scipy_kurtosis(x, fisher=True, bias=True)),
    "crest_factor": lambda x: float(
        np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-9)
    ),
    # Additional features for slow-moving signals (temperature, pressure)
    "mean": lambda x: float(np.mean(x)),
    "gradient_mean": lambda x: float(np.mean(np.abs(np.diff(x)))),
    "integrated_area": lambda x: float(_trapezoid(np.abs(x))),
    "fft_band_low":  lambda x: float(np.sum(np.abs(np.fft.rfft(x))[:len(x)//16])),   # 0–800 Hz
    "fft_band_mid":  lambda x: float(np.sum(np.abs(np.fft.rfft(x))[len(x)//16:len(x)//4])),  # 800–3.2 kHz
    "fft_band_high": lambda x: float(np.sum(np.abs(np.fft.rfft(x))[len(x)//4:])),     # 3.2 kHz+
}
 
# Default feature set — optimised for vibration/acoustic signals.
# For temperature or pressure sensors, consider ["mean", "std", "gradient_mean"].
DEFAULT_FEATURE_NAMES: list[str] = [
    "rms",
    "peak",
    "std",
    "kurtosis",
    "crest_factor",
    "fft_band_low",
    "fft_band_mid",
    "fft_band_high"
]
 
 
def _col_to_prefix(col: str) -> str:
    # Convert a column name to a short, safe prefix for feature naming.
    
    col = col.lower()
    col = col.replace("acceleration", "accel")
    col = col.replace("ae (v)", "ae")
    # Remove unit suffixes and parentheses
    for ch in "()":
        col = col.replace(ch, "")
    # Collapse whitespace to underscore
    col = "_".join(col.split())
    return col
 
 
def extract_features(
    window_df: pd.DataFrame,
    *,
    channel_columns: list[str] = CHANNEL_COLUMNS,
    feature_names: list[str] = DEFAULT_FEATURE_NAMES,) -> dict[str, float]:

    unknown = set(feature_names) - set(FEATURE_REGISTRY)
    if unknown:
        raise ValueError(
            f"Unknown feature(s): {unknown}. "
            f"Available: {list(FEATURE_REGISTRY)}"
        )
 
    features: dict[str, float] = {}
 
    for col in channel_columns:
        if col not in window_df.columns:
            raise KeyError(
                f"Column '{col}' not found in window_df. "
                f"Available columns: {list(window_df.columns)}"
            )
        prefix = _col_to_prefix(col)
        vals = window_df[col].to_numpy(dtype=float)
 
        if len(vals) == 0:
            raise ValueError(f"Empty window passed to extract_features (column '{col}').")
 
        for feat_name in feature_names:
            features[f"{prefix}_{feat_name}"] = FEATURE_REGISTRY[feat_name](vals)
 
    return features
 
 
def stack_features(rows: list[dict]) -> pd.DataFrame:
    # Convert a list of feature dicts (one per window) into a DataFrame.

    if not rows:
        raise ValueError("No feature rows to stack — is the input DataFrame empty?")
    return pd.DataFrame(rows)