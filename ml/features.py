import numpy as np
import pandas as pd
from typing import Optional, List
from config import WINDOW_SIZE

FEATURE_NAMES: List[str] = [
    "temp_mean",
    "temp_std",
    "temp_slope",
    "vib_mean",
    "vib_std",
    "vib_slope",
    "vib_max",
    "power_mean",
    "power_std",
    "power_slope"
]

def _calc_slope(series: np.ndarray) -> float:
    """Computes linear fit slope per reading using np.polyfit."""
    n = len(series)
    if n < 2:
        return 0.0
    x = np.arange(n)
    try:
        slope, _ = np.polyfit(x, series, 1)
        return float(slope)
    except Exception:
        return 0.0

def extract_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extracts 10 specific statistical and trend features from a machine's recent readings:
    - temp_mean, temp_std, temp_slope
    - vib_mean, vib_std, vib_slope, vib_max
    - power_mean, power_std, power_slope
    
    If fewer than 5 rows, returns None.
    If between 5 and WINDOW_SIZE rows, pads by repeating the first row until WINDOW_SIZE.
    Returns a single-row DataFrame with FEATURE_NAMES columns.
    """
    if df is None or len(df) < 5:
        return None

    df_subset = df[["temperature", "vibration", "power"]].copy()

    # Handle windows shorter than WINDOW_SIZE (pad repeating the first value)
    if len(df_subset) < WINDOW_SIZE:
        pad_count = WINDOW_SIZE - len(df_subset)
        first_row = df_subset.iloc[[0]]
        padding = pd.concat([first_row] * pad_count, ignore_index=True)
        df_subset = pd.concat([padding, df_subset], ignore_index=True)
    elif len(df_subset) > WINDOW_SIZE:
        df_subset = df_subset.iloc[-WINDOW_SIZE:].reset_index(drop=True)

    t = df_subset["temperature"].values.astype(float)
    v = df_subset["vibration"].values.astype(float)
    p = df_subset["power"].values.astype(float)

    feature_dict = {
        "temp_mean": float(np.mean(t)),
        "temp_std": float(np.std(t, ddof=1) if len(t) > 1 else 0.0),
        "temp_slope": _calc_slope(t),
        "vib_mean": float(np.mean(v)),
        "vib_std": float(np.std(v, ddof=1) if len(v) > 1 else 0.0),
        "vib_slope": _calc_slope(v),
        "vib_max": float(np.max(v)),
        "power_mean": float(np.mean(p)),
        "power_std": float(np.std(p, ddof=1) if len(p) > 1 else 0.0),
        "power_slope": _calc_slope(p)
    }

    return pd.DataFrame([feature_dict], columns=FEATURE_NAMES)
