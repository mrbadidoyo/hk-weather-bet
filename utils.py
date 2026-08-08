"""
Utility functions for HK Weather Prediction System.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Tuple, Dict
from pathlib import Path

def get_bucket_probs(temp_mean: float, temp_std: float,
                     bucket_defs: List[Tuple[str, float, float]]) -> Dict[str, float]:
    """
    Return probability for each bucket assuming Normal(temp_mean, temp_std).
    Bucket definition: (label, lower_bound, upper_bound) where bounds are inclusive/exclusive
    as appropriate; we treat them as continuous intervals.
    """
    probs = {}
    for label, lo, hi in bucket_defs:
        # Probability that temperature falls within [lo, hi)
        prob = norm.cdf(hi, loc=temp_mean, scale=temp_std) - \
               norm.cdf(lo, loc=temp_mean, scale=temp_std)
        # Clamp to [0,1] just in case
        probs[label] = max(0.0, min(1.0, prob))
    return probs


def load_recent_errors(days: int = 30) -> pd.DataFrame:
    """
    Load recent forecast errors from processed data if available.
    Expects a CSV file with columns: date, forecast_max, actual_max, forecast_min, actual_min
    Returns DataFrame with error columns.
    """
    processed_dir = Path(__file__).parent / "data" / "processed"
    # Try common file names
    candidates = ["forecast_errors.csv", "backtest_results.csv", "model_performance.csv"]
    for fname in candidates:
        fpath = processed_dir / fname
        if fpath.exists():
            try:
                df = pd.read_csv(fpath, parse_dates=["date"])
                # Keep most recent `days`
                if len(df) > days:
                    df = df.iloc[-days:]
                return df
            except Exception:
                continue
    return pd.DataFrame()


def compute_historical_std(days: int = 30) -> Tuple[float, float]:
    """
    Compute standard deviation of forecast errors for max and min temperatures
    using recent historical data.
    Returns (std_max, std_min). If insufficient data, returns (1.0, 0.8) as fallback.
    """
    df = load_recent_errors(days=days)
    if df.empty:
        # fallback defaults
        return 1.0, 0.8
    # Ensure required columns exist
    if {"forecast_max", "actual_max"}.issubset(df.columns):
        errors_max = df["actual_max"] - df["forecast_max"]
        std_max = float(errors_max.std(ddof=0))  # population std
    else:
        std_max = 1.0
    if {"forecast_min", "actual_min"}.issubset(df.columns):
        errors_min = df["actual_min"] - df["forecast_min"]
        std_min = float(errors_min.std(ddof=0))
    else:
        std_min = 0.8
    # Avoid zero std
    if std_max <= 0:
        std_max = 1.0
    if std_min <= 0:
        std_min = 0.8
    return std_max, std_min