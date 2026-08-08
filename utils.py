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


def bucket_probs_from_quantiles(bucket_defs: List[Tuple[str, float, float]],
                                q10: float, q25: float, q50: float,
                                q75: float, q90: float) -> Dict[str, float]:
    """
    Compute bucket probabilities using a piecewise linear CDF defined by
    the quantiles (10th, 25th, 50th, 75th, 90th). Tails beyond q10 and q90
    are assumed to have 0 and 1 probability respectively.
    """
    # Build CDF points: (x, p)
    points = [
        (-np.inf, 0.0),
        (q10, 0.10),
        (q25, 0.25),
        (q50, 0.50),
        (q75, 0.75),
        (q90, 0.90),
        (np.inf, 1.0),
    ]

    def cdf(x: float) -> float:
        # Linear interpolation between points
        for i in range(len(points) - 1):
            x_lo, p_lo = points[i]
            x_hi, p_hi = points[i + 1]
            if x_lo <= x <= x_hi:
                if x_hi == np.inf:
                    return p_hi
                if x_lo == -np.inf:
                    return p_lo
                # Linear interpolation
                t = (x - x_lo) / (x_hi - x_lo)
                return p_lo + t * (p_hi - p_lo)
        return 1.0 if x >= points[-1][0] else 0.0

    probs = {}
    for label, lo, hi in bucket_defs:
        prob = cdf(hi) - cdf(lo)
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


def compute_historical_quantiles(days: int = 30) -> Tuple[Tuple[float, float, float, float, float],
                                                            Tuple[float, float, float, float, float]]:
    """
    Compute quantiles (10th, 25th, 50th, 75th, 90th) of forecast errors for max and min temperatures
    using recent historical data.
    Returns ((q10_max, q25_max, q50_max, q75_max, q90_max), (q10_min, q25_min, q50_min, q75_min, q90_min)).
    If insufficient data, returns fallback values based on normal distribution with std=1.0/0.8.
    """
    df = load_recent_errors(days=days)
    if df.empty or not ({"forecast_max", "actual_max"}.issubset(df.columns) and
                        {"forecast_min", "actual_min"}.issubset(df.columns)):
        # Fallback to normal distribution quantiles with std=1.0 for max, 0.8 for min
        from scipy.stats import norm
        max_q = norm.ppf([0.10, 0.25, 0.50, 0.75, 0.90], loc=0, scale=1.0)
        min_q = norm.ppf([0.10, 0.25, 0.50, 0.75, 0.90], loc=0, scale=0.8)
        return tuple(max_q), tuple(min_q)

    # Calculate errors
    errors_max = df["actual_max"] - df["forecast_max"]
    errors_min = df["actual_min"] - df["forecast_min"]

    # Compute quantiles
    max_q = np.quantile(errors_max, [0.10, 0.25, 0.50, 0.75, 0.90])
    min_q = np.quantile(errors_min, [0.10, 0.25, 0.50, 0.75, 0.90])

    return tuple(max_q), tuple(min_q)