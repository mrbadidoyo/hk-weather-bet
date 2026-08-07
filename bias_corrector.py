"""
Multi-Day Bias Corrector (v2) — Range-aware / Conditional
=========================================================
Learns from prediction errors and applies corrections that depend on
the predicted temperature range (instead of a single global shift).

This specifically targets the cold bias problem:
- High temp: systematic underprediction in 30-32°C and 34-36°C ranges
- Low temp: systematic underprediction in 28-29°C range
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from config import PROCESSED_DATA_DIR

BIAS_LOG = PROCESSED_DATA_DIR / "bias_corrections.jsonl"

# Temperature ranges used for conditional bias (aligned with common Polymarket buckets)
HIGH_RANGES = [
    ("<=29", -999, 30),
    ("30-32", 30, 33),
    ("33-35", 33, 36),
    ("36+", 36, 999),
]

LOW_RANGES = [
    ("<=25", -999, 26),
    ("26-27", 26, 28),
    ("28-29", 28, 30),
    ("30+", 30, 999),
]


def _get_range_label(temp: float, ranges: list) -> str:
    """Return the range label that contains the given temperature."""
    for label, lo, hi in ranges:
        if lo <= temp < hi:
            return label
    return ranges[-1][0]


def log_forecast_error(target_date, temp_type, predicted_mean, actual_temp):
    """
    Log a forecast error for bias tracking.

    Args:
        target_date: Date string (YYYY-MM-DD)
        temp_type: 'high' or 'low'
        predicted_mean: Model's predicted mean temperature
        actual_temp: Actual observed temperature
    """
    error = actual_temp - predicted_mean  # positive = model underpredicted

    ranges = HIGH_RANGES if temp_type == "high" else LOW_RANGES
    pred_range = _get_range_label(predicted_mean, ranges)
    actual_range = _get_range_label(actual_temp, ranges)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "target_date": str(target_date),
        "temp_type": temp_type,
        "predicted_mean": float(predicted_mean),
        "actual_temp": float(actual_temp),
        "error": float(error),
        "pred_range": pred_range,
        "actual_range": actual_range,
    }

    BIAS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(BIAS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _load_recent_errors(temp_type: str, lookback_days: int = 30) -> list:
    """Load recent error entries for a given temp_type."""
    if not BIAS_LOG.exists():
        return []

    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    errors = []

    try:
        lines = BIAS_LOG.read_text(encoding="utf-8").strip().split("\n")
    except Exception:
        return []

    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        if entry.get("temp_type") != temp_type:
            continue
        if entry.get("timestamp", "") < cutoff:
            continue
        errors.append(entry)

    errors.sort(key=lambda e: e.get("target_date", ""))
    return errors


def _ewma(values: list, alpha: float = 0.35) -> float:
    """Exponential weighted moving average (more weight to recent)."""
    if not values:
        return 0.0
    ewma = values[0]
    for v in values[1:]:
        ewma = alpha * v + (1 - alpha) * ewma
    return float(ewma)


def get_bias_correction(
    temp_type: str,
    predicted_mean: float = None,
    lookback_days: int = 30,
    min_samples_global: int = 5,
    min_samples_range: int = 3,
):
    """
    Calculate range-aware bias correction.

    Priority:
    1. If enough samples in the same predicted temperature range → use range-specific EWMA
    2. Else fall back to global EWMA
    3. Cap correction to avoid over-correction

    Args:
        temp_type: 'high' or 'low'
        predicted_mean: Current model prediction (used to select the range). 
                        If None, only global bias is returned.
        lookback_days: How many days of history to consider
        min_samples_global / min_samples_range: Minimum samples required

    Returns:
        dict with:
        - correction: °C to ADD to the prediction
        - source: 'range' | 'global' | 'none'
        - range_label: which temperature band was used (if any)
        - n_samples: samples used for the chosen correction
        - n_global: total recent samples
        - mean_error / ewma_error
        - confidence: HIGH / MEDIUM / LOW
    """
    errors = _load_recent_errors(temp_type, lookback_days)

    empty = {
        "correction": 0.0,
        "source": "none",
        "range_label": None,
        "n_samples": 0,
        "n_global": 0,
        "mean_error": 0.0,
        "ewma_error": 0.0,
        "confidence": "LOW",
    }

    if not errors:
        return empty

    raw_errors = [e["error"] for e in errors]
    global_mean = float(np.mean(raw_errors))
    global_ewma = _ewma(raw_errors)
    n_global = len(errors)

    # --- Try range-specific correction ---
    range_correction = None
    range_label = None
    n_range = 0
    range_ewma = 0.0
    range_mean = 0.0

    if predicted_mean is not None:
        ranges = HIGH_RANGES if temp_type == "high" else LOW_RANGES
        range_label = _get_range_label(predicted_mean, ranges)

        range_errors = [
            e["error"] for e in errors
            if e.get("pred_range") == range_label
        ]

        # Fallback: also accept entries that only have predicted_mean (older logs)
        if len(range_errors) < min_samples_range:
            range_errors = [
                e["error"] for e in errors
                if _get_range_label(e.get("predicted_mean", -999), ranges) == range_label
            ]

        n_range = len(range_errors)
        if n_range >= min_samples_range:
            range_mean = float(np.mean(range_errors))
            range_ewma = _ewma(range_errors)
            range_correction = range_ewma

    # --- Decide which correction to use ---
    if range_correction is not None:
        correction = range_correction
        source = "range"
        n_used = n_range
        mean_error = range_mean
        ewma_error = range_ewma
    elif n_global >= min_samples_global:
        correction = global_ewma
        source = "global"
        n_used = n_global
        mean_error = global_mean
        ewma_error = global_ewma
        range_label = None
    else:
        return empty

    # Cap to prevent over-correction (slightly tighter for range-specific)
    max_abs = 1.8 if source == "range" else 2.0
    correction = float(np.clip(correction, -max_abs, max_abs))

    # Confidence
    if n_used >= 8:
        confidence = "HIGH"
    elif n_used >= 4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "correction": round(correction, 2),
        "source": source,
        "range_label": range_label,
        "n_samples": n_used,
        "n_global": n_global,
        "mean_error": round(mean_error, 2),
        "ewma_error": round(ewma_error, 2),
        "confidence": confidence,
    }


def apply_bias_correction(predictions: dict, temp_type: str) -> float:
    """
    Apply range-aware bias correction to a prediction mean.

    Args:
        predictions: Dict that must contain either:
            - f"{temp_type}_mean"  (e.g. "high_mean")
            - or "mean"
        temp_type: 'high' or 'low'

    Returns:
        Corrected mean temperature (float) or original if no correction available.
    """
    key = f"{temp_type}_mean"
    original_mean = predictions.get(key)
    if original_mean is None:
        original_mean = predictions.get("mean")

    if original_mean is None:
        return original_mean

    bias = get_bias_correction(temp_type, predicted_mean=float(original_mean))
    corrected = float(original_mean) + bias["correction"]
    return round(corrected, 2)


def get_bias_report(lookback_days: int = 30) -> dict:
    """
    Generate a detailed bias report (global + per-range) for both high and low.
    """
    report = {}

    for temp_type in ["high", "low"]:
        errors = _load_recent_errors(temp_type, lookback_days)
        ranges = HIGH_RANGES if temp_type == "high" else LOW_RANGES

        # Global
        global_bias = get_bias_correction(temp_type, predicted_mean=None, lookback_days=lookback_days)

        # Per-range
        range_stats = {}
        for label, lo, hi in ranges:
            # Pick a representative mid-point for the range so get_bias_correction uses it
            mid = 25.0 if lo == -999 else (39.0 if hi == 999 else (lo + hi) / 2)
            b = get_bias_correction(temp_type, predicted_mean=mid, lookback_days=lookback_days)

            if b["source"] == "range":
                range_stats[label] = {
                    "correction": b["correction"],
                    "n_samples": b["n_samples"],
                    "mean_error": b["mean_error"],
                    "confidence": b["confidence"],
                    "direction": (
                        "underpredicts" if b["mean_error"] > 0.15
                        else "overpredicts" if b["mean_error"] < -0.15
                        else "calibrated"
                    ),
                }

        report[temp_type] = {
            "global": {
                "correction": global_bias["correction"],
                "n_samples": global_bias["n_samples"],
                "mean_error": global_bias["mean_error"],
                "confidence": global_bias["confidence"],
                "direction": (
                    "underpredicts" if global_bias["mean_error"] > 0.15
                    else "overpredicts" if global_bias["mean_error"] < -0.15
                    else "calibrated"
                ),
            },
            "by_range": range_stats,
        }

    return report


def format_bias_summary(lookback_days: int = 30) -> str:
    """Human-readable bias summary (global + range breakdown)."""
    report = get_bias_report(lookback_days)
    lines = [f"Bias Correction (Last {lookback_days} Days) — Range-Aware:"]

    for temp_type in ["high", "low"]:
        r = report[temp_type]
        g = r["global"]
        label = temp_type.upper()

        if g["n_samples"] == 0:
            lines.append(f"  {label}: No data yet")
            continue

        lines.append(
            f"  {label} GLOBAL: {g['direction']} by {abs(g['mean_error']):.2f}°C "
            f"(n={g['n_samples']}, corr={g['correction']:+.2f}°C, {g['confidence']})"
        )

        if r["by_range"]:
            for rng, stats in r["by_range"].items():
                lines.append(
                    f"    └ {rng}: {stats['direction']} "
                    f"(n={stats['n_samples']}, corr={stats['correction']:+.2f}°C, {stats['confidence']})"
                )
        else:
            lines.append("    └ (not enough per-range samples yet)")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Convenience helpers for backtest / live use
# ------------------------------------------------------------------

def correct_mean(temp_type: str, predicted_mean: float, lookback_days: int = 30) -> float:
    """One-liner: return bias-corrected mean."""
    bias = get_bias_correction(temp_type, predicted_mean=predicted_mean, lookback_days=lookback_days)
    return round(float(predicted_mean) + bias["correction"], 2)


def get_correction_detail(temp_type: str, predicted_mean: float) -> dict:
    """Return full correction info for logging / Telegram alerts."""
    return get_bias_correction(temp_type, predicted_mean=predicted_mean)
