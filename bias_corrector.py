"""
Multi-Day Bias Corrector — Learns from prediction errors over time.
Tracks rolling forecast errors and applies corrections to future predictions.
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from config import PROCESSED_DATA_DIR

BIAS_LOG = PROCESSED_DATA_DIR / "bias_corrections.jsonl"


def log_forecast_error(target_date, temp_type, predicted_mean, actual_temp):
    """
    Log a forecast error for bias tracking.
    
    Args:
        target_date: Date string (YYYY-MM-DD)
        temp_type: 'high' or 'low'
        predicted_mean: Model's predicted mean temperature
        actual_temp: Actual observed temperature
    """
    error = actual_temp - predicted_mean
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "target_date": target_date,
        "temp_type": temp_type,
        "predicted_mean": predicted_mean,
        "actual_temp": actual_temp,
        "error": error,
    }
    
    with open(BIAS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_bias_correction(temp_type, lookback_days=14):
    """
    Calculate bias correction from recent forecast errors.
    
    Uses exponential weighted moving average (EWMA) to give more
    weight to recent errors. If model consistently overpredicts
    by 0.5°C, correction will shift predictions down.
    
    Args:
        temp_type: 'high' or 'low'
        lookback_days: How many days of errors to consider
    
    Returns:
        dict with:
        - correction: Temperature adjustment to add to predictions (°C)
        - n_samples: Number of recent errors used
        - mean_error: Raw mean error (positive = model underpredicts)
        - ewma_error: EWMA of errors
        - confidence: 'HIGH' if >7 samples, 'MEDIUM' if >3, 'LOW' otherwise
    """
    if not BIAS_LOG.exists():
        return {"correction": 0.0, "n_samples": 0, "mean_error": 0.0, 
                "ewma_error": 0.0, "confidence": "LOW"}
    
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    
    errors = []
    for line in BIAS_LOG.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        entry = json.loads(line)
        if (entry["temp_type"] == temp_type and 
            entry["timestamp"] >= cutoff):
            errors.append(entry)
    
    if not errors:
        return {"correction": 0.0, "n_samples": 0, "mean_error": 0.0,
                "ewma_error": 0.0, "confidence": "LOW"}
    
    # Sort by date
    errors.sort(key=lambda e: e["target_date"])
    
    # Calculate mean error (positive = model underpredicts, negative = overpredicts)
    raw_errors = [e["error"] for e in errors]
    mean_error = np.mean(raw_errors)
    
    # EWMA: more weight to recent errors
    alpha = 0.3  # Smoothing factor
    ewma = raw_errors[0]
    for err in raw_errors[1:]:
        ewma = alpha * err + (1 - alpha) * ewma
    
    # Correction = the error direction (add to prediction to fix bias)
    correction = ewma
    
    # Confidence based on sample size
    n = len(errors)
    if n >= 7:
        confidence = "HIGH"
    elif n >= 4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    # Cap correction at ±2°C to avoid overcorrection
    correction = np.clip(correction, -2.0, 2.0)
    
    return {
        "correction": round(float(correction), 2),
        "n_samples": n,
        "mean_error": round(float(mean_error), 2),
        "ewma_error": round(float(ewma), 2),
        "confidence": confidence,
    }


def apply_bias_correction(predictions, temp_type):
    """
    Apply bias correction to prediction mean.
    
    Args:
        predictions: Dict with prediction data
        temp_type: 'high' or 'low'
    
    Returns:
        Corrected mean temperature
    """
    key = f"{temp_type}_mean"
    original_mean = predictions.get(key)
    
    if original_mean is None:
        return original_mean
    
    bias = get_bias_correction(temp_type)
    corrected_mean = original_mean + bias["correction"]
    
    return round(corrected_mean, 2)


def get_bias_report():
    """
    Generate a human-readable bias report for both high and low temps.
    
    Returns dict with 'high' and 'low' keys containing bias stats.
    """
    report = {}
    
    for temp_type in ["high", "low"]:
        bias = get_bias_correction(temp_type, lookback_days=30)
        report[temp_type] = {
            "correction": bias["correction"],
            "mean_error": bias["mean_error"],
            "ewma_error": bias["ewma_error"],
            "n_samples": bias["n_samples"],
            "confidence": bias["confidence"],
            "direction": "underpredicts" if bias["mean_error"] > 0.2 else 
                         "overpredicts" if bias["mean_error"] < -0.2 else "calibrated",
        }
    
    return report


def format_bias_summary():
    """Format bias report into a readable summary string."""
    report = get_bias_report()
    
    lines = ["Bias Correction (Last 30 Days):"]
    for temp_type in ["high", "low"]:
        b = report[temp_type]
        label = temp_type.upper()
        if b["n_samples"] == 0:
            lines.append(f"  {label}: No data yet")
        else:
            direction = b["direction"]
            lines.append(
                f"  {label}: {direction} by {abs(b['mean_error']):.2f}°C "
                f"(n={b['n_samples']}, correction={b['correction']:+.2f}°C, "
                f"confidence={b['confidence']})"
            )
    
    return "\n".join(lines)
