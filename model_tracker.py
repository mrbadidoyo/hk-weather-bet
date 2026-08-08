"""
Model Performance Tracker — Tracks prediction accuracy over time.
Logs predictions vs actual outcomes for model evaluation.
"""
# Revision note: tracking behavior preserved; revision marker only.
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from config import PROCESSED_DATA_DIR

PERFORMANCE_LOG = PROCESSED_DATA_DIR / "model_performance.jsonl"


def log_prediction(target_date, temp_type, bucket_probs, recommended_bets, actual_temp=None, provisional_result=None, provisional_market_price=None):
    """
    Log a prediction for later evaluation.
    
    Args:
        target_date: Date string (YYYY-MM-DD)
        temp_type: 'high' or 'low'
        bucket_probs: Dict of {bucket: probability}
        recommended_bets: Dict with 'main' and 'lottery' bets
        actual_temp: Actual temperature (if resolved)
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "target_date": target_date,
        "temp_type": temp_type,
        "bucket_probs": bucket_probs,
        "recommended_bets": recommended_bets,
        "actual_temp": actual_temp,
        "resolved": actual_temp is not None,
        "provisional_result": provisional_result,
        "provisional_market_price": provisional_market_price,
        "provisional_source": "polymarket_highest_price" if provisional_result is not None else None,
    }
    
    with open(PERFORMANCE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def upsert_prediction(target_date, temp_type, bucket_probs, recommended_bets,
                      provisional_result=None, provisional_market_price=None):
    """Create or refresh the single unresolved prediction for a date/type."""
    if not PERFORMANCE_LOG.exists():
        log_prediction(target_date, temp_type, bucket_probs, recommended_bets,
                       provisional_result=provisional_result,
                       provisional_market_price=provisional_market_price)
        return

    raw = PERFORMANCE_LOG.read_text(encoding="utf-8").strip()
    lines = raw.split("\n") if raw else []
    updated = []
    found = False
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if (not found and entry.get("target_date") == target_date
                and entry.get("temp_type") == temp_type
                and not entry.get("resolved", False)):
            entry["timestamp"] = datetime.now().isoformat()
            entry["bucket_probs"] = bucket_probs
            entry["recommended_bets"] = recommended_bets
            entry["provisional_result"] = provisional_result
            entry["provisional_market_price"] = provisional_market_price
            entry["provisional_source"] = ("polymarket_highest_price"
                                            if provisional_result is not None else None)
            found = True
        updated.append(json.dumps(entry))

    if not found:
        updated.append(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "target_date": target_date,
            "temp_type": temp_type,
            "bucket_probs": bucket_probs,
            "recommended_bets": recommended_bets,
            "actual_temp": None,
            "resolved": False,
            "provisional_result": provisional_result,
            "provisional_market_price": provisional_market_price,
            "provisional_source": ("polymarket_highest_price"
                                    if provisional_result is not None else None),
        }))

    PERFORMANCE_LOG.write_text("\n".join(updated) + "\n", encoding="utf-8")


def update_prediction(target_date, temp_type, actual_temp):
    """Update a logged prediction with actual outcome."""
    if not PERFORMANCE_LOG.exists():
        return
    
    lines = PERFORMANCE_LOG.read_text(encoding="utf-8").strip().split("\n")
    updated = []
    
    for line in lines:
        entry = json.loads(line)
        if (entry["target_date"] == target_date and 
            entry["temp_type"] == temp_type and 
            not entry["resolved"]):
            entry["actual_temp"] = actual_temp
            entry["resolved"] = True
        updated.append(json.dumps(entry))
    
    PERFORMANCE_LOG.write_text("\n".join(updated) + "\n", encoding="utf-8")


def get_performance_stats(days=30):
    """
    Calculate performance statistics over the last N days.
    
    Returns dict with:
    - brier_score: Lower is better (0 = perfect, 0.25 = no skill)
    - win_rate: % of recommended bets that won
    - roi: Return on investment
    - calibration: How well probabilities match outcomes
    """
    if not PERFORMANCE_LOG.exists():
        return None
    
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    
    entries = []
    for line in PERFORMANCE_LOG.read_text(encoding="utf-8").strip().split("\n"):
        entry = json.loads(line)
        if entry["timestamp"] >= cutoff and entry["resolved"]:
            entries.append(entry)
    
    if not entries:
        return None
    
    # Calculate metrics
    brier_scores = []
    wins = 0
    total_bets = 0
    total_staked = 0
    total_payout = 0
    
    for entry in entries:
        actual = entry["actual_temp"]
        probs = entry["bucket_probs"]
        bets = entry["recommended_bets"]
        
        # Brier score for all buckets
        for bucket, prob in probs.items():
            # Determine if this bucket won
            # Need to parse bucket to check if actual temp falls in range
            won = check_bucket_won(bucket, actual, entry["temp_type"])
            brier_scores.append((prob - won) ** 2)
        
        # Check recommended bets
        for bet_type in ["main", "lottery"]:
            if bets.get(bet_type):
                bet = bets[bet_type]
                total_bets += 1
                won = check_bucket_won(bet["bucket"], actual, entry["temp_type"])
                if won:
                    wins += 1
                    # Payout = 1 / market_price (simplified)
                    total_payout += 1.0
                    total_staked += bet["market"]
                else:
                    total_staked += bet["market"]
    
    avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else None
    win_rate = wins / total_bets if total_bets > 0 else None
    roi = (total_payout - total_staked) / total_staked if total_staked > 0 else None
    
    return {
        "brier_score": avg_brier,
        "win_rate": win_rate,
        "roi": roi,
        "total_bets": total_bets,
        "resolved_days": len(entries),
        "period_days": days,
    }


def check_bucket_won(bucket, actual_temp, temp_type):
    """Check if a bucket contains the actual temperature."""
    import re
    
    # Parse bucket like "34°C", "27°C or below", "37°C or higher"
    match = re.search(r'(\d+)', bucket)
    if not match:
        return False
    
    bucket_temp = int(match.group(1))
    
    if "below" in bucket.lower():
        return actual_temp < bucket_temp
    elif "higher" in bucket.lower():
        return actual_temp >= bucket_temp
    else:
        # Single degree bucket: [temp, temp+1)
        return bucket_temp <= actual_temp < bucket_temp + 1


def format_performance_report():
    """Format performance stats into a readable report."""
    stats = get_performance_stats(30)
    
    if not stats:
        return "No performance data available yet."
    
    lines = [
        "📊 *Model Performance (Last 30 Days)*",
        "",
        f"Brier Score: {stats['brier_score']:.3f}" if stats['brier_score'] else "Brier Score: N/A",
        f"Win Rate: {stats['win_rate']:.1%}" if stats['win_rate'] else "Win Rate: N/A",
        f"ROI: {stats['roi']:+.1%}" if stats['roi'] is not None else "ROI: N/A",
        f"Total Bets: {stats['total_bets']}",
        f"Resolved Days: {stats['resolved_days']}",
        "",
        "_Brier Score: 0=perfect, 0.25=no skill_"
    ]
    
    return "\n".join(lines)
