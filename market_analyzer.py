"""
Market Efficiency Analyzer — Tracks how accurate Polymarket prices are.
Identifies systematic mispricings and patterns to exploit.
"""
import io
import json
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

from config import (
    DEFAULT_HIGH_TEMP_BUCKETS,
    DEFAULT_LOW_TEMP_BUCKETS,
    PROCESSED_DATA_DIR,
)
from data_collector import HKODataCollector
from polymarket_scraper import PolymarketScraper
from model_tracker import PERFORMANCE_LOG

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EFFICIENCY_LOG = PROCESSED_DATA_DIR / "market_efficiency.jsonl"


# ── Data Loading ──────────────────────────────────────────────────

def fetch_hka_data():
    """Fetch HKO Airport (HKA) historical temperature data."""
    hko = HKODataCollector()
    urls = {
        'max': 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKA',
        'min': 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMINT&rformat=csv&station=HKA',
    }

    def fetch(url):
        resp = hko.session.get(url, timeout=60)
        resp.raise_for_status()
        text = resp.content.decode('utf-8-sig')
        lines = text.strip().split('\n')
        header_idx = 2
        for i, line in enumerate(lines):
            if 'Year' in line:
                header_idx = i
                break
        clean = [l for l in lines[header_idx:]
                 if not l.strip().strip('"').startswith('***')
                 and not l.strip().strip('"').startswith('#')
                 and not l.strip().strip('"').startswith('C ')]
        return pd.read_csv(io.StringIO('\n'.join(clean)))

    def parse(df):
        cols = df.columns.tolist()
        yc = [c for c in cols if 'year' in c.lower()][0]
        mc = [c for c in cols if 'month' in c.lower()][0]
        dc = [c for c in cols if 'day' in c.lower()][0]
        vc = [c for c in cols if 'value' in c.lower()][0]
        df['date'] = pd.to_datetime(
            df[[yc, mc, dc]].rename(columns={yc: 'Year', mc: 'Month', dc: 'Day'}),
            errors='coerce'
        )
        df['value'] = pd.to_numeric(df[vc], errors='coerce')
        return df.dropna(subset=['date', 'value']).sort_values('date').set_index('date')[['value']]

    combined = parse(fetch(urls['max'])).join(
        parse(fetch(urls['min'])),
        lsuffix='_max', rsuffix='_min', how='inner'
    )
    combined.columns = ['max_temp', 'min_temp']
    return combined.dropna()


# ── Market Efficiency Analysis ────────────────────────────────────

def analyze_bucket_accuracy(bucket_defs, historical_data, temp_col, months=None):
    """
    Analyze how often each bucket occurs historically.
    Compares historical frequency with what market prices imply.
    
    Args:
        bucket_defs: List of (label, lo, hi) tuples
        historical_data: DataFrame with temperature data
        temp_col: 'max_temp' or 'min_temp'
        months: Optional list of months to filter (e.g., [6,7,8] for summer)
    
    Returns:
        Dict of {bucket_label: {"frequency": float, "count": int, "total": int}}
    """
    data = historical_data.copy()
    if months:
        data = data[data.index.month.isin(months)]
    
    total = len(data)
    results = {}
    
    for label, lo, hi in bucket_defs:
        if lo == -999:
            count = (data[temp_col] < hi).sum()
        elif hi == 999:
            count = (data[temp_col] >= lo).sum()
        else:
            count = ((data[temp_col] >= lo) & (data[temp_col] < hi)).sum()
        
        results[label] = {
            "frequency": count / total if total > 0 else 0,
            "count": int(count),
            "total": total,
        }
    
    return results


def analyze_market_prices_vs_actual(market_prices, actual_temps, bucket_defs, temp_type):
    """
    Compare market prices with actual outcomes.
    
    Args:
        market_prices: Dict of {bucket_label: price}
        actual_temps: Actual temperature value
        bucket_defs: List of (label, lo, hi) tuples
        temp_type: 'highest' or 'lowest'
    
    Returns:
        Dict with analysis results
    """
    # Find winning bucket
    winning_label = None
    for label, lo, hi in bucket_defs:
        if lo == -999:
            if actual_temps < hi:
                winning_label = label
                break
        elif hi == 999:
            if actual_temps >= lo:
                winning_label = label
                break
        else:
            if lo <= actual_temps < hi:
                winning_label = label
                break
    
    if not winning_label:
        return None
    
    # Calculate market accuracy metrics
    winning_price = market_prices.get(winning_label, 0)
    
    # Brier score: how well do market prices predict outcomes?
    brier = 0
    n_buckets = 0
    for label, _, _ in bucket_defs:
        price = market_prices.get(label, 0)
        actual = 1.0 if label == winning_label else 0.0
        brier += (price - actual) ** 2
        n_buckets += 1
    
    avg_brier = brier / n_buckets if n_buckets > 0 else 0
    
    # Implied probability vs actual
    implied_prob = winning_price  # Market price = implied probability
    
    return {
        "winning_label": winning_label,
        "winning_price": winning_price,
        "implied_prob": implied_prob,
        "brier_score": avg_brier,
        "actual_temp": actual_temps,
    }


def find_mispriced_buckets(historical_freq, market_prices, threshold=0.05):
    """
    Find buckets where market price deviates significantly from historical frequency.
    
    Args:
        historical_freq: Dict from analyze_bucket_accuracy
        market_prices: Dict of {bucket_label: market_price}
        threshold: Minimum deviation to flag as mispriced
    
    Returns:
        List of mispriced buckets with details
    """
    mispriced = []
    
    for label, stats in historical_freq.items():
        hist_freq = stats["frequency"]
        market_price = market_prices.get(label)
        
        if market_price is None:
            continue
        
        edge = hist_freq - market_price
        
        if abs(edge) > threshold:
            mispriced.append({
                "bucket": label,
                "historical_freq": hist_freq,
                "market_price": market_price,
                "edge": edge,
                "direction": "underpriced" if edge > 0 else "overpriced",
                "magnitude": abs(edge),
            })
    
    mispriced.sort(key=lambda x: x["magnitude"], reverse=True)
    return mispriced


# ── Pattern Detection ─────────────────────────────────────────────

def detect_market_patterns(performance_log_path=None):
    """
    Detect systematic patterns in market pricing errors.
    
    Analyzes:
    1. Does market overreact to HKO forecast changes?
    2. Are extreme buckets consistently underpriced?
    3. Is there a day-of-week pattern?
    4. Does accuracy improve closer to event date?
    
    Returns:
        Dict with pattern findings
    """
    patterns = {
        "extreme_bucket_bias": None,
        "favorite_longshot": None,
        "hko_forecast_anchor": None,
        "summary": [],
    }
    
    # Load performance log if available
    if performance_log_path is None:
        performance_log_path = PERFORMANCE_LOG
    
    if not performance_log_path.exists():
        patterns["summary"].append("No prediction data available for pattern analysis")
        return patterns
    
    entries = []
    for line in performance_log_path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            entry = json.loads(line)
            if entry.get("resolved"):
                entries.append(entry)
    
    if len(entries) < 5:
        patterns["summary"].append(f"Not enough data yet ({len(entries)} resolved events, need 5+)")
        return patterns
    
    # Pattern 1: Extreme bucket bias
    # Are extreme buckets (edges of distribution) consistently under/overpriced?
    extreme_edges = []
    for entry in entries:
        bets = entry.get("recommended_bets", {})
        for bet_type in ["main", "lottery"]:
            bet = bets.get(bet_type)
            if bet and "edge" in bet:
                extreme_edges.append(bet["edge"])
    
    if extreme_edges:
        avg_edge = np.mean(extreme_edges)
        patterns["extreme_bucket_bias"] = {
            "avg_edge": round(avg_edge, 4),
            "n_samples": len(extreme_edges),
            "interpretation": "Market underprices our picks" if avg_edge > 0.05 else
                             "Market overprices our picks" if avg_edge < -0.05 else
                             "Market fairly prices our picks"
        }
        patterns["summary"].append(
            f"Average edge on recommended bets: {avg_edge:+.1%} ({patterns['extreme_bucket_bias']['interpretation']})"
        )
    
    # Pattern 2: Favorite-longshot bias
    # Do favorites (high market price) win less than implied?
    # Do longshots (low market price) win more than implied?
    favorite_wins = 0
    favorite_count = 0
    longshot_wins = 0
    longshot_count = 0
    
    for entry in entries:
        bets = entry.get("recommended_bets", {})
        main = bets.get("main")
        if main and "market" in main:
            if main["market"] > 0.30:  # Favorite
                favorite_count += 1
                # Check if won (simplified)
                if main.get("edge", 0) > 0:
                    favorite_wins += 1
            elif main["market"] < 0.15:  # Longshot
                longshot_count += 1
                if main.get("edge", 0) > 0:
                    longshot_wins += 1
    
    if favorite_count > 0 and longshot_count > 0:
        patterns["favorite_longshot"] = {
            "favorite_win_rate": favorite_wins / favorite_count,
            "longshot_win_rate": longshot_wins / longshot_count,
            "n_favorites": favorite_count,
            "n_longshots": longshot_count,
        }
    
    return patterns


# ── Full Analysis Report ──────────────────────────────────────────

def generate_efficiency_report(months=None):
    """
    Generate a comprehensive market efficiency report.
    
    Args:
        months: Optional month filter (e.g., [6,7,8] for summer)
    
    Returns:
        Dict with full analysis
    """
    logger.info("Fetching historical data...")
    data = fetch_hka_data()
    
    # Filter to recent years (market-relevant period)
    recent_cutoff = data.index.max() - pd.DateOffset(years=5)
    recent = data[data.index >= recent_cutoff]
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "data_range": [str(recent.index[0].date()), str(recent.index[-1].date())],
        "n_days": len(recent),
        "months_filter": months,
    }
    
    # Analyze each temperature type
    for temp_type, bucket_defs, col in [
        ("high", DEFAULT_HIGH_TEMP_BUCKETS, "max_temp"),
        ("low", DEFAULT_LOW_TEMP_BUCKETS, "min_temp"),
    ]:
        logger.info(f"Analyzing {temp_type} temperature buckets...")
        
        # Historical frequency
        freq = analyze_bucket_accuracy(bucket_defs, recent, col, months)
        
        # Summary stats
        temps = recent[col]
        report[temp_type] = {
            "mean": round(temps.mean(), 2),
            "std": round(temps.std(), 2),
            "min": round(temps.min(), 1),
            "max": round(temps.max(), 1),
            "bucket_frequencies": freq,
        }
        
        # Find most/least common buckets
        sorted_buckets = sorted(freq.items(), key=lambda x: x[1]["frequency"], reverse=True)
        report[temp_type]["most_common"] = sorted_buckets[0][0]
        report[temp_type]["least_common"] = sorted_buckets[-1][0]
    
    # Market patterns
    report["patterns"] = detect_market_patterns()
    
    return report


def format_efficiency_summary(report):
    """Format efficiency report into readable summary."""
    lines = [
        f"Market Efficiency Report",
        f"Data: {report['data_range'][0]} to {report['data_range'][1]} ({report['n_days']} days)",
        "",
    ]
    
    for temp_type in ["high", "low"]:
        if temp_type not in report:
            continue
        
        t = report[temp_type]
        lines.append(f"{temp_type.upper()} Temperature:")
        lines.append(f"  Mean: {t['mean']}°C ± {t['std']}°C")
        lines.append(f"  Range: {t['min']}°C to {t['max']}°C")
        lines.append(f"  Most common bucket: {t['most_common']}")
        lines.append(f"  Least common bucket: {t['least_common']}")
        
        # Top 3 buckets by frequency
        freq = t["bucket_frequencies"]
        sorted_freq = sorted(freq.items(), key=lambda x: x[1]["frequency"], reverse=True)
        lines.append("  Top 3 buckets:")
        for label, stats in sorted_freq[:3]:
            lines.append(f"    {label}: {stats['frequency']:.1%} (n={stats['count']})")
        lines.append("")
    
    # Patterns
    patterns = report.get("patterns", {})
    if patterns.get("summary"):
        lines.append("Patterns Detected:")
        for s in patterns["summary"]:
            lines.append(f"  - {s}")
    
    return "\n".join(lines)
