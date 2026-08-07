"""
Polymarket Strategy Module
Converts temperature predictions into bucket probabilities and identifies +EV bets
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from config import DEFAULT_HIGH_TEMP_BUCKETS, DEFAULT_LOW_TEMP_BUCKETS

logger = logging.getLogger(__name__)


@dataclass
class Bucket:
    """A Polymarket temperature bucket"""
    label: str
    lower: float  # inclusive
    upper: float  # exclusive (except for open-ended)
    is_open_lower: bool = False  # True for "<X" buckets
    is_open_upper: bool = False  # True for "X+" buckets

    def contains(self, value: float) -> bool:
        if self.is_open_lower:
            return value < self.upper
        elif self.is_open_upper:
            return value >= self.lower
        else:
            return self.lower <= value < self.upper


@dataclass
class MarketBet:
    """A potential bet on a Polymarket weather market"""
    bucket_label: str
    model_prob: float
    market_price: float  # 0-1, price of YES share
    implied_prob: float  # market's implied probability
    edge: float  # model_prob - market_price
    ev: float  # expected value per $1 bet
    kelly_fraction: float  # Kelly criterion fraction
    confidence: float  # model confidence (0-1)
    recommendation: str  # BUY / SKIP / SELL


@dataclass
class MarketAnalysis:
    """Complete analysis of a weather market"""
    date: str
    market_type: str  # "high" or "low"
    predicted_mean: float
    predicted_std: float
    buckets: list[Bucket]
    bucket_probs: dict[str, float]
    bets: list[MarketBet] = field(default_factory=list)
    best_bet: MarketBet = None
    total_edge: float = 0.0


def parse_buckets(bucket_defs: list[tuple]) -> list[Bucket]:
    """Parse bucket definitions into Bucket objects"""
    buckets = []
    for label, lower, upper in bucket_defs:
        is_open_lower = label.startswith("<")
        is_open_upper = label.endswith("+")
        buckets.append(Bucket(
            label=label,
            lower=lower,
            upper=upper,
            is_open_lower=is_open_lower,
            is_open_upper=is_open_upper,
        ))
    return buckets


def compute_bucket_probabilities(
    mean: float,
    std: float,
    buckets: list[Bucket],
    method: str = "normal",
    quantile_data: dict = None,
) -> dict[str, float]:
    """
    Compute probability of temperature falling in each bucket.

    Args:
        mean: Predicted mean temperature
        std: Predicted standard deviation
        buckets: List of Bucket objects
        method: "normal" for Gaussian assumption, "empirical" for quantile-based
        quantile_data: Dict with p10, p25, p50, p75, p90 for empirical method

    Returns:
        Dict mapping bucket label -> probability
    """
    probs = {}

    if method == "empirical" and quantile_data is not None:
        # Use quantile regression outputs for better distribution estimate
        # Fit a skewed distribution to the quantiles
        p10, p25, p50, p75, p90 = (
            quantile_data["p10"], quantile_data["p25"],
            quantile_data["p50"], quantile_data["p75"],
            quantile_data["p90"],
        )

        # Use piecewise linear CDF interpolation
        quantile_points = [
            (-np.inf, 0.0),
            (p10, 0.10),
            (p25, 0.25),
            (p50, 0.50),
            (p75, 0.75),
            (p90, 0.90),
            (np.inf, 1.0),
        ]

        for bucket in buckets:
            if bucket.is_open_lower:
                # P(X < upper) = CDF(upper)
                probs[bucket.label] = _interpolate_cdf(bucket.upper, quantile_points)
            elif bucket.is_open_upper:
                # P(X >= lower) = 1 - CDF(lower)
                probs[bucket.label] = 1.0 - _interpolate_cdf(bucket.lower, quantile_points)
            else:
                # P(lower <= X < upper) = CDF(upper) - CDF(lower)
                probs[bucket.label] = (
                    _interpolate_cdf(bucket.upper, quantile_points)
                    - _interpolate_cdf(bucket.lower, quantile_points)
                )
    else:
        # Normal distribution assumption
        for bucket in buckets:
            if bucket.is_open_lower:
                probs[bucket.label] = stats.norm.cdf(bucket.upper, mean, std)
            elif bucket.is_open_upper:
                probs[bucket.label] = 1.0 - stats.norm.cdf(bucket.lower, mean, std)
            else:
                probs[bucket.label] = (
                    stats.norm.cdf(bucket.upper, mean, std)
                    - stats.norm.cdf(bucket.lower, mean, std)
                )

    # Normalize to sum to 1.0
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    return probs


def _interpolate_cdf(x: float, quantile_points: list) -> float:
    """Linearly interpolate CDF value from quantile points"""
    for i in range(len(quantile_points) - 1):
        x_lo, p_lo = quantile_points[i]
        x_hi, p_hi = quantile_points[i + 1]
        if x_lo <= x <= x_hi:
            if x_hi == np.inf:
                return p_hi
            if x_lo == -np.inf:
                return p_lo
            # Linear interpolation
            t = (x - x_lo) / (x_hi - x_lo)
            return p_lo + t * (p_hi - p_lo)
    return 1.0 if x >= quantile_points[-1][0] else 0.0


def evaluate_bets(
    bucket_probs: dict[str, float],
    market_prices: dict[str, float],
    confidence: float = 0.7,
    min_edge: float = 0.05,
    min_ev: float = 0.03,
) -> list[MarketBet]:
    """
    Evaluate betting opportunities.

    Args:
        bucket_probs: Model-predicted probability for each bucket
        market_prices: Current Polymarket YES prices for each bucket (0-1)
        confidence: Model confidence (0-1), scales position sizing
        min_edge: Minimum probability edge to consider
        min_ev: Minimum expected value to consider

    Returns:
        List of MarketBet objects sorted by EV
    """
    bets = []

    for label, model_prob in bucket_probs.items():
        if label not in market_prices:
            continue

        market_price = market_prices[label]
        implied_prob = market_price  # in efficient market, price ≈ probability

        # Edge: how much more likely we think this outcome is vs market
        edge = model_prob - market_price

        # Expected value: if we buy YES at market_price
        # EV = model_prob * (1 - market_price) - (1 - model_prob) * market_price
        # EV = model_prob - market_price = edge
        ev = edge  # simplified, same as edge for YES bets

        # Kelly criterion: optimal fraction of bankroll
        # f* = (bp - q) / b where b = odds, p = win prob, q = 1-p
        # For Polymarket: b = (1/price - 1), p = model_prob
        if market_price > 0 and market_price < 1:
            b = (1.0 / market_price) - 1.0
            kelly = (b * model_prob - (1 - model_prob)) / b
            kelly = max(0, kelly)
        else:
            kelly = 0.0

        # Scale Kelly by confidence
        scaled_kelly = kelly * confidence

        # Recommendation
        if edge >= min_edge and ev >= min_ev and scaled_kelly > 0.01:
            recommendation = "BUY"
        elif edge <= -min_edge:
            recommendation = "SELL"  # overpriced bucket
        else:
            recommendation = "SKIP"

        bets.append(MarketBet(
            bucket_label=label,
            model_prob=model_prob,
            market_price=market_price,
            implied_prob=implied_prob,
            edge=edge,
            ev=ev,
            kelly_fraction=scaled_kelly,
            confidence=confidence,
            recommendation=recommendation,
        ))

    # Sort by EV descending
    bets.sort(key=lambda b: b.ev, reverse=True)
    return bets


def analyze_market(
    date: str,
    market_type: str,
    predicted_mean: float,
    predicted_std: float,
    market_prices: dict[str, float],
    bucket_defs: list[tuple] = None,
    quantile_data: dict = None,
    confidence: float = 0.7,
) -> MarketAnalysis:
    """
    Complete analysis of a single Polymarket weather market.

    Args:
        date: Target date (YYYY-MM-DD)
        market_type: "high" or "low"
        predicted_mean: Model-predicted mean temperature
        predicted_std: Model-predicted std deviation
        market_prices: Current market prices per bucket
        bucket_defs: Custom bucket definitions
        quantile_data: Quantile regression outputs for empirical method
        confidence: Model confidence

    Returns:
        MarketAnalysis with all betting recommendations
    """
    # Parse buckets
    if bucket_defs is None:
        bucket_defs = DEFAULT_HIGH_TEMP_BUCKETS if market_type == "high" else DEFAULT_LOW_TEMP_BUCKETS
    buckets = parse_buckets(bucket_defs)

    # Compute probabilities
    method = "empirical" if quantile_data else "normal"
    bucket_probs = compute_bucket_probabilities(
        predicted_mean, predicted_std, buckets,
        method=method, quantile_data=quantile_data,
    )

    # Evaluate bets
    bets = evaluate_bets(bucket_probs, market_prices, confidence)

    # Find best bet
    buy_bets = [b for b in bets if b.recommendation == "BUY"]
    best_bet = buy_bets[0] if buy_bets else None

    # Total edge
    total_edge = sum(b.edge * b.kelly_fraction for b in buy_bets)

    analysis = MarketAnalysis(
        date=date,
        market_type=market_type,
        predicted_mean=predicted_mean,
        predicted_std=predicted_std,
        buckets=buckets,
        bucket_probs=bucket_probs,
        bets=bets,
        best_bet=best_bet,
        total_edge=total_edge,
    )

    return analysis


def format_analysis(analysis: MarketAnalysis) -> str:
    """Pretty-print a market analysis"""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  HK Weather Market Analysis: {analysis.date} ({analysis.market_type.upper()} TEMP)")
    lines.append(f"{'=' * 70}")
    lines.append(f"\n  Predicted: {analysis.predicted_mean:.1f}°C ± {analysis.predicted_std:.1f}°C")

    # Bucket probability table
    lines.append(f"\n  {'Bucket':<12} {'Model Prob':>10} {'Market':>10} {'Edge':>8} {'EV':>8} {'Kelly%':>8} {'Action':>8}")
    lines.append(f"  {'-' * 64}")

    for bet in analysis.bets:
        action_icon = "[BUY] " if bet.recommendation == "BUY" else ("[SELL]" if bet.recommendation == "SELL" else "[---]")
        lines.append(
            f"  {bet.bucket_label:<12} {bet.model_prob:>9.1%} {bet.market_price:>9.2f} "
            f"{bet.edge:>+7.1%} {bet.ev:>+7.1%} {bet.kelly_fraction:>7.1%} {action_icon}{bet.recommendation:>6}"
        )

    # Best bet
    if analysis.best_bet:
        lines.append(f"\n  ** BEST BET: {analysis.best_bet.bucket_label}")
        lines.append(f"    Edge: {analysis.best_bet.edge:+.1%} | EV: {analysis.best_bet.ev:+.1%} | Kelly: {analysis.best_bet.kelly_fraction:.1%}")
    else:
        lines.append(f"\n  No +EV bets found. Market appears well-priced.")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_multi_analysis(analyses: list[MarketAnalysis]) -> str:
    """Format multiple market analyses with summary"""
    lines = []

    for a in analyses:
        lines.append(format_analysis(a))

    # Summary table
    lines.append(f"\n\n{'=' * 70}")
    lines.append("  SUMMARY - ALL MARKETS")
    lines.append(f"{'=' * 70}")
    lines.append(f"  {'Date':<12} {'Type':<6} {'Predicted':>10} {'Best Bucket':>12} {'Edge':>8} {'Kelly%':>8}")
    lines.append(f"  {'-' * 58}")

    for a in analyses:
        if a.best_bet:
            lines.append(
                f"  {a.date:<12} {a.market_type:<6} {a.predicted_mean:>9.1f}°C "
                f"{a.best_bet.bucket_label:>12} {a.best_bet.edge:>+7.1%} {a.best_bet.kelly_fraction:>7.1%}"
            )
        else:
            lines.append(
                f"  {a.date:<12} {a.market_type:<6} {a.predicted_mean:>9.1f}°C "
                f"{'No +EV':>12} {'---':>8} {'---':>8}"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with example data
    # Simulating a summer day in HK with predicted high of 32.5°C
    market_prices = {
        "<30": 0.08,
        "30-31": 0.15,
        "31-32": 0.22,
        "32-33": 0.25,
        "33-34": 0.18,
        "34-35": 0.08,
        "35+": 0.04,
    }

    analysis = analyze_market(
        date="2026-08-05",
        market_type="high",
        predicted_mean=32.5,
        predicted_std=1.2,
        market_prices=market_prices,
        quantile_data={"p10": 31.0, "p25": 31.8, "p50": 32.5, "p75": 33.2, "p90": 34.0},
        confidence=0.75,
    )

    print(format_analysis(analysis))
