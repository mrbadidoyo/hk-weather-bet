"""
Live Analysis — Dynamic buckets + real NWP ensemble
====================================================
Fetches live Polymarket HK weather markets, extracts actual bucket ranges,
pulls Open-Meteo ensemble (ECMWF+GFS), blends probabilities, and prints
edge / Kelly recommendations.

Usage:
    python live_analysis.py
    python live_analysis.py --days 5 --nwp-weight 0.45
"""
import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_HIGH_TEMP_BUCKETS, DEFAULT_LOW_TEMP_BUCKETS
from polymarket_scraper import PolymarketScraper
from polymarket_strategy import (
    parse_buckets,
    compute_bucket_probabilities,
    evaluate_bets,
    get_buckets_for_market,
)
from nwp_collector import NWPCollector, blend_probs

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def analyze_event(event, ens_data, nwp: NWPCollector, nwp_weight: float = 0.4):
    """Analyze one Polymarket event with dynamic buckets + NWP."""
    snapshot = PolymarketScraper().get_market_snapshot(event)
    prices = snapshot["prices"]
    bucket_defs = snapshot["bucket_defs"]
    event_type = snapshot["type"]  # highest / lowest
    event_date = snapshot["date"]

    if not prices:
        print(f"  [skip] No prices for {snapshot['title']}")
        return None

    if not bucket_defs:
        # Fallback to defaults
        bucket_defs = (
            DEFAULT_HIGH_TEMP_BUCKETS
            if event_type == "highest"
            else DEFAULT_LOW_TEMP_BUCKETS
        )
        print(f"  [warn] Could not parse dynamic buckets, using defaults")

    temp_type = "max" if event_type == "highest" else "min"

    # Map event date → ensemble date index
    date_idx = 0
    if ens_data and event_date in ens_data.get("dates", []):
        date_idx = ens_data["dates"].index(event_date)
    elif ens_data:
        # Approximate by offset from today
        try:
            target = date.fromisoformat(event_date)
            today = date.today()
            date_idx = max(0, (target - today).days)
        except Exception:
            date_idx = 0

    # NWP probabilities using THE SAME buckets as the market
    nwp_probs = None
    nwp_stats = None
    if ens_data:
        nwp_probs = nwp.ensemble_to_bucket_probs(
            ens_data, date_idx, bucket_defs, temp_type
        )
        nwp_stats = nwp.ensemble_stats(ens_data, date_idx, temp_type)

    # Model side: if we only have NWP for now, use it as the model too
    # (classifier would be plugged in here when available)
    model_probs = nwp_probs

    if model_probs is None and nwp_stats:
        # Fallback: normal distribution from ensemble mean/std
        buckets = parse_buckets(bucket_defs)
        model_probs = compute_bucket_probabilities(
            nwp_stats["mean"], max(nwp_stats["std"], 0.5), buckets
        )

    if model_probs is None:
        print(f"  [skip] No probability estimate for {snapshot['title']}")
        return None

    # Blend (when classifier is present, pass it as model_probs)
    labels = [b[0] for b in bucket_defs]
    final_probs = blend_probs(model_probs, nwp_probs, nwp_weight=nwp_weight, labels=labels)

    # Align market prices to labels (exact key match preferred)
    market_prices = {}
    for label in labels:
        if label in prices:
            market_prices[label] = prices[label]
        else:
            # Fuzzy: try without degree symbol variations
            for k, v in prices.items():
                if label.replace("°", "").replace(" ", "") in k.replace("°", "").replace(" ", ""):
                    market_prices[label] = v
                    break

    bets = evaluate_bets(final_probs, market_prices, min_edge=0.05, min_ev=0.03)

    # Print
    print(f"\n{'='*70}")
    print(f"  {snapshot['title']}")
    print(f"  Date: {event_date}  |  Type: {event_type}  |  Buckets: {len(bucket_defs)}")
    if nwp_stats:
        print(f"  NWP: mean={nwp_stats['mean']:.1f}°C  std={nwp_stats['std']:.2f}  "
              f"members={nwp_stats['n_members']}  (idx={date_idx})")
    print(f"{'='*70}")

    print(f"\n  {'Bucket':<22} {'Model':>7} {'Market':>8} {'Edge':>7}  Rec")
    print(f"  {'-'*55}")

    for label in labels:
        mp = final_probs.get(label, 0)
        px = market_prices.get(label)
        edge = (mp - px) if px is not None else None
        rec = ""
        if edge is not None and edge >= 0.05:
            rec = "BUY"
        elif edge is not None and edge <= -0.05:
            rec = "FADE"
        px_str = f"{px:.1%}" if px is not None else "  -"
        edge_str = f"{edge:+.1%}" if edge is not None else "  -"
        print(f"  {label:<22} {mp:>6.1%} {px_str:>8} {edge_str:>7}  {rec}")

    if bets:
        print(f"\n  Value bets (edge ≥ 5%):")
        for b in bets[:5]:
            print(f"    → {b.bucket_label}: model={b.model_prob:.1%}  "
                  f"mkt={b.market_price:.1%}  edge={b.edge:+.1%}  "
                  f"Kelly={b.kelly_fraction:.1%}  [{b.recommendation}]")
    else:
        print(f"\n  No value bets above threshold.")

    return {
        "snapshot": snapshot,
        "final_probs": final_probs,
        "nwp_stats": nwp_stats,
        "bets": bets,
    }


def main():
    parser = argparse.ArgumentParser(description="Live HK weather market analysis")
    parser.add_argument("--days", type=int, default=5, help="Days ahead to scan")
    parser.add_argument("--nwp-weight", type=float, default=0.4,
                        help="Weight for NWP in blend (0-1)")
    args = parser.parse_args()

    print("=" * 70)
    print("  LIVE ANALYSIS — Dynamic Buckets + NWP Ensemble")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  nwp_weight={args.nwp_weight}")
    print("=" * 70)

    # 1. Scrape markets
    print("\n[1/3] Fetching Polymarket HK weather markets...")
    scraper = PolymarketScraper()
    events = scraper.find_events_by_slug(days_ahead=args.days)
    print(f"  Found {len(events)} events")

    if not events:
        print("  No active markets. (Seasonal: typically May–Oct)")
        return

    # 2. Fetch NWP ensemble once
    print("\n[2/3] Fetching Open-Meteo ensemble (ECMWF + GFS)...")
    nwp = NWPCollector()
    try:
        ens = nwp.fetch_ensemble_forecast(forecast_days=max(args.days + 2, 7))
        print(f"  {ens['n_total']} members ({ens['n_ecmwf']} ECMWF + {ens['n_gfs']} GFS)")
        print(f"  Dates: {ens['dates'][:min(5, len(ens['dates']))]} ...")
    except Exception as e:
        print(f"  NWP fetch failed: {e}")
        ens = None

    # 3. Analyze each event
    print("\n[3/3] Analyzing markets with dynamic buckets...")
    results = []
    for event in events:
        r = analyze_event(event, ens, nwp, nwp_weight=args.nwp_weight)
        if r:
            results.append(r)

    print(f"\n{'='*70}")
    print(f"  Done. Analyzed {len(results)} / {len(events)} markets.")
    print("=" * 70)


if __name__ == "__main__":
    main()
