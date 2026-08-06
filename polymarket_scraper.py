"""
Polymarket Price Scraper — Fetches HK weather market prices.
Uses slug-pattern discovery + Gamma API for market data + CLOB API for prices.

Market structure (as of Aug 2026):
- Daily events: highest/lowest temperature in Hong Kong
- 11 single-degree buckets each
- Resolution: HKO "Absolute Daily Max" from Daily Extract
- URL pattern: polymarket.com/event/{type}-temperature-in-hong-kong-on-{month}-{day}-{year}
"""
import logging
import re
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)

# ── Proxy Configuration ──────────────────────────────────────────
# Set PROXY_URL to your Cloudflare Worker URL to bypass ISP blocking.
# Leave empty to use direct Polymarket APIs (requires VPN/DNS in blocked regions).
#
# Deploy your own proxy: see cloudflare_worker/README.md
# Example: PROXY_URL = "https://polymarket-proxy.your-subdomain.workers.dev"
PROXY_URL = ""  # <-- Set your worker URL here

# API endpoints (routed through proxy if PROXY_URL is set)
if PROXY_URL:
    GAMMA_API = f"{PROXY_URL}/gamma"
    CLOB_API = f"{PROXY_URL}/clob"
    logger.info(f"Using proxy: {PROXY_URL}")
else:
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]


class PolymarketScraper:
    """Scrape Polymarket HK weather market prices."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/html, */*",
        })

    # ── Market Discovery ──────────────────────────────────────────

    def find_events_by_slug(self, days_ahead=10):
        """
        Find HK weather events by constructing URL slugs.
        Checks today + days_ahead days for both high and low temp markets.
        Returns list of event dicts from Gamma API.
        """
        found = []
        today = date.today()

        for day_offset in range(days_ahead):
            d = today + timedelta(days=day_offset)
            month_name = MONTHS[d.month - 1]
            day_str = d.day
            year = d.year

            for temp_type in ["highest", "lowest"]:
                slug = f"{temp_type}-temperature-in-hong-kong-on-{month_name}-{day_str}-{year}"

                try:
                    resp = self.session.get(
                        f"{GAMMA_API}/events",
                        params={"slug": slug},
                        timeout=15,
                    )
                    if resp.status_code != 200:
                        continue
                    events = resp.json()
                    if events and isinstance(events, list) and len(events) > 0:
                        event = events[0]
                        event["_slug"] = slug
                        event["_date"] = d.isoformat()
                        event["_type"] = temp_type
                        found.append(event)
                        logger.info(f"Found: {event.get('title', slug)}")
                except Exception as e:
                    logger.warning(f"Error fetching slug {slug}: {e}")

                time.sleep(0.2)

        return found

    # ── Price Fetching ────────────────────────────────────────────

    def get_price_history(self, condition_id, interval="all", fidelity=60):
        """Get price history from CLOB API."""
        try:
            resp = self.session.get(
                f"{CLOB_API}/prices-history",
                params={"market": condition_id, "interval": interval, "fidelity": fidelity},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("history", [])
        except Exception as e:
            logger.warning(f"Error fetching price history for {condition_id}: {e}")
            return []

    def get_current_prices(self, event):
        """
        Extract current prices from event data.
        The Gamma API event response includes markets with outcomePrices.
        """
        prices = {}
        for market in event.get("markets", []):
            question = market.get("question", "")
            outcome = market.get("outcome", "")
            outcome_prices = market.get("outcomePrices", "")

            # Parse outcomePrices — it's a JSON string like '["0.25", "0.75"]'
            if isinstance(outcome_prices, str):
                try:
                    price_list = json.loads(outcome_prices)
                    yes_price = float(price_list[0]) if len(price_list) > 0 else None
                except (json.JSONDecodeError, ValueError, IndexError):
                    yes_price = None
            elif isinstance(outcome_prices, list):
                yes_price = float(outcome_prices[0]) if len(outcome_prices) > 0 else None
            else:
                yes_price = None

            # Extract bucket label from question
            # Pattern: "Highest temperature in Hong Kong on August 7?" with outcome "27°C or below"
            bucket = outcome if outcome else question

            if yes_price is not None:
                prices[bucket] = yes_price

        return prices

    # ── Full Scrape Pipeline ──────────────────────────────────────

    def scrape_and_save(self, days_ahead=10):
        """Full pipeline: find events, fetch prices, save to CSV."""
        print("=" * 60)
        print("  Polymarket HK Weather Market Scraper")
        print("=" * 60)

        # Step 1: Find events
        print(f"\n[1/3] Finding HK weather markets (next {days_ahead} days)...")
        events = self.find_events_by_slug(days_ahead=days_ahead)
        print(f"  Found {len(events)} events")

        if not events:
            print("\n  No HK weather markets found.")
            print("  Markets are seasonal — typically active May-Oct.")
            return None

        for e in events:
            title = e.get("title", e.get("_slug", "?"))
            n_markets = len(e.get("markets", []))
            print(f"    {title} ({n_markets} buckets)")

        # Step 2: Extract current prices + fetch history
        print(f"\n[2/3] Extracting prices...")
        all_rows = []

        for event in events:
            slug = event.get("_slug", "")
            event_date = event.get("_date", "")
            event_type = event.get("_type", "")
            title = event.get("title", "")

            # Current prices from Gamma API
            current = self.get_current_prices(event)
            print(f"\n  {title}:")
            for bucket, price in sorted(current.items()):
                print(f"    {bucket:<20} {price:.1%}")
                all_rows.append({
                    "event_title": title,
                    "event_slug": slug,
                    "event_date": event_date,
                    "event_type": event_type,
                    "bucket": bucket,
                    "price": price,
                    "timestamp": datetime.now().isoformat(),
                    "source": "gamma_api_current",
                })

            # Price history from CLOB API
            for market in event.get("markets", []):
                condition_id = market.get("conditionId")
                if not condition_id:
                    continue
                history = self.get_price_history(condition_id, interval="all", fidelity=60)
                if history:
                    for point in history:
                        all_rows.append({
                            "event_title": title,
                            "event_slug": slug,
                            "event_date": event_date,
                            "event_type": event_type,
                            "bucket": market.get("outcome", ""),
                            "price": point.get("p"),
                            "timestamp": pd.to_datetime(point.get("t"), unit="s"),
                            "source": "clob_history",
                        })
                time.sleep(0.3)

        # Step 3: Save
        if all_rows:
            df = pd.DataFrame(all_rows)
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            output_path = PROCESSED_DATA_DIR / "polymarket_prices.csv"
            df.to_csv(output_path, index=False)

            print(f"\n[3/3] Saved {len(df)} rows to {output_path}")
            print(f"  Events: {df['event_title'].nunique()}")
            print(f"  Buckets: {df['bucket'].nunique()}")
            print(f"  Sources: {df['source'].value_counts().to_dict()}")

            return df
        else:
            print(f"\n[3/3] No price data extracted.")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scraper = PolymarketScraper()
    result = scraper.scrape_and_save(days_ahead=10)
