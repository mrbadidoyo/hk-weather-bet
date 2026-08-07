"""
NWP (Numerical Weather Prediction) Ensemble Forecast Collector.
Uses Open-Meteo's free API to fetch ECMWF (50 members) + GFS (30 members)
ensemble forecasts for Hong Kong Airport (VHHH).
"""
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import (
    OPEN_METEO_LAT,
    OPEN_METEO_LON,
    OPEN_METEO_TIMEZONE,
    OPEN_METEO_APIS,
    NWP_MODELS,
)
from polymarket_strategy import parse_buckets

logger = logging.getLogger(__name__)


class NWPCollector:
    """Collects and processes NWP ensemble forecasts for HK."""

    def __init__(self, lat=OPEN_METEO_LAT, lon=OPEN_METEO_LON):
        self.lat = lat
        self.lon = lon
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "HKWeatherBet/1.0 (research)"
        })

    def fetch_ensemble_forecast(self, forecast_days=16):
        """
        Fetch ensemble forecast from ECMWF (50 members) + GFS (30 members).

        Open-Meteo returns a flat dict with keys like:
          temperature_2m_max_member01_ecmwf_ifs025_ensemble
          temperature_2m_max_member01_ncep_gefs_seamless

        Returns:
            dict: {
                'dates': [...],
                'members_max': [[m01_day0, m01_day1, ...], [m02_day0, ...], ...],
                'members_min': [[m01_day0, ...], ...],
                'n_ecmwf': int,
                'n_gfs': int,
                'ensemble_mean_max': [...],
                'ensemble_mean_min': [...],
            }
        """
        url = OPEN_METEO_APIS["ensemble"]
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "models": ",".join(NWP_MODELS),
            "timezone": OPEN_METEO_TIMEZONE,
            "forecast_days": min(forecast_days, 16),
        }

        logger.info(f"Fetching NWP ensemble from {url}")
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        all_keys = list(daily.keys())

        # Parse member keys
        members_max = []
        members_min = []
        ensemble_mean_max = None
        ensemble_mean_min = None
        n_ecmwf = 0
        n_gfs = 0

        # Member key patterns
        max_member_re = re.compile(r'^temperature_2m_max_member(\d+)_(\w+)$')
        min_member_re = re.compile(r'^temperature_2m_min_member(\d+)_(\w+)$')

        for key in sorted(all_keys):
            vals = daily[key]
            if vals is None:
                continue

            # Check for member keys
            max_match = max_member_re.match(key)
            min_match = min_member_re.match(key)

            if max_match:
                members_max.append(vals)
                model = max_match.group(2)
                if "ecmwf" in model:
                    n_ecmwf += 1
                elif "ncep" in model or "gfs" in model:
                    n_gfs += 1
            elif min_match:
                members_min.append(vals)
            elif key.startswith("temperature_2m_max_") and "member" not in key:
                ensemble_mean_max = vals
            elif key.startswith("temperature_2m_min_") and "member" not in key:
                ensemble_mean_min = vals

        logger.info(f"  Parsed {len(members_max)} max members ({n_ecmwf} ECMWF + {n_gfs} GFS), "
                     f"{len(members_min)} min members, {len(dates)} days")

        return {
            "dates": dates,
            "members_max": members_max,
            "members_min": members_min,
            "ensemble_mean_max": ensemble_mean_max,
            "ensemble_mean_min": ensemble_mean_min,
            "n_ecmwf": n_ecmwf,
            "n_gfs": n_gfs,
            "n_total": len(members_max),
        }

    def fetch_deterministic_forecast(self, forecast_days=16):
        """Fetch deterministic forecast from Open-Meteo best-match model."""
        url = OPEN_METEO_APIS["forecast"]
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "models": "best_match",
            "timezone": OPEN_METEO_TIMEZONE,
            "forecast_days": min(forecast_days, 16),
        }
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        return {
            "dates": daily.get("time", []),
            "tmax": daily.get("temperature_2m_max", []),
            "tmin": daily.get("temperature_2m_min", []),
        }

    def ensemble_to_bucket_probs(self, ens_data, date_idx, bucket_defs, temp_type="max"):
        """
        Convert ensemble member forecasts to bucket probabilities by counting.

        Args:
            ens_data: dict from fetch_ensemble_forecast()
            date_idx: index into dates list (0 = today, 1 = tomorrow, ...)
            bucket_defs: list of (label, lower, upper)
            temp_type: "max" or "min"

        Returns:
            dict: {bucket_label: probability} or None
        """
        buckets = parse_buckets(bucket_defs)
        members = ens_data["members_max"] if temp_type == "max" else ens_data["members_min"]

        if date_idx >= len(ens_data["dates"]):
            return None

        all_values = []
        for member_vals in members:
            if date_idx < len(member_vals) and member_vals[date_idx] is not None:
                all_values.append(member_vals[date_idx])

        if not all_values:
            return None

        probs = {}
        for b in buckets:
            count = sum(1 for v in all_values if b.contains(v))
            probs[b.label] = count / len(all_values)

        return probs

    def ensemble_stats(self, ens_data, date_idx, temp_type="max"):
        """Get summary statistics from ensemble for a given date index."""
        members = ens_data["members_max"] if temp_type == "max" else ens_data["members_min"]

        if date_idx >= len(ens_data["dates"]):
            return None

        all_values = []
        for member_vals in members:
            if date_idx < len(member_vals) and member_vals[date_idx] is not None:
                all_values.append(member_vals[date_idx])

        if not all_values:
            return None

        arr = np.array(all_values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "n_members": len(all_values),
        }


def blend_probs(model_probs, nwp_probs, nwp_weight=0.4, labels=None):
    """
    Blend model (classifier/empirical) probabilities with NWP ensemble probabilities.

    Uses the union of labels so dynamic buckets from the market are respected.
    Missing keys are treated as 0 before normalization.

    Args:
        model_probs: dict {bucket_label: probability}
        nwp_probs: dict {bucket_label: probability} from ensemble member counts
        nwp_weight: weight for NWP (0-1). 0.4 = 60% model + 40% NWP (default)
        labels: optional explicit list of labels to use (e.g. live market buckets)

    Returns:
        dict: {bucket_label: blended_probability} summing to 1.0
    """
    if nwp_probs is None and model_probs is None:
        return {}
    if nwp_probs is None:
        return dict(model_probs)
    if model_probs is None:
        return dict(nwp_probs)

    if labels is None:
        labels = sorted(set(model_probs.keys()) | set(nwp_probs.keys()))

    blended = {}
    for label in labels:
        e = model_probs.get(label, 0.0)
        n = nwp_probs.get(label, 0.0)
        blended[label] = (1.0 - nwp_weight) * e + nwp_weight * n

    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    return blended


def live_nwp_bucket_probs(bucket_defs, temp_type="max", date_idx=0, forecast_days=7):
    """
    One-shot helper: fetch live NWP ensemble and return bucket probabilities
    for the requested day index using the given (dynamic) bucket definitions.

    Args:
        bucket_defs: list of (label, lower, upper) — preferably from live market
        temp_type: "max" or "min"
        date_idx: 0 = today, 1 = tomorrow, ...
        forecast_days: how many days to request from Open-Meteo

    Returns:
        (probs_dict, stats_dict) or (None, None) on failure
    """
    try:
        nwp = NWPCollector()
        ens = nwp.fetch_ensemble_forecast(forecast_days=forecast_days)
        probs = nwp.ensemble_to_bucket_probs(ens, date_idx, bucket_defs, temp_type)
        stats = nwp.ensemble_stats(ens, date_idx, temp_type)
        return probs, stats
    except Exception as e:
        logger.warning(f"live_nwp_bucket_probs failed: {e}")
        return None, None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from config import DEFAULT_HIGH_TEMP_BUCKETS, DEFAULT_LOW_TEMP_BUCKETS

    nwp = NWPCollector()

    print("=" * 60)
    print("  NWP Ensemble Forecast for Hong Kong Airport (VHHH)")
    print("=" * 60)

    print("\nFetching ensemble forecast...")
    ens = nwp.fetch_ensemble_forecast(forecast_days=7)
    print(f"  {ens['n_total']} total members ({ens['n_ecmwf']} ECMWF + {ens['n_gfs']} GFS)")
    print(f"  Dates: {ens['dates'][:3]}...")

    print(f"\n  {'Date':<12} {'Max Mean':>9} {'Max Std':>8} {'Min Mean':>9} {'Min Std':>8} {'Members':>8}")
    print(f"  {'-' * 58}")

    for i in range(min(7, len(ens["dates"]))):
        max_stats = nwp.ensemble_stats(ens, i, "max")
        min_stats = nwp.ensemble_stats(ens, i, "min")
        if max_stats and min_stats:
            print(f"  {ens['dates'][i]:<12} {max_stats['mean']:>8.1f}C {max_stats['std']:>7.2f}C "
                  f"{min_stats['mean']:>8.1f}C {min_stats['std']:>7.2f}C {max_stats['n_members']:>8}")

    print(f"\n  --- Bucket Probabilities ---")
    for i in range(min(3, len(ens["dates"]))):
        print(f"\n  {ens['dates'][i]}:")

        max_probs = nwp.ensemble_to_bucket_probs(ens, i, DEFAULT_HIGH_TEMP_BUCKETS, "max")
        min_probs = nwp.ensemble_to_bucket_probs(ens, i, DEFAULT_LOW_TEMP_BUCKETS, "min")

        if max_probs:
            best = max(max_probs, key=max_probs.get)
            print(f"    High: {' | '.join(f'{k}:{v:.0%}' for k, v in max_probs.items())}")
            print(f"    Best: {best} ({max_probs[best]:.0%})")

        if min_probs:
            best = max(min_probs, key=min_probs.get)
            print(f"    Low:  {' | '.join(f'{k}:{v:.0%}' for k, v in min_probs.items())}")
            print(f"    Best: {best} ({min_probs[best]:.0%})")
