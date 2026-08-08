"""
Telegram Prediction Alert — Runs hourly, monitors tomorrow's HK temperature event.
Sends predictions vs Polymarket prices until event is resolved.

Improvements v2:
- Dynamic bucket ranges from Polymarket
- Kelly Criterion for bet sizing
- Model performance tracking
- Ensemble NWP integration
"""
import sys
import io
import re
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import requests

from config import (
    DEFAULT_HIGH_TEMP_BUCKETS,
    DEFAULT_LOW_TEMP_BUCKETS,
    PROCESSED_DATA_DIR,
)
from data_collector import HKODataCollector
from polymarket_scraper import PolymarketScraper
from model_tracker import log_prediction, update_prediction, get_performance_stats
from bias_corrector import log_forecast_error, get_bias_correction, format_bias_summary

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Telegram config
TELEGRAM_BOT_TOKEN = "8909269274:AAF5XoUwEh9zZXekHniWFVVwUTijEn9Vz-4"
TELEGRAM_CHAT_ID = "225257336"

# Bankroll config for Kelly Criterion
BANKROLL = 100.0  # Starting bankroll in USD
KELLY_FRACTION = 0.25  # Quarter-Kelly for safety


# ── Data Fetching ────────────────────────────────────────────────

def get_recent_data(days=365):
    """Fetch recent HKA temperature data (1 year for DOY matching)."""
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
        df['date'] = pd.to_datetime(df[[yc, mc, dc]].rename(columns={yc: 'Year', mc: 'Month', dc: 'Day'}), errors='coerce')
        df['value'] = pd.to_numeric(df[vc], errors='coerce')
        return df.dropna(subset=['date', 'value']).sort_values('date').set_index('date')[['value']]

    combined = parse(fetch(urls['max'])).join(parse(fetch(urls['min'])),
                                               lsuffix='_max', rsuffix='_min', how='inner')
    combined.columns = ['max_temp', 'min_temp']
    return combined.dropna().tail(days)


def get_hko_forecast():
    """Get HKO 9-day forecast."""
    hko = HKODataCollector()
    try:
        forecast = hko.fetch_9day_forecast()
        wf = forecast.get("weatherForecast", [])
        results = []
        for day in wf:
            date_str = day.get("forecastDate", "")
            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                continue
            max_t = float(day.get("forecastMaxtemp", {}).get("value", 0))
            min_t = float(day.get("forecastMintemp", {}).get("value", 0))
            if max_t > 0 and min_t > 0:
                results.append({
                    "date": dt,
                    "max_temp": max_t,
                    "min_temp": min_t,
                })
        return results
    except Exception as e:
        logger.warning(f"Error fetching HKO forecast: {e}")
        return []


def get_actual_temperature(target_date):
    """Check if HKO has published actual temperature for target_date."""
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
        df['date'] = pd.to_datetime(df[[yc, mc, dc]].rename(columns={yc: 'Year', mc: 'Month', dc: 'Day'}), errors='coerce')
        df['value'] = pd.to_numeric(df[vc], errors='coerce')
        return df.dropna(subset=['date', 'value']).sort_values('date').set_index('date')[['value']]

    try:
        max_df = parse(fetch(urls['max']))
        min_df = parse(fetch(urls['min']))
        
        # Check if target_date is in the data
        target = pd.Timestamp(target_date)
        max_val = max_df.loc[target, 'value'] if target in max_df.index else None
        min_val = min_df.loc[target, 'value'] if target in min_df.index else None
        
        return max_val, min_val
    except Exception as e:
        logger.warning(f"Error fetching actual temperature: {e}")
        return None, None


# ── Prediction Engine ────────────────────────────────────────────

def predict_buckets(temp_type, recent_data, target_date, forecast_temp=None, bucket_defs=None):
    """
    Predict bucket probabilities for target_date using historical distribution + forecast.
    
    Args:
        temp_type: 'max' or 'min'
        recent_data: DataFrame with historical temperature data
        target_date: Date to predict
        forecast_temp: Optional forecast temperature for blending
        bucket_defs: Optional list of (label, lo, hi) tuples. If None, uses defaults.
    """
    col = f'{temp_type}_temp'
    if bucket_defs is None:
        bucket_defs = DEFAULT_HIGH_TEMP_BUCKETS if temp_type == 'max' else DEFAULT_LOW_TEMP_BUCKETS

    # Historical same-day stats (±5 days window)
    doy = target_date.timetuple().tm_yday
    recent_data_copy = recent_data.copy()
    recent_data_copy['doy'] = recent_data_copy.index.dayofyear

    # Get historical values for same DOY ±5
    doy_window = recent_data_copy[
        (recent_data_copy['doy'] >= doy - 5) & (recent_data_copy['doy'] <= doy + 5)
    ][col]

    # Fallback: use same month if DOY window is empty
    if len(doy_window) < 10:
        month = target_date.month
        doy_window = recent_data_copy[recent_data_copy.index.month == month][col]
    
    # Fallback: use last 60 days if month is also empty
    if len(doy_window) < 10:
        doy_window = recent_data[col].tail(60)

    hist_mean = doy_window.mean()
    hist_std = max(doy_window.std(), 0.5)

    # Recent trend (last 7 days up to today)
    today = datetime.now().date()
    recent_7 = recent_data[col].loc[:pd.Timestamp(today)].tail(7)
    recent_mean = recent_7.mean()

    # If we have HKO forecast, blend it in with higher weight
    if forecast_temp is not None and forecast_temp > 0:
        # HKO forecast is very accurate for 1-3 day horizon
        # 30% historical, 20% recent, 50% forecast
        blended_mean = 0.30 * hist_mean + 0.20 * recent_mean + 0.50 * forecast_temp
        blended_std = max(hist_std * 0.7, 0.8)
    else:
        # 55% historical, 45% recent
        blended_mean = 0.55 * hist_mean + 0.45 * recent_mean
        blended_std = hist_std

    # Use empirical distribution (KDE) for probabilities, SHIFTED by forecast
    from scipy import stats as sp_stats

    values = doy_window.dropna().values
    if len(values) < 5:
        values = recent_data[col].dropna().tail(60).values

    # Calculate the shift needed to align historical mean with blended_mean
    hist_center = np.mean(values)
    shift = blended_mean - hist_center

    # Shift the values to center around the forecast
    shifted_values = values + shift

    # KDE on shifted values
    try:
        kde = sp_stats.gaussian_kde(shifted_values, bw_method=0.4)
    except Exception:
        kde = None

    probs = {}
    for label, lo, hi in bucket_defs:
        actual_lo = max(lo, shifted_values.min() - 2) if lo == -999 else lo
        actual_hi = min(hi, shifted_values.max() + 2) if hi == 999 else hi

        if kde is not None:
            x = np.linspace(actual_lo, actual_hi, 200)
            y = kde(x)
            prob = np.trapezoid(y, x)
        else:
            # Gaussian fallback
            prob = sp_stats.norm.cdf(actual_hi, blended_mean, hist_std) - \
                   sp_stats.norm.cdf(actual_lo, blended_mean, hist_std)

        probs[label] = max(prob, 0.001)

    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    return probs, blended_mean, hist_std


# ── Polymarket Prices ────────────────────────────────────────────

def get_market_prices():
    """Get current Polymarket prices for upcoming HK weather events."""
    scraper = PolymarketScraper()
    events = scraper.find_events_by_slug(days_ahead=3)

    market_data = {}
    for event in events:
        event_date = event.get("_date", "")
        event_type = event.get("_type", "")
        title = event.get("title", "")

        prices = scraper.get_current_prices(event)
        # Clean bucket names
        clean_prices = {}
        for bucket_key, price in prices.items():
            match = re.search(r'(\d+°C(?:\s+or\s+(?:below|higher))?)', bucket_key)
            if match:
                clean_bucket = match.group(1)
            else:
                clean_bucket = bucket_key
            clean_prices[clean_bucket] = price

        key = f"{event_date}_{event_type}"
        market_data[key] = {
            "title": title,
            "date": event_date,
            "type": event_type,
            "prices": clean_prices,
        }

    return market_data


def check_if_resolved(market_prices, date_str):
    """Check if event is resolved by looking at prices (winner = 1.0)."""
    for temp_type in ["highest", "lowest"]:
        key = f"{date_str}_{temp_type}"
        if key in market_prices:
            prices = market_prices[key]["prices"]
            # If any bucket has price >= 0.95, it's likely resolved
            for bucket, price in prices.items():
                if price >= 0.95:
                    return True, bucket
    return False, None


# ── Improvement #1: Dynamic Bucket Ranges ────────────────────────

def extract_dynamic_buckets(market_prices, date_str, temp_type):
    """
    Extract actual bucket definitions from Polymarket market data.
    Uses shared parser from polymarket_strategy (handles "32°C",
    "27°C or below", "37°C or higher", ranges, etc.).

    Returns list of (label, lo, hi) tuples.
    """
    from polymarket_strategy import buckets_from_labels

    key = f"{date_str}_{temp_type}"
    fallback = (
        DEFAULT_HIGH_TEMP_BUCKETS
        if temp_type == "highest"
        else DEFAULT_LOW_TEMP_BUCKETS
    )

    if key not in market_prices:
        return fallback

    prices = market_prices[key].get("prices", {})
    if not prices:
        return fallback

    buckets = buckets_from_labels(list(prices.keys()))
    return buckets if buckets else fallback


# ── Improvement #3: Kelly Criterion ──────────────────────────────

def calculate_kelly_stake(prob, odds, bankroll=BANKROLL, kelly_frac=KELLY_FRACTION):
    """
    Calculate optimal bet size using Kelly Criterion.
    
    Args:
        prob: Model's estimated probability of winning
        odds: Decimal odds (1 / market_price)
        bankroll: Current bankroll
        kelly_frac: Fraction of Kelly to use (0.25 = quarter-Kelly for safety)
    
    Returns:
        Recommended stake amount
    """
    if prob <= 0 or odds <= 1:
        return 0.0
    
    # Kelly formula: f = (bp - q) / b
    # b = odds - 1 (net odds)
    # p = win probability
    # q = lose probability = 1 - p
    b = odds - 1
    p = prob
    q = 1 - p
    
    kelly_pct = (b * p - q) / b
    
    # Only bet if edge is positive
    if kelly_pct <= 0:
        return 0.0
    
    # Apply fractional Kelly for safety
    stake = bankroll * kelly_pct * kelly_frac
    
    # Cap at 10% of bankroll
    stake = min(stake, bankroll * 0.10)
    
    return round(stake, 2)


# ── Improvement #5: Ensemble NWP ─────────────────────────────────

def get_ensemble_nwp_forecast(lat=22.3167, lon=114.1667):
    """
    Fetch ensemble NWP forecast from Open-Meteo.
    Uses 80 ensemble members (50 ECMWF + 30 GFS) for uncertainty quantification.
    
    Returns dict with:
    - mean_temp: Ensemble mean
    - std_temp: Ensemble standard deviation (uncertainty)
    - percentiles: {10, 25, 50, 75, 90}
    - members: Raw ensemble values
    """
    try:
        from nwp_collector import NWPCollector
        collector = NWPCollector(lat=lat, lon=lon)
        
        # Fetch ensemble data
        ensemble_data = collector.fetch_ensemble_forecast(forecast_days=3)
        
        if not ensemble_data:
            return None
        
        # Extract tomorrow's temperatures
        tomorrow = datetime.now().date() + timedelta(days=1)
        tomorrow_str = tomorrow.isoformat()
        
        dates = ensemble_data.get("dates", [])
        if tomorrow_str not in dates:
            return None
        
        tomorrow_idx = dates.index(tomorrow_str)
        
        # Get ensemble members for tomorrow
        members_max = ensemble_data.get("members_max", [])
        members_min = ensemble_data.get("members_min", [])
        
        max_temps = [m[tomorrow_idx] for m in members_max if len(m) > tomorrow_idx and m[tomorrow_idx] is not None]
        min_temps = [m[tomorrow_idx] for m in members_min if len(m) > tomorrow_idx and m[tomorrow_idx] is not None]
        
        if not max_temps:
            return None
        
        max_temps = np.array(max_temps)
        min_temps = np.array(min_temps)
        
        return {
            "max": {
                "mean": np.mean(max_temps),
                "std": np.std(max_temps),
                "percentiles": {
                    10: np.percentile(max_temps, 10),
                    25: np.percentile(max_temps, 25),
                    50: np.percentile(max_temps, 50),
                    75: np.percentile(max_temps, 75),
                    90: np.percentile(max_temps, 90),
                },
                "members": max_temps.tolist(),
            },
            "min": {
                "mean": np.mean(min_temps),
                "std": np.std(min_temps),
                "percentiles": {
                    10: np.percentile(min_temps, 10),
                    25: np.percentile(min_temps, 25),
                    50: np.percentile(min_temps, 50),
                    75: np.percentile(min_temps, 75),
                    90: np.percentile(min_temps, 90),
                },
                "members": min_temps.tolist(),
            }
        }
    except Exception as e:
        logger.debug(f"Error fetching ensemble NWP (non-critical): {e}")
        return None


# ── Message Formatting ───────────────────────────────────────────

def format_message(target_date, predictions, market_prices, actual_temps):
    """Format prediction + market data into Telegram message."""
    from zoneinfo import ZoneInfo

    try:
        generated_at = datetime.now(ZoneInfo("Asia/Hong_Kong"))
    except Exception:
        generated_at = datetime.now()

    now = generated_at.strftime("%Y-%m-%d %H:%M HKT")
    date_str = target_date.strftime("%Y-%m-%d")

    def _format_temp_display(value):
        if value is None:
            return "?"
        try:
            return f"{int(round(float(value)))}°C"
        except (TypeError, ValueError):
            return str(value)

    def _contract_label(bucket_label):
        bucket_text = str(bucket_label).strip()
        match = re.search(r"(\d+)", bucket_text)
        if match:
            return f"{int(match.group(1))}°C"
        return bucket_text

    def _contract_temp_value(bucket_label):
        bucket_text = str(bucket_label).strip()
        match = re.search(r"(\d+)", bucket_text)
        if match:
            return int(match.group(1))
        return 0

    def _temp_sort_key(bucket_item):
        return _contract_temp_value(bucket_item[0])

    def _format_cents(price):
        if price is None:
            return "?¢"
        try:
            return f"{int(round(float(price) * 100))}¢"
        except (TypeError, ValueError):
            return "?¢"

    def _format_percent(value):
        if value is None:
            return None
        try:
            return f"{float(value):+.1%}"
        except (TypeError, ValueError):
            return None

    def _format_model_market_line(model_prob, market_prob):
        return f"Model {model_prob:.0%} | Market {market_prob:.0%}"

    def _get_bias_report():
        """Normalize bias report to a flat shape used by the Telegram message."""
        try:
            from bias_corrector import get_bias_report
            raw = get_bias_report()
            out = {}
            for side in ("high", "low"):
                side_data = raw.get(side, {})
                # New format: {"global": {...}, "by_range": {...}}
                if isinstance(side_data, dict) and "global" in side_data:
                    g = side_data.get("global", {})
                    out[side] = {
                        "correction": g.get("correction", 0.0),
                        "n_samples": g.get("n_samples", 0),
                        "confidence": g.get("confidence", "LOW"),
                        "direction": g.get("direction", "calibrated"),
                        "by_range": side_data.get("by_range", {}),
                    }
                else:
                    # Old flat format fallback
                    out[side] = {
                        "correction": side_data.get("correction", 0.0),
                        "n_samples": side_data.get("n_samples", 0),
                        "confidence": side_data.get("confidence", "LOW"),
                        "direction": side_data.get("direction", "calibrated"),
                        "by_range": {},
                    }
            return out
        except Exception:
            return {
                "high": {"correction": 0.0, "n_samples": 0, "by_range": {}},
                "low": {"correction": 0.0, "n_samples": 0, "by_range": {}},
            }

    def _market_price_for_label(label, market_map):
        if not market_map:
            return None
        if label in market_map:
            return market_map[label]
        label_num = re.search(r"(\d+)", label)
        label_num = label_num.group(1) if label_num else None
        if label_num:
            for market_label, price in market_map.items():
                market_text = str(market_label)
                market_num = re.search(r"(\d+)", market_text)
                if market_num and market_num.group(1) == label_num:
                    return price
        return None

    def _value_rating(edge):
        if edge >= 0.10:
            return "Strong Value"
        if edge >= 0.05:
            return "Moderate Value"
        return "Low Value"

    def _recommendation_markers(label, best_main, best_lottery):
        markers = []
        if best_main and _contract_label(best_main["bucket"]) == label:
            markers.append("MAIN")
        if best_lottery and _contract_label(best_lottery["bucket"]) == label:
            markers.append("LOTTERY")
        return markers

    def _format_expected_value(bet):
        ev = bet.get("ev") if isinstance(bet, dict) else None
        if ev is None:
            return None
        try:
            return f"Expected Value: {float(ev):+.1%}"
        except (TypeError, ValueError):
            return None

    def _format_value_tag(edge):
        try:
            edge_value = float(edge)
        except (TypeError, ValueError):
            return None
        if abs(edge_value) < 0.01:
            return "⚪ FAIR VALUE"
        if edge_value > 0:
            return "🟢 UNDERPRICED"
        return "🔴 OVERPRICED"

    lines = [
        "🌡️ HK WEATHER ALERT",
        "",
        f"🕒 Generated: {now}",
        f"📅 Target: {date_str}",
    ]

    is_resolved, _ = check_if_resolved(market_prices, date_str)
    status = "RESOLVED" if is_resolved and actual_temps else "PENDING"
    lines.append(f"⏳ Status: {status}")
    lines.extend(["", "🌤 Forecast", ""])

    forecast = predictions.get("forecast", {})
    lines.append(f"High: {_format_temp_display(forecast.get('max_temp'))}")
    lines.append(f"Low : {_format_temp_display(forecast.get('min_temp'))}")

    high_probs = predictions.get("high_probs", {})
    low_probs = predictions.get("low_probs", {})
    market_key_high = f"{date_str}_highest"
    market_key_low = f"{date_str}_lowest"
    market_high = market_prices.get(market_key_high, {}).get("prices", {})
    market_low = market_prices.get(market_key_low, {}).get("prices", {})

    def _leading_bucket(probs: dict, market_map: dict):
        """Return (model_leader_label, model_prob, market_leader_label, market_price)."""
        model_label, model_p = None, None
        if probs:
            model_label = max(probs, key=probs.get)
            model_p = probs.get(model_label)
        market_label, market_p = None, None
        if market_map:
            market_label = max(market_map, key=market_map.get)
            market_p = market_map.get(market_label)
        return model_label, model_p, market_label, market_p

    lines.extend(["", "🏆 Leading Bucket", ""])
    for side_name, probs, mkt in [
        ("High", high_probs, market_high),
        ("Low", low_probs, market_low),
    ]:
        m_lab, m_p, k_lab, k_p = _leading_bucket(probs, mkt)
        if m_lab is not None:
            lines.append(
                f"{side_name} model : {_contract_label(m_lab)} ({m_p:.0%})"
            )
        else:
            lines.append(f"{side_name} model : -")
        if k_lab is not None:
            lines.append(
                f"{side_name} market: {_contract_label(k_lab)} ({k_p:.0%})"
            )
        else:
            lines.append(f"{side_name} market: -")
        lines.append("")

    lines.extend(["Bias Correction", ""])

    bias_report = _get_bias_report()
    for side, label in [("high", "High"), ("low", "Low ")]:
        b = bias_report.get(side, {})
        corr = b.get("correction", 0.0)
        n = b.get("n_samples", 0)
        conf = b.get("confidence", "")
        if n > 0:
            lines.append(f"{label}: {corr:+.1f}°C (n={n}, {conf})")
        else:
            lines.append(f"{label}: {corr:+.1f}°C")
    lines.append("")
    lines.append("━━━━━━━━━━━━━━")

    best_bets = {"high": {"main": None, "lottery": None}, "low": {"main": None, "lottery": None}}
    if not is_resolved:
        best_bets = find_recommended_bets(predictions, market_prices, date_str)

    def _best_for_side(side):
        return best_bets.get(side, {"main": None, "lottery": None})

    def _side_market(side):
        return market_high if side == "high" else market_low

    def _side_probs(side):
        return high_probs if side == "high" else low_probs

    lines.extend(["", "🎯 MAIN BETS", ""])
    for side, emoji, title in [("high", "🔴", "HIGH"), ("low", "🔵", "LOW")]:
        best_side = _best_for_side(side)
        main_bet = best_side.get("main")
        lines.append(f"{emoji} {title}")
        lines.append("")
        if main_bet:
            market_map = _side_market(side)
            display_label = _contract_label(main_bet["bucket"])
            lines.append(f"Buy YES {display_label} @ {_format_cents(main_bet['market'])}")
            lines.append("")
            lines.append(f"Model {main_bet['model']:.0%} | Market {main_bet['market']:.0%}")
            ev_line = _format_expected_value(main_bet)
            if ev_line:
                lines.append(f"Edge {_format_percent(main_bet['edge'])} | {ev_line}")
            else:
                lines.append(f"Edge {_format_percent(main_bet['edge'])}")
            value_tag = _format_value_tag(main_bet.get('edge'))
            if value_tag:
                lines.append(value_tag)
            lines.append(f"Kelly ${main_bet.get('kelly_stake', 0.0):.2f}")
        else:
            lines.append("No main bet found.")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━")
    lines.append("")
    lines.append("🎰 LOTTERY BETS")
    lines.append("")
    for side, emoji, title in [("high", "🔴", "HIGH"), ("low", "🔵", "LOW")]:
        best_side = _best_for_side(side)
        lottery_bet = best_side.get("lottery")
        lines.append(f"{emoji} {title}")
        lines.append("")
        if lottery_bet:
            display_label = _contract_label(lottery_bet["bucket"])
            lines.append(f"Buy YES {display_label} @ {_format_cents(lottery_bet['market'])}")
            lines.append("")
            lines.append(f"Model {lottery_bet['model']:.0%} | Market {lottery_bet['market']:.0%}")
            lines.append(f"Edge {_format_percent(lottery_bet['edge'])}")
        else:
            lines.append("No lottery value bet found.")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━")
    lines.append("")
    lines.append("💎 OTHER VALUE BETS")
    lines.append("")

    all_bets = find_best_bets(predictions, market_prices, date_str)
    excluded = set()
    for side in ("high", "low"):
        for key in ("main", "lottery"):
            bet = best_bets.get(side, {}).get(key)
            if bet:
                excluded.add((side.upper(), _contract_label(bet["bucket"])))

    for side, side_label in [("HIGH", "high"), ("LOW", "low")]:
        side_bets = [
            b for b in all_bets
            if b.get("type") == side and (side, _contract_label(b["bucket"])) not in excluded
        ]
        side_bets.sort(key=lambda b: b.get("edge", 0), reverse=True)
        side_bets = side_bets[:3]
        lines.append(side)
        if side_bets:
            for bet in side_bets:
                label = _contract_label(bet["bucket"])
                lines.append(f"• {label} ({_format_percent(bet['edge'])})")
        else:
            lines.append("• None")
        lines.append("")

    return "\n".join(lines)

def find_recommended_bets(predictions, market_prices, date_str):
    """
    Find 2 types of bets for HIGH and LOW:
    1. MAIN BET = closest to HKO forecast, reasonable price
    2. LOTTERY TICKET = underpriced value bet (edge > 10%, prob > 15%, market < 30%)
    
    Returns dict with 'high' and 'low' keys, each containing 'main' and 'lottery'.
    """
    results = {
        "high": {"main": None, "lottery": None},
        "low": {"main": None, "lottery": None}
    }

    forecast = predictions.get("forecast", {})
    
    for temp_type, prob_key, market_type, result_key, forecast_key in [
        ("HIGH", "high_probs", "highest", "high", "max_temp"),
        ("LOW", "low_probs", "lowest", "low", "min_temp"),
    ]:
        probs = predictions.get(prob_key, {})
        market_key = f"{date_str}_{market_type}"
        market = market_prices.get(market_key, {}).get("prices", {})
        forecast_temp = forecast.get(forecast_key)
        
        # Find MAIN BET: bucket with highest probability that's close to forecast
        main_candidates = []
        for bucket, model_prob in probs.items():
            # Extract temperature from bucket name
            import re
            match = re.search(r'(\d+)', bucket)
            if match:
                bucket_temp = int(match.group(1))
                # Find market price
                market_price = None
                for mk, mp in market.items():
                    if bucket.split("°")[0] in mk or mk.split("°")[0] in bucket:
                        market_price = mp
                        break
                
                if market_price is not None and model_prob > 0.10:
                    # Score: probability * closeness to forecast
                    if forecast_temp:
                        distance = abs(bucket_temp - forecast_temp)
                        # Prefer buckets within 1-2 degrees of forecast
                        closeness_score = max(0, 1 - distance * 0.3)
                    else:
                        closeness_score = 0.5
                    
                    score = model_prob * (0.5 + 0.5 * closeness_score)
                    main_candidates.append({
                        "type": temp_type,
                        "bucket": bucket,
                        "model": model_prob,
                        "market": market_price,
                        "edge": model_prob - market_price,
                        "score": score,
                        "emoji": "🔴" if temp_type == "HIGH" else "🔵",
                        "temp": bucket_temp,
                    })
        
        if main_candidates:
            main_candidates.sort(key=lambda x: x["score"], reverse=True)
            best_main = main_candidates[0]
            best_main["confidence"] = "MAIN"
            # Improvement #3: Calculate Kelly stake
            if best_main["market"] > 0:
                odds = 1.0 / best_main["market"]
                best_main["kelly_stake"] = calculate_kelly_stake(best_main["model"], odds)
            else:
                best_main["kelly_stake"] = 0.0
            results[result_key]["main"] = best_main
        
        # Find LOTTERY TICKET: underpriced value bet
        lottery_candidates = []
        for bucket, model_prob in probs.items():
            for mk, mp in market.items():
                if bucket.split("°")[0] in mk or mk.split("°")[0] in bucket:
                    edge = model_prob - mp
                    
                    # Apply lottery criteria: high edge, reasonable prob, cheap
                    if edge > 0.10 and model_prob > 0.15 and mp < 0.30:
                        score = edge * model_prob
                        lottery_candidates.append({
                            "type": temp_type,
                            "bucket": bucket,
                            "model": model_prob,
                            "market": mp,
                            "edge": edge,
                            "score": score,
                            "emoji": "🔴" if temp_type == "HIGH" else "🔵",
                        })
                    break
        
        if lottery_candidates:
            lottery_candidates.sort(key=lambda x: x["score"], reverse=True)
            best_lottery = lottery_candidates[0]
            
            # Add confidence label
            if best_lottery["edge"] > 0.20 and best_lottery["model"] > 0.25:
                best_lottery["confidence"] = "HIGH"
            elif best_lottery["edge"] > 0.15:
                best_lottery["confidence"] = "MEDIUM"
            else:
                best_lottery["confidence"] = "LOW"
            
            # Improvement #3: Calculate Kelly stake
            if best_lottery["market"] > 0:
                odds = 1.0 / best_lottery["market"]
                best_lottery["kelly_stake"] = calculate_kelly_stake(best_lottery["model"], odds)
            else:
                best_lottery["kelly_stake"] = 0.0
            
            results[result_key]["lottery"] = best_lottery

    return results


def find_best_bets(predictions, market_prices, date_str):
    """Find all bets where model probability > market price (edge > 5%)."""
    bets = []

    for temp_type, prob_key, market_type in [
        ("HIGH", "high_probs", "highest"),
        ("LOW", "low_probs", "lowest"),
    ]:
        probs = predictions.get(prob_key, {})
        market_key = f"{date_str}_{market_type}"
        market = market_prices.get(market_key, {}).get("prices", {})

        for bucket, model_prob in probs.items():
            for mk, mp in market.items():
                if bucket.split("°")[0] in mk or mk.split("°")[0] in bucket:
                    edge = model_prob - mp
                    if edge > 0.05:
                        bets.append({
                            "type": temp_type,
                            "bucket": bucket,
                            "model": model_prob,
                            "market": mp,
                            "edge": edge,
                            "emoji": "🔴" if temp_type == "HIGH" else "🔵",
                        })
                    break

    bets.sort(key=lambda x: x["edge"], reverse=True)
    return bets


# ── Main ─────────────────────────────────────────────────────────

def send_telegram(message):
    """Send message via Telegram bot."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        result = resp.json()
        if result.get("ok"):
            print("[OK] Telegram message sent!")
        else:
            print(f"[ERR] Telegram error: {result}")
            if "can't parse" in str(result).lower():
                payload["parse_mode"] = ""
                resp = requests.post(url, json=payload, timeout=15)
                result = resp.json()
                if result.get("ok"):
                    print("[OK] Sent (plain text fallback)")
                else:
                    print(f"[ERR] Still failed: {result}")
    except Exception as e:
        print(f"[ERR] Telegram send error: {e}")


def build_prediction_snapshot(
    target_date,
    status,
    forecast_high,
    forecast_low,
    bias_high,
    bias_low,
    predictions,
    market_prices,
    high_bucket_defs,
    low_bucket_defs,
):
    """Build a structured snapshot from existing prediction outputs."""
    try:
        from zoneinfo import ZoneInfo
        generated_at = datetime.now(ZoneInfo("Asia/Hong_Kong"))
    except Exception:
        generated_at = datetime.now()

    def _contract_label(bucket_label):
        bucket_text = str(bucket_label).strip()
        match = re.search(r"(\d+)", bucket_text)
        if match:
            return f"{int(match.group(1))}°C"
        return bucket_text

    def _market_section(date_str, side):
        key = f"{date_str}_{'highest' if side == 'high' else 'lowest'}"
        return market_prices.get(key, {}).get("prices", {})

    def _contract_market_price(label, market_map):
        if label in market_map:
            return market_map[label]
        label_num = re.search(r"(\d+)", label)
        label_num = label_num.group(1) if label_num else None
        if label_num:
            for market_label, price in market_map.items():
                market_text = str(market_label)
                market_num = re.search(r"(\d+)", market_text)
                if market_num and market_num.group(1) == label_num:
                    return price
        return None

    def _bucket_sort_key(bucket_def):
        match = re.search(r"(\d+)", str(bucket_def[0]))
        return int(match.group(1)) if match else 0

    date_str = target_date.strftime("%Y-%m-%d")
    recommendations = find_recommended_bets(predictions, market_prices, date_str)
    high_probs = predictions.get("high_probs", {})
    low_probs = predictions.get("low_probs", {})
    high_market = _market_section(date_str, "high")
    low_market = _market_section(date_str, "low")

    def _pick_best(side_key, kind):
        bet = recommendations.get(side_key, {}).get(kind)
        if not bet:
            return None
        return _contract_label(bet["bucket"])

    def _contract_entry(label, side_key, market_map, probs, main_label, lottery_label):
        model_prob = probs.get(label)
        if model_prob is None:
            for key, value in probs.items():
                if _contract_label(key) == label:
                    model_prob = value
                    break
        model_prob = model_prob if model_prob is not None else 0.0
        market_prob = _contract_market_price(label, market_map)
        market_prob = market_prob if market_prob is not None else 0.0
        edge = model_prob - market_prob
        if label == main_label:
            classification = "Main Bet"
        elif label == lottery_label:
            classification = "Lottery Bet"
        elif edge > 0.05:
            classification = "Other Value Bet"
        else:
            classification = "Not Recommended"

        kelly = 0.0
        for kind in ("main", "lottery"):
            bet = recommendations.get(side_key, {}).get(kind)
            if bet and _contract_label(bet["bucket"]) == label:
                kelly = bet.get("kelly_stake", 0.0)
                break

        return {
            "contract": label,
            "model_probability": model_prob,
            "market_probability": market_prob,
            "edge": edge,
            "ev": edge,
            "kelly": kelly,
            "classification": classification,
        }

    high_main_label = _pick_best("high", "main")
    high_lottery_label = _pick_best("high", "lottery")
    low_main_label = _pick_best("low", "main")
    low_lottery_label = _pick_best("low", "lottery")

    high_contracts = {}
    for bucket_def in sorted(high_bucket_defs, key=_bucket_sort_key):
        label = _contract_label(bucket_def[0])
        high_contracts[label] = _contract_entry(
            label,
            "high",
            high_market,
            high_probs,
            high_main_label,
            high_lottery_label,
        )

    low_contracts = {}
    for bucket_def in sorted(low_bucket_defs, key=_bucket_sort_key):
        label = _contract_label(bucket_def[0])
        low_contracts[label] = _contract_entry(
            label,
            "low",
            low_market,
            low_probs,
            low_main_label,
            low_lottery_label,
        )

    return {
        "schema_version": 1,
        "snapshot_id": generated_at.strftime("%Y%m%d_%H%M%S_%f"),
        "timestamp": generated_at.isoformat(),
        "target_date": date_str,
        "status": status,
        "forecast": {
            "high": forecast_high,
            "low": forecast_low,
            "bias_high": bias_high.get("correction") if isinstance(bias_high, dict) else bias_high,
            "bias_low": bias_low.get("correction") if isinstance(bias_low, dict) else bias_low,
        },
        "recommendations": {
            "high": {
                "main_contract": high_main_label,
                "lottery_contract": high_lottery_label,
                "model_probability": (recommendations.get("high", {}).get("main") or {}).get("model"),
                "market_probability": (recommendations.get("high", {}).get("main") or {}).get("market"),
                "edge": (recommendations.get("high", {}).get("main") or {}).get("edge"),
                "ev": (recommendations.get("high", {}).get("main") or {}).get("edge"),
                "kelly": (recommendations.get("high", {}).get("main") or {}).get("kelly_stake", 0.0),
            },
            "low": {
                "main_contract": low_main_label,
                "lottery_contract": low_lottery_label,
                "model_probability": (recommendations.get("low", {}).get("main") or {}).get("model"),
                "market_probability": (recommendations.get("low", {}).get("main") or {}).get("market"),
                "edge": (recommendations.get("low", {}).get("main") or {}).get("edge"),
                "ev": (recommendations.get("low", {}).get("main") or {}).get("edge"),
                "kelly": (recommendations.get("low", {}).get("main") or {}).get("kelly_stake", 0.0),
            },
        },
        "market": {
            "high": high_market,
            "low": low_market,
        },
        "probabilities": {
            "high": high_probs,
            "low": low_probs,
        },
        "contracts": {
            "high": high_contracts,
            "low": low_contracts,
        },
    }


def save_prediction_snapshot(snapshot):
    """Append one snapshot to the JSONL history file."""
    snapshot_path = PROCESSED_DATA_DIR / "prediction_snapshots.jsonl"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")


def _underpriced_alert_seen(date_str, hour_key, side):
    """Check whether an underpriced alert was already sent for this hour and side."""
    state_path = PROCESSED_DATA_DIR / "underpriced_alerts.jsonl"
    if not state_path.exists():
        return False
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("target_date") == date_str and record.get("hour_key") == hour_key and record.get("side") == side and record.get("alerted"):
                    return True
    except OSError:
        return False
    return False


def _last_underpriced_edge(date_str, hour_key, side):
    """Return the previous recorded edge for the same date/side before this hour."""
    state_path = PROCESSED_DATA_DIR / "underpriced_alerts.jsonl"
    if not state_path.exists():
        return None
    previous = None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("target_date") != date_str or record.get("side") != side:
                    continue
                record_hour = record.get("hour_key")
                if not record_hour or record_hour >= hour_key:
                    continue
                edge = record.get("edge")
                if edge is None:
                    continue
                previous = edge
    except OSError:
        return None
    return previous


def _save_underpriced_alert(date_str, hour_key, side, edge, alerted):
    """Append one underpriced-alert evaluation record to the history file."""
    state_path = PROCESSED_DATA_DIR / "underpriced_alerts.jsonl"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "target_date": date_str,
        "hour_key": hour_key,
        "side": side,
        "edge": edge,
        "alerted": alerted,
        "timestamp": datetime.now().isoformat(),
    }
    with open(state_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(target_dates=None):
    """
    Main execution: predict + fetch prices + send Telegram.
    target_dates: optional list of date objects. Defaults to today (if unresolved) + tomorrow.
    """
    print("=" * 60)
    print("  HK Weather Prediction Alert v2")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    today = datetime.now().date()

    # Collect shared data once
    print("\n[1/7] Fetching recent temperature data...")
    recent = get_recent_data(365)
    print(f"  {len(recent)} days of data loaded\n")

    print("[2/7] Fetching HKO 9-day forecast...")
    forecasts = get_hko_forecast()
    print(f"  {len(forecasts)} days of forecast loaded\n")

    print("[3/7] Fetching ensemble NWP forecast...")
    ensemble_nwp = get_ensemble_nwp_forecast()
    if ensemble_nwp:
        print(f"  Ensemble NWP loaded: {len(ensemble_nwp['max']['members'])} members")
        print(f"  Max temp: {ensemble_nwp['max']['mean']:.1f}°C ± {ensemble_nwp['max']['std']:.1f}°C")
        print(f"  Min temp: {ensemble_nwp['min']['mean']:.1f}°C ± {ensemble_nwp['min']['std']:.1f}°C")
    else:
        print("  Ensemble NWP not available, using HKO only\n")

    print("[5/7] Fetching Polymarket market prices...")
    market_prices = get_market_prices()
    print(f"  {len(market_prices)} events found\n")

    # Determine target dates
    if target_dates is None:
        target_dates = []
        for offset in [0, 1]:
            d = today + timedelta(days=offset)
            ds = d.strftime("%Y-%m-%d")
            if f"{ds}_highest" in market_prices or f"{ds}_lowest" in market_prices:
                target_dates.append(d)

    if not target_dates:
        print("  No active Polymarket events found for requested date(s).")
        return

    for target_date in target_dates:
        date_str = target_date.strftime("%Y-%m-%d")
        print(f"\n{'='*60}")
        print(f"  Processing: {date_str}")
        print(f"{'='*60}")

        # Check if already resolved
        print(f"\n[4/7] Checking if event is resolved...")
        actual_max, actual_min = get_actual_temperature(target_date)
        if actual_max is not None:
            print(f"  RESOLVED: High {actual_max}°C, Low {actual_min}°C")
            update_prediction(date_str, "high", actual_max)
            update_prediction(date_str, "low", actual_min)
        else:
            print(f"  Event not yet resolved\n")

        # Extract dynamic bucket ranges for this date
        print("[6/7] Extracting dynamic bucket ranges...")
        high_bucket_defs = extract_dynamic_buckets(market_prices, date_str, "highest")
        low_bucket_defs = extract_dynamic_buckets(market_prices, date_str, "lowest")
        print(f"  HIGH buckets: {len(high_bucket_defs)} ({high_bucket_defs[0][0]} to {high_bucket_defs[-1][0]})")
        print(f"  LOW buckets: {len(low_bucket_defs)} ({low_bucket_defs[0][0]} to {low_bucket_defs[-1][0]})\n")

        # Generate predictions
        print("[7/7] Generating predictions...")
        
        # Find HKO forecast for target date
        target_forecast = None
        for fc in forecasts:
            if fc["date"].date() == target_date:
                target_forecast = fc
                break
        
        # Blend HKO forecast with ensemble NWP if available
        if target_forecast:
            hko_max = target_forecast["max_temp"]
            hko_min = target_forecast["min_temp"]
            
            if ensemble_nwp:
                max_fc = 0.6 * hko_max + 0.4 * ensemble_nwp["max"]["mean"]
                min_fc = 0.6 * hko_min + 0.4 * ensemble_nwp["min"]["mean"]
                print(f"  Blended Forecast: High {max_fc:.1f}°C / Low {min_fc:.1f}°C (HKO + NWP)")
            else:
                max_fc = hko_max
                min_fc = hko_min
                print(f"  HKO Forecast: High {max_fc}°C / Low {min_fc}°C")
        else:
            max_fc = None
            min_fc = None
            print(f"  No forecast available for {date_str}")

        # Apply range-aware bias correction (uses predicted_mean to pick the band)
        print("\n  Applying bias correction (range-aware)...")
        bias_high = get_bias_correction("high", predicted_mean=max_fc if max_fc is not None else None)
        bias_low = get_bias_correction("low", predicted_mean=min_fc if min_fc is not None else None)

        if max_fc is not None and bias_high.get("n_samples", 0) > 0:
            max_fc = float(max_fc) + bias_high["correction"]
            src = bias_high.get("source", "global")
            rng = bias_high.get("range_label") or "-"
            print(
                f"  HIGH correction: {bias_high['correction']:+.2f}°C "
                f"(source={src}, range={rng}, n={bias_high['n_samples']}, {bias_high['confidence']})"
            )
        if min_fc is not None and bias_low.get("n_samples", 0) > 0:
            min_fc = float(min_fc) + bias_low["correction"]
            src = bias_low.get("source", "global")
            rng = bias_low.get("range_label") or "-"
            print(
                f"  LOW correction: {bias_low['correction']:+.2f}°C "
                f"(source={src}, range={rng}, n={bias_low['n_samples']}, {bias_low['confidence']})"
            )
        if bias_high.get("n_samples", 0) == 0 and bias_low.get("n_samples", 0) == 0:
            print("  No bias data yet (will start learning after first resolved event)")
        
        # Predict high temp buckets
        high_probs, high_mean, high_std = predict_buckets(
            'max', recent, target_date, forecast_temp=max_fc, bucket_defs=high_bucket_defs
        )
        # Predict low temp buckets
        low_probs, low_mean, low_std = predict_buckets(
            'min', recent, target_date, forecast_temp=min_fc, bucket_defs=low_bucket_defs
        )

        predictions = {
            "forecast": {"max_temp": max_fc, "min_temp": min_fc},
            "high_probs": high_probs,
            "low_probs": low_probs,
            "high_mean": high_mean,
            "low_mean": low_mean,
            "ensemble_nwp": ensemble_nwp,
        }

        best_high = max(high_probs, key=high_probs.get)
        best_low = max(low_probs, key=low_probs.get)
        print(f"\n  Prediction:")
        print(f"    High: {best_high} ({high_probs[best_high]:.0%})")
        print(f"    Low: {best_low} ({low_probs[best_low]:.0%})")

        # Format message with date label
        actual_temps = (actual_max, actual_min) if actual_max is not None else None
        message = format_message(target_date, predictions, market_prices, actual_temps)

        snapshot = build_prediction_snapshot(
            target_date=target_date,
            status="RESOLVED" if actual_max is not None else "PENDING",
            forecast_high=max_fc,
            forecast_low=min_fc,
            bias_high=bias_high,
            bias_low=bias_low,
            predictions=predictions,
            market_prices=market_prices,
            high_bucket_defs=high_bucket_defs,
            low_bucket_defs=low_bucket_defs,
        )
        save_prediction_snapshot(snapshot)
        print(f"  [OK] Snapshot saved: {snapshot['snapshot_id']}")
        
        # Log predictions
        if actual_max is None:
            best_bets = find_recommended_bets(predictions, market_prices, date_str)
            log_prediction(date_str, "high", high_probs, best_bets.get("high", {}))
            log_prediction(date_str, "low", low_probs, best_bets.get("low", {}))
            print("\n  [OK] Predictions logged for performance tracking")

            alert_lines = []
            hour_key = datetime.now().strftime("%Y-%m-%d-%H")
            for side, label in [("high", "HIGH"), ("low", "LOW")]:
                main_bet = best_bets.get(side, {}).get("main")
                current_edge = float(main_bet.get("edge", 0)) if main_bet else 0.0
                previous_edge = _last_underpriced_edge(date_str, hour_key, side)
                previous_edge = float(previous_edge) if previous_edge is not None else None
                should_alert = (
                    main_bet
                    and current_edge > 0.03
                    and (previous_edge is not None and current_edge > previous_edge)
                    and not _underpriced_alert_seen(date_str, hour_key, side)
                )
                _save_underpriced_alert(date_str, hour_key, side, current_edge, should_alert)
                if should_alert and main_bet is not None:
                    bucket = main_bet.get("bucket", "")
                    market_cents = main_bet.get("market", 0) * 100
                    alert_lines.append(
                        f"🟢 UNDERPRICED ALERT {label} {bucket} @ {market_cents:.0f}¢ | Edge {current_edge:+.1%}"
                    )

            if alert_lines:
                send_telegram("\n".join(alert_lines))
                print("  [OK] Underpriced alert sent")
        
        if actual_max is not None:
            log_forecast_error(date_str, "high", high_mean, actual_max)
            log_forecast_error(date_str, "low", low_mean, actual_min)
            print("  [OK] Forecast errors logged for bias correction")

        try:
            print(f"\n--- Message Preview ({date_str}) ---")
            print(message.encode('utf-8', errors='replace').decode('utf-8'))
            print(f"--- End Preview ---\n")
        except Exception:
            print("\n--- Message Preview (encoding skipped) ---\n")

        send_telegram(message)
        time.sleep(1)  # Small delay between sends


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Specific date mode: python telegram_alert.py 2026-08-10
        from datetime import datetime as _dt
        run(target_dates=[_dt.strptime(sys.argv[1], "%Y-%m-%d").date()])
    else:
        run()
