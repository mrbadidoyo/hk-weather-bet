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
    Buckets shift per day based on expected temperatures.
    
    Returns list of (label, lo, hi) tuples.
    """
    key = f"{date_str}_{temp_type}"
    if key not in market_prices:
        # Fallback to default buckets
        return DEFAULT_HIGH_TEMP_BUCKETS if temp_type == "highest" else DEFAULT_LOW_TEMP_BUCKETS
    
    prices = market_prices[key]["prices"]
    buckets = []
    
    for bucket_label in prices.keys():
        # Parse bucket label like "34°C", "27°C or below", "37°C or higher"
        match = re.search(r'(\d+)', bucket_label)
        if not match:
            continue
        
        temp = int(match.group(1))
        
        if "below" in bucket_label.lower():
            # "27°C or below" → (-999, 28)
            buckets.append((bucket_label, -999, temp + 1))
        elif "higher" in bucket_label.lower():
            # "37°C or higher" → (37, 999)
            buckets.append((bucket_label, temp, 999))
        else:
            # "34°C" → (34, 35)
            buckets.append((bucket_label, temp, temp + 1))
    
    # Sort by lower bound
    buckets.sort(key=lambda x: x[1])
    
    return buckets if buckets else (DEFAULT_HIGH_TEMP_BUCKETS if temp_type == "highest" else DEFAULT_LOW_TEMP_BUCKETS)


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
        logger.warning(f"Error fetching ensemble NWP: {e}")
        return None


# ── Message Formatting ───────────────────────────────────────────

def format_message(target_date, predictions, market_prices, actual_temps):
    """Format prediction + market data into Telegram message."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_str = target_date.strftime("%Y-%m-%d")
    
    lines = [f"🌡️ *HK Weather Alert* — {now}", ""]
    lines.append(f"📅 *Target: {date_str}*")
    
    # Check if resolved
    is_resolved, _ = check_if_resolved(market_prices, date_str)
    
    if is_resolved and actual_temps:
        lines.append("✅ *RESOLVED*")
        lines.append(f"Actual: High {actual_temps[0]}°C / Low {actual_temps[1]}°C")
    else:
        lines.append("⏳ *PENDING* (monitoring until resolved)")
        forecast = predictions.get("forecast", {})
        lines.append(f"HKO Forecast: High {forecast.get('max_temp', '?')}°C / Low {forecast.get('min_temp', '?')}°C")
    
    lines.append("")

    # High temp
    lines.append("🔴 *HIGH Temperature:*")
    high_probs = predictions.get("high_probs", {})
    sorted_high = sorted(high_probs.items(), key=lambda x: x[1], reverse=True)

    market_key = f"{date_str}_highest"
    market_high = market_prices.get(market_key, {}).get("prices", {})

    # Determine winning bucket if resolved
    winning_high = None
    if actual_temps and actual_temps[0] is not None:
        actual_max = actual_temps[0]
        for label, lo, hi in DEFAULT_HIGH_TEMP_BUCKETS:
            if lo <= actual_max < hi:
                winning_high = label
                break

    for bucket, prob in sorted_high[:7]:
        market_price = None
        for mk, mp in market_high.items():
            if bucket.split("°")[0] in mk or mk.split("°")[0] in bucket:
                market_price = mp
                break

        is_winner = bucket == winning_high if winning_high else False
        
        if market_price is not None:
            edge = prob - market_price
            if is_winner:
                edge_str = f"🏆 WINNER"
            elif edge > 0.05:
                edge_str = f"✅ {edge:+.1%}"
            else:
                edge_str = f"⚪ {edge:+.1%}"
            lines.append(f"  {bucket:<18} Model: {prob:.0%} | Market: {market_price:.0%} | {edge_str}")
        else:
            if is_winner:
                lines.append(f"  {bucket:<18} Model: {prob:.0%} | 🏆 WINNER")
            else:
                lines.append(f"  {bucket:<18} Model: {prob:.0%}")

    lines.append("")

    # Low temp
    lines.append("🔵 *LOW Temperature:*")
    low_probs = predictions.get("low_probs", {})
    sorted_low = sorted(low_probs.items(), key=lambda x: x[1], reverse=True)

    market_key = f"{date_str}_lowest"
    market_low = market_prices.get(market_key, {}).get("prices", {})

    winning_low = None
    if actual_temps and actual_temps[1] is not None:
        actual_min = actual_temps[1]
        for label, lo, hi in DEFAULT_LOW_TEMP_BUCKETS:
            if lo <= actual_min < hi:
                winning_low = label
                break

    for bucket, prob in sorted_low[:7]:
        market_price = None
        for mk, mp in market_low.items():
            if bucket.split("°")[0] in mk or mk.split("°")[0] in bucket:
                market_price = mp
                break

        is_winner = bucket == winning_low if winning_low else False
        
        if market_price is not None:
            edge = prob - market_price
            if is_winner:
                edge_str = f"🏆 WINNER"
            elif edge > 0.05:
                edge_str = f"✅ {edge:+.1%}"
            else:
                edge_str = f"⚪ {edge:+.1%}"
            lines.append(f"  {bucket:<18} Model: {prob:.0%} | Market: {market_price:.0%} | {edge_str}")
        else:
            if is_winner:
                lines.append(f"  {bucket:<18} Model: {prob:.0%} | 🏆 WINNER")
            else:
                lines.append(f"  {bucket:<18} Model: {prob:.0%}")

    lines.append("")
    
    # RECOMMENDED BETS (only if not resolved)
    best_bets = {"high": {"main": None, "lottery": None}, "low": {"main": None, "lottery": None}}
    if not is_resolved:
        best_bets = find_recommended_bets(predictions, market_prices, date_str)
        
        lines.append("🎯 *RECOMMENDED BETS:*")
        lines.append("")
        
        # HIGH recommendations
        lines.append("  🔴 *HIGH:*")
        if best_bets["high"]["main"]:
            b = best_bets["high"]["main"]
            kelly = b.get("kelly_stake", 0)
            # Check if main bet is underpriced
            if b["edge"] > 0:
                lines.append(f"     📌 Main: {b['bucket']} @ {b['market']:.0%} 💎")
                lines.append(f"        Model: {b['model']:.0%} | Edge: {b['edge']:+.1%} | UNDERPRICED!")
                if kelly > 0:
                    lines.append(f"        💵 Kelly stake: ${kelly:.2f}")
            else:
                lines.append(f"     📌 Main: {b['bucket']} @ {b['market']:.0%}")
                lines.append(f"        Model: {b['model']:.0%} | Edge: {b['edge']:+.1%}")
        else:
            lines.append("     📌 Main: No suitable bet")
        
        if best_bets["high"]["lottery"]:
            b = best_bets["high"]["lottery"]
            kelly = b.get("kelly_stake", 0)
            lines.append(f"     🎰 Lottery: {b['bucket']} @ {b['market']:.0%}")
            lines.append(f"        Model: {b['model']:.0%} | Edge: {b['edge']:+.1%} | {b['confidence']}")
            if kelly > 0:
                lines.append(f"        💵 Kelly stake: ${kelly:.2f}")
        else:
            lines.append("     🎰 Lottery: No value bet found")
        
        lines.append("")
        
        # LOW recommendations
        lines.append("  🔵 *LOW:*")
        if best_bets["low"]["main"]:
            b = best_bets["low"]["main"]
            kelly = b.get("kelly_stake", 0)
            # Check if main bet is underpriced
            if b["edge"] > 0:
                lines.append(f"     📌 Main: {b['bucket']} @ {b['market']:.0%} 💎")
                lines.append(f"        Model: {b['model']:.0%} | Edge: {b['edge']:+.1%} | UNDERPRICED!")
                if kelly > 0:
                    lines.append(f"        💵 Kelly stake: ${kelly:.2f}")
            else:
                lines.append(f"     📌 Main: {b['bucket']} @ {b['market']:.0%}")
                lines.append(f"        Model: {b['model']:.0%} | Edge: {b['edge']:+.1%}")
        else:
            lines.append("     📌 Main: No suitable bet")
        
        if best_bets["low"]["lottery"]:
            b = best_bets["low"]["lottery"]
            kelly = b.get("kelly_stake", 0)
            lines.append(f"     🎰 Lottery: {b['bucket']} @ {b['market']:.0%}")
            lines.append(f"        Model: {b['model']:.0%} | Edge: {b['edge']:+.1%} | {b['confidence']}")
            if kelly > 0:
                lines.append(f"        💵 Kelly stake: ${kelly:.2f}")
        else:
            lines.append("     🎰 Lottery: No value bet found")
        
        lines.append("")
        lines.append("  _📌 Main = sesuai forecast | 🎰 Lottery = underpriced | 💎 = main also underpriced_")
        lines.append("  _💵 Kelly stake = optimal bet size (quarter-Kelly, bankroll=${})_".format(int(BANKROLL)))
        lines.append("")

    # Other value bets
    if not is_resolved:
        lines.append("💰 *Other Value Bets:*")
        all_bets = find_best_bets(predictions, market_prices, date_str)
        # Exclude the recommended bets
        other_bets = []
        for b in all_bets:
            is_recommended = False
            if best_bets["high"]["main"] and b["bucket"] == best_bets["high"]["main"]["bucket"] and b["type"] == "HIGH":
                is_recommended = True
            if best_bets["high"]["lottery"] and b["bucket"] == best_bets["high"]["lottery"]["bucket"] and b["type"] == "HIGH":
                is_recommended = True
            if best_bets["low"]["main"] and b["bucket"] == best_bets["low"]["main"]["bucket"] and b["type"] == "LOW":
                is_recommended = True
            if best_bets["low"]["lottery"] and b["bucket"] == best_bets["low"]["lottery"]["bucket"] and b["type"] == "LOW":
                is_recommended = True
            if not is_recommended:
                other_bets.append(b)
        
        if other_bets:
            for bet in other_bets[:4]:
                lines.append(f"  {bet['emoji']} {bet['bucket']} — "
                            f"Model: {bet['model']:.0%} vs Market: {bet['market']:.0%} "
                            f"(Edge: {bet['edge']:+.1%})")
        else:
            lines.append("  Tidak ada value bet lain.")

    lines.append("")
    lines.append("_🎯 = recommended | Edge = Model - Market_")
    
    # Add bias correction summary
    from bias_corrector import get_bias_report
    bias_report = get_bias_report()
    bias_parts = []
    for ttype in ["high", "low"]:
        b = bias_report[ttype]
        if b["n_samples"] > 0:
            bias_parts.append(f"{ttype.upper()}: {b['correction']:+.2f}°C")
    if bias_parts:
        lines.append(f"_Bias correction: {' | '.join(bias_parts)} (n={bias_report['high']['n_samples']+bias_report['low']['n_samples']})_")

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


def run():
    """Main execution: predict tomorrow + fetch prices + send Telegram."""
    print("=" * 60)
    print("  HK Weather Prediction Alert v2")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Target: tomorrow only
    today = datetime.now().date()
    target_date = today + timedelta(days=1)
    date_str = target_date.strftime("%Y-%m-%d")
    
    print(f"\n  Target date: {date_str}")

    # Get recent data (1 year for DOY matching)
    print("\n[1/7] Fetching recent temperature data...")
    recent = get_recent_data(365)
    print(f"  {len(recent)} days of data loaded")

    # Get HKO forecast
    print("\n[2/7] Fetching HKO 9-day forecast...")
    forecasts = get_hko_forecast()
    print(f"  {len(forecasts)} days of forecast loaded")

    # Improvement #5: Get ensemble NWP forecast
    print("\n[3/7] Fetching ensemble NWP forecast...")
    ensemble_nwp = get_ensemble_nwp_forecast()
    if ensemble_nwp:
        print(f"  Ensemble NWP loaded: {len(ensemble_nwp['max']['members'])} members")
        print(f"  Max temp: {ensemble_nwp['max']['mean']:.1f}°C ± {ensemble_nwp['max']['std']:.1f}°C")
        print(f"  Min temp: {ensemble_nwp['min']['mean']:.1f}°C ± {ensemble_nwp['min']['std']:.1f}°C")
    else:
        print("  Ensemble NWP not available, using HKO only")

    # Check if already resolved (actual temperature available)
    print("\n[4/7] Checking if event is resolved...")
    actual_max, actual_min = get_actual_temperature(target_date)
    if actual_max is not None:
        print(f"  RESOLVED: High {actual_max}°C, Low {actual_min}°C")
        # Update performance tracker
        if actual_max is not None:
            update_prediction(date_str, "high", actual_max)
        if actual_min is not None:
            update_prediction(date_str, "low", actual_min)
        
        # Improvement #6: Log forecast errors for bias correction
        # We need yesterday's prediction mean to compute the error
        # Use the blended forecast from the prediction
    else:
        print(f"  Event not yet resolved")

    # Get Polymarket prices
    print("\n[5/7] Fetching Polymarket market prices...")
    market_prices = get_market_prices()
    print(f"  {len(market_prices)} events found")

    # Improvement #1: Extract dynamic bucket ranges
    print("\n[6/7] Extracting dynamic bucket ranges...")
    high_bucket_defs = extract_dynamic_buckets(market_prices, date_str, "highest")
    low_bucket_defs = extract_dynamic_buckets(market_prices, date_str, "lowest")
    print(f"  HIGH buckets: {len(high_bucket_defs)} ({high_bucket_defs[0][0]} to {high_bucket_defs[-1][0]})")
    print(f"  LOW buckets: {len(low_bucket_defs)} ({low_bucket_defs[0][0]} to {low_bucket_defs[-1][0]})")

    # Generate predictions for tomorrow
    print("\n[7/7] Generating predictions...")
    
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
            # 60% HKO + 40% Ensemble NWP mean
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

    # Improvement #6: Apply bias correction from recent errors
    print("\n  Applying bias correction...")
    bias_high = get_bias_correction("high")
    bias_low = get_bias_correction("low")
    if max_fc is not None and bias_high["n_samples"] > 0:
        max_fc += bias_high["correction"]
        print(f"  HIGH correction: {bias_high['correction']:+.2f}°C (n={bias_high['n_samples']}, {bias_high['confidence']})")
    if min_fc is not None and bias_low["n_samples"] > 0:
        min_fc += bias_low["correction"]
        print(f"  LOW correction: {bias_low['correction']:+.2f}°C (n={bias_low['n_samples']}, {bias_low['confidence']})")
    if bias_high["n_samples"] == 0 and bias_low["n_samples"] == 0:
        print("  No bias data yet (will start learning after first resolved event)")
    
    # Predict high temp buckets using dynamic bucket definitions
    high_probs, high_mean, high_std = predict_buckets(
        'max', recent, target_date, forecast_temp=max_fc, bucket_defs=high_bucket_defs
    )
    # Predict low temp buckets using dynamic bucket definitions
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

    # Format and send message
    actual_temps = (actual_max, actual_min) if actual_max is not None else None
    message = format_message(target_date, predictions, market_prices, actual_temps)
    
    # Improvement #4: Log predictions for performance tracking
    if actual_max is None:  # Only log if not resolved yet
        best_bets = find_recommended_bets(predictions, market_prices, date_str)
        log_prediction(date_str, "high", high_probs, best_bets.get("high", {}))
        log_prediction(date_str, "low", low_probs, best_bets.get("low", {}))
        print("\n  [OK] Predictions logged for performance tracking")
    
    # Improvement #6: Log forecast errors for resolved events (bias learning)
    if actual_max is not None:
        log_forecast_error(date_str, "high", high_mean, actual_max)
        log_forecast_error(date_str, "low", low_mean, actual_min)
        print("  [OK] Forecast errors logged for bias correction")

    try:
        print(f"\n--- Message Preview ---")
        print(message.encode('utf-8', errors='replace').decode('utf-8'))
        print(f"--- End Preview ---\n")
    except Exception:
        print("\n--- Message Preview (encoding skipped) ---\n")

    send_telegram(message)


if __name__ == "__main__":
    run()
