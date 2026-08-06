"""
Telegram Prediction Alert — Runs hourly, monitors tomorrow's HK temperature event.
Sends predictions vs Polymarket prices until event is resolved.
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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Telegram config
TELEGRAM_BOT_TOKEN = "8909269274:AAF5XoUwEh9zZXekHniWFVVwUTijEn9Vz-4"
TELEGRAM_CHAT_ID = "225257336"


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

def predict_buckets(temp_type, recent_data, target_date, forecast_temp=None):
    """
    Predict bucket probabilities for target_date using historical distribution + forecast.
    """
    col = f'{temp_type}_temp'
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
    
    # Best value bets (only if not resolved)
    if not is_resolved:
        lines.append("💰 *VALUE BETS (edge >5%):*")
        best_bets = find_best_bets(predictions, market_prices, date_str)
        if best_bets:
            for bet in best_bets[:5]:
                lines.append(f"  {bet['emoji']} {bet['bucket']} — "
                            f"Model: {bet['model']:.0%} vs Market: {bet['market']:.0%} "
                            f"(Edge: {bet['edge']:+.1%})")
        else:
            lines.append("  No strong value bets found.")

    lines.append("")
    lines.append("_✅ = edge >5% | 🏆 = winning bucket_")

    return "\n".join(lines)


def find_best_bets(predictions, market_prices, date_str):
    """Find bets where model probability > market price by significant margin."""
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
    print("  HK Weather Prediction Alert")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Target: tomorrow only
    today = datetime.now().date()
    target_date = today + timedelta(days=1)
    date_str = target_date.strftime("%Y-%m-%d")
    
    print(f"\n  Target date: {date_str}")

    # Get recent data (1 year for DOY matching)
    print("\n[1/5] Fetching recent temperature data...")
    recent = get_recent_data(365)
    print(f"  {len(recent)} days of data loaded")

    # Get HKO forecast
    print("\n[2/5] Fetching HKO 9-day forecast...")
    forecasts = get_hko_forecast()
    print(f"  {len(forecasts)} days of forecast loaded")

    # Check if already resolved (actual temperature available)
    print("\n[3/5] Checking if event is resolved...")
    actual_max, actual_min = get_actual_temperature(target_date)
    if actual_max is not None:
        print(f"  RESOLVED: High {actual_max}°C, Low {actual_min}°C")
    else:
        print(f"  Event not yet resolved")

    # Get Polymarket prices
    print("\n[4/5] Fetching Polymarket market prices...")
    market_prices = get_market_prices()
    print(f"  {len(market_prices)} events found")

    # Generate predictions for tomorrow
    print("\n[5/5] Generating predictions...")
    
    # Find forecast for target date
    target_forecast = None
    for fc in forecasts:
        if fc["date"].date() == target_date:
            target_forecast = fc
            break
    
    if target_forecast:
        max_fc = target_forecast["max_temp"]
        min_fc = target_forecast["min_temp"]
        print(f"  HKO Forecast: High {max_fc}°C / Low {min_fc}°C")
    else:
        max_fc = None
        min_fc = None
        print(f"  No HKO forecast available for {date_str}")

    # Predict high temp buckets
    high_probs, high_mean, high_std = predict_buckets(
        'max', recent, target_date, forecast_temp=max_fc
    )
    # Predict low temp buckets
    low_probs, low_mean, low_std = predict_buckets(
        'min', recent, target_date, forecast_temp=min_fc
    )

    predictions = {
        "forecast": {"max_temp": max_fc, "min_temp": min_fc},
        "high_probs": high_probs,
        "low_probs": low_probs,
        "high_mean": high_mean,
        "low_mean": low_mean,
    }

    best_high = max(high_probs, key=high_probs.get)
    best_low = max(low_probs, key=low_probs.get)
    print(f"\n  Prediction:")
    print(f"    High: {best_high} ({high_probs[best_high]:.0%})")
    print(f"    Low: {best_low} ({low_probs[best_low]:.0%})")

    # Format and send message
    actual_temps = (actual_max, actual_min) if actual_max is not None else None
    message = format_message(target_date, predictions, market_prices, actual_temps)
    
    try:
        print(f"\n--- Message Preview ---")
        print(message.encode('utf-8', errors='replace').decode('utf-8'))
        print(f"--- End Preview ---\n")
    except Exception:
        print("\n--- Message Preview (encoding skipped) ---\n")

    send_telegram(message)


if __name__ == "__main__":
    run()
