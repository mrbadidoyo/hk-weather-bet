"""
HK Weather Prediction Dashboard - Streamlit GUI
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DEFAULT_HIGH_TEMP_BUCKETS,
    DEFAULT_LOW_TEMP_BUCKETS,
    MODELS_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESOLUTION_STATION,
    RESOLUTION_SOURCE_URL,
)
from data_collector import HKODataCollector
from nwp_collector import NWPCollector, blend_probs
from polymarket_strategy import (
    analyze_market,
    compute_bucket_probabilities,
    parse_buckets,
)

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="HK Weather Bet",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric { background: rgba(30,41,59,.45); border-radius: 10px; padding: 12px 16px; }
    .bet-buy  { background:#0d4f2e; color:#4ade80; padding:6px 14px; border-radius:6px; font-weight:700; }
    .bet-skip { background:#3b3b3b; color:#d4d4d4; padding:6px 14px; border-radius:6px; }
    .bet-sell { background:#4f1717; color:#f87171; padding:6px 14px; border-radius:6px; font-weight:700; }
    div[data-testid="stHorizontalBlock"] > div { min-width: 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner="Fetching HKO forecast...")
def fetch_hko_forecast():
    hko = HKODataCollector()
    return hko.fetch_9day_forecast()


@st.cache_data(ttl=600, show_spinner="Fetching current weather...")
def fetch_current_weather():
    hko = HKODataCollector()
    return hko.fetch_current_weather()


def load_market_prices():
    path = RAW_DATA_DIR / "market_prices.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_market_prices(prices: dict):
    path = RAW_DATA_DIR / "market_prices.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prices, f, indent=2)


def _normalize_bucket_key(label: str) -> str:
    """Normalize bucket label for fuzzy matching across sources."""
    import re
    s = (label or "").lower().replace("°", "").replace("c", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def map_live_prices_to_defaults(live_prices: dict, default_buckets: list) -> dict:
    """
    Map Polymarket live prices onto DEFAULT_* bucket labels.
    Tries exact match, then normalized / numeric match.
    """
    import re
    mapped = {}
    live_by_norm = {_normalize_bucket_key(k): float(v) for k, v in live_prices.items()}

    for label, lo, hi in default_buckets:
        # 1) exact
        if label in live_prices:
            mapped[label] = float(live_prices[label])
            continue
        # 2) normalized
        norm = _normalize_bucket_key(label)
        if norm in live_by_norm:
            mapped[label] = live_by_norm[norm]
            continue
        # 3) numeric core match (e.g. "34" inside both labels)
        m = re.search(r"(\d+)", label)
        if m:
            num = m.group(1)
            for lk, lv in live_prices.items():
                if re.search(rf"\b{num}\b", str(lk)):
                    # Prefer same open-ended type
                    lab_low = "below" in label.lower()
                    lab_high = "higher" in label.lower()
                    live_low = "below" in str(lk).lower()
                    live_high = "higher" in str(lk).lower()
                    if lab_low == live_low and lab_high == live_high:
                        mapped[label] = float(lv)
                        break
            else:
                # last resort: any label containing the number
                for lk, lv in live_prices.items():
                    if num in str(lk):
                        mapped[label] = float(lv)
                        break

    return mapped


def build_polymarket_event_url(date_iso: str, temp_type: str) -> str:
    """
    Build Polymarket event URL for HK temperature market.

    temp_type: 'highest' or 'lowest' (also accepts 'high' / 'low')
    """
    MONTHS = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    t = temp_type.lower()
    if t in ("high", "max"):
        t = "highest"
    elif t in ("low", "min"):
        t = "lowest"

    try:
        dt = datetime.strptime(date_iso, "%Y-%m-%d")
    except ValueError:
        return "https://polymarket.com/"

    month_name = MONTHS[dt.month - 1]
    slug = f"{t}-temperature-in-hong-kong-on-{month_name}-{dt.day}-{dt.year}"
    return f"https://polymarket.com/event/{slug}"


def fetch_live_polymarket_prices(date_iso: str, days_ahead: int = 10) -> dict:
    """
    Fetch live Polymarket prices for a given date.

    Returns:
        {
          "high": {bucket_label: price, ...},
          "low":  {bucket_label: price, ...},
          "raw_high": {...},
          "raw_low": {...},
          "slug_high": str | None,
          "slug_low": str | None,
          "events_found": int,
        }
    """
    from polymarket_scraper import PolymarketScraper

    scraper = PolymarketScraper()
    events = scraper.find_events_by_slug(days_ahead=days_ahead)

    result = {
        "high": {},
        "low": {},
        "raw_high": {},
        "raw_low": {},
        "slug_high": None,
        "slug_low": None,
        "events_found": len(events),
    }

    for event in events:
        if event.get("_date") != date_iso:
            continue
        snap = scraper.get_market_snapshot(event)
        prices = snap.get("prices") or {}
        etype = (snap.get("type") or event.get("_type") or "").lower()
        slug = snap.get("slug") or event.get("_slug")

        if etype == "highest":
            result["raw_high"] = prices
            result["high"] = map_live_prices_to_defaults(prices, DEFAULT_HIGH_TEMP_BUCKETS)
            result["slug_high"] = slug
        elif etype == "lowest":
            result["raw_low"] = prices
            result["low"] = map_live_prices_to_defaults(prices, DEFAULT_LOW_TEMP_BUCKETS)
            result["slug_low"] = slug

    return result


def models_exist():
    return (
        (MODELS_DIR / "temp_predictor_max_temp.joblib").exists()
        and (MODELS_DIR / "temp_predictor_min_temp.joblib").exists()
    )


def pricing_status(edge) -> str:
    """UNDERPRICED / OVERPRICED / FAIR VALUE from model vs market edge."""
    try:
        e = float(edge)
    except (TypeError, ValueError):
        return "⚪ FAIR VALUE"
    if abs(e) < 0.01:
        return "⚪ FAIR VALUE"
    if e > 0:
        return "🟢 UNDERPRICED"
    return "🔴 OVERPRICED"


def bet_risk(market_price, model_prob, edge, kelly_fraction=0.0, forecast_temp=None, bucket_label=None) -> str:
    """
    Risk label for a bucket bet.

    Heuristics:
    - HIGH: very cheap lottery-like price, thin edge, or far from forecast
    - MEDIUM: moderate price / size / distance
    - LOW: reasonable price, solid edge, near forecast
    """
    import re

    try:
        px = float(market_price)
    except (TypeError, ValueError):
        px = 0.5
    try:
        mp = float(model_prob)
    except (TypeError, ValueError):
        mp = 0.0
    try:
        e = float(edge)
    except (TypeError, ValueError):
        e = 0.0
    try:
        kelly = float(kelly_fraction)
    except (TypeError, ValueError):
        kelly = 0.0

    distance = None
    if forecast_temp is not None and bucket_label is not None:
        m = re.search(r"(\d+)", str(bucket_label))
        if m:
            try:
                distance = abs(int(m.group(1)) - float(forecast_temp))
            except (TypeError, ValueError):
                distance = None

    score = 0  # higher = riskier

    if px <= 0.08:
        score += 2
    elif px <= 0.15:
        score += 1

    if abs(e) < 0.03:
        score += 1
    elif e >= 0.10 and px >= 0.15:
        score -= 1

    if mp < 0.10:
        score += 2
    elif mp < 0.20:
        score += 1

    if kelly >= 0.15:
        score += 1

    if distance is not None:
        if distance >= 3:
            score += 2
        elif distance >= 2:
            score += 1

    if score >= 3:
        return "🔴 HIGH"
    if score >= 1:
        return "🟡 MEDIUM"
    return "🟢 LOW"


def generate_synthetic_data():
    """Quick synthetic data for demo purposes"""
    dates = pd.date_range("2021-01-01", "2025-12-31")
    n = len(dates)
    doy = dates.dayofyear
    seasonal = 5.0 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    noise_max = np.random.normal(0, 1.5, n)
    noise_min = np.random.normal(0, 1.2, n)
    return pd.DataFrame({
        "hko_max_temp": 28 + seasonal + noise_max,
        "hko_min_temp": 23 + seasonal * 0.7 + noise_min,
        "hko_mean_temp": 25.5 + seasonal * 0.85 + np.random.normal(0, 1.3, n),
        "hko_rh": np.clip(78 + 8 * np.sin(2 * np.pi * (doy - 120) / 365.25) + np.random.normal(0, 8, n), 30, 100),
        "hko_wind_speed": np.clip(12 + 3 * np.sin(2 * np.pi * (doy - 30) / 365.25) + np.random.normal(0, 3, n), 0, 50),
        "hko_pressure": 1013 - 5 * np.cos(2 * np.pi * doy / 365.25) + np.random.normal(0, 3, n),
        "wu_max_temp_c": 28.5 + seasonal + noise_max + np.random.normal(0.3, 0.6, n),
        "wu_min_temp_c": 23.5 + seasonal * 0.7 + noise_min + np.random.normal(0.2, 0.5, n),
        "wu_avg_temp_c": (28.5 + seasonal + noise_max + 23.5 + seasonal * 0.7 + noise_min) / 2,
    }, index=dates)


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## HK Weather Bet")
    st.caption("Temperature Prediction for Polymarket")
    st.divider()
    page = st.radio("Navigate", [
        "Dashboard",
        "Betting Analysis",
        "Bankroll",
        "Backtest",
        "Historical Data",
        "Model Training",
        "Settings",
    ], index=0)
    st.divider()
    st.caption(f"Resolution source: **{RESOLUTION_STATION}**")
    st.caption(f"[Wunderground VHHH]({RESOLUTION_SOURCE_URL})")


# ====================================================================
# PAGE: Dashboard
# ====================================================================
if page == "Dashboard":
    st.markdown("# Dashboard")

    # Fetch live data
    try:
        forecast = fetch_hko_forecast()
        current = fetch_current_weather()
    except Exception as e:
        st.error(f"Cannot reach HKO API: {e}")
        st.stop()

    # ── Current conditions ──────────────────────────────────────────
    st.markdown("### Current Conditions (HKO)")
    try:
        temp_val = current.get("temperature", {}).get("data", [{}])[0].get("value", "N/A")
        rh_val = current.get("humidity", {}).get("data", [{}])[0].get("value", "N/A")
        icon_code = current.get("icon", [0])[0] if current.get("icon") else 0
        icon_map = {50: "Sunny", 51: "Fine", 52: "Fine", 53: "Fine", 54: "Cloudy",
                    60: "Cloudy", 61: "Overcast", 62: "Light rain", 63: "Rain",
                    64: "Heavy rain", 70: "Fine", 71: "Fine", 72: "Fine", 73: "Fine",
                    74: "Fine", 75: "Cloudy", 76: "Rain", 77: "Rain", 80: "Fine",
                    81: "Cloudy", 82: "Wet", 83: "Squally", 84: "Wet", 85: "Cloudy",
                    90: "Sunny", 91: "Fine", 92: "Fine", 93: "Fine", 94: "Fine"}
        weather_text = icon_map.get(icon_code, "Unknown")
    except Exception:
        temp_val, rh_val, weather_text = "N/A", "N/A", "N/A"

    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature", f"{temp_val}°C")
    c2.metric("Humidity", f"{rh_val}%")
    c3.metric("Conditions", weather_text)

    st.divider()

    # ── 9-Day Forecast with probabilities ───────────────────────────
    st.markdown("### 9-Day Forecast & Bucket Probabilities")

    wf = forecast.get("weatherForecast", [])
    if not wf:
        st.warning("No forecast data available.")
        st.stop()

    # Build forecast table
    rows = []
    for day in wf:
        date_str = day.get("forecastDate", "")
        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
            date_display = dt.strftime("%a %b %d")
        except ValueError:
            dt = None
            date_display = date_str

        max_t = float(day.get("forecastMaxtemp", {}).get("value", 0))
        min_t = float(day.get("forecastMintemp", {}).get("value", 0))
        weather = day.get("forecastWeather", "")[:40]

        # Compute probabilities
        std_h, std_l = 1.0, 0.8
        high_buckets = parse_buckets(DEFAULT_HIGH_TEMP_BUCKETS)
        low_buckets = parse_buckets(DEFAULT_LOW_TEMP_BUCKETS)
        high_probs = compute_bucket_probabilities(max_t, std_h, high_buckets)
        low_probs = compute_bucket_probabilities(min_t, std_l, low_buckets)

        # Find highest probability bucket
        best_high = max(high_probs, key=high_probs.get)
        best_low = max(low_probs, key=low_probs.get)

        rows.append({
            "Date": date_display,
            "High (°C)": max_t,
            "Low (°C)": min_t,
            "Most Likely High": f"{best_high} ({high_probs[best_high]:.0%})",
            "Most Likely Low": f"{best_low} ({low_probs[best_low]:.0%})",
            "Weather": weather,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── NWP Ensemble Forecast ──────────────────────────────────────
    st.divider()
    st.markdown("### NWP Ensemble Forecast (ECMWF + GFS)")
    st.caption("80 ensemble members from Open-Meteo — counts members per bucket for probability")

    try:
        @st.cache_data(ttl=3600)
        def fetch_nwp_ensemble():
            nwp = NWPCollector()
            return nwp.fetch_ensemble_forecast(forecast_days=7)

        ens_data = fetch_nwp_ensemble()
        nwp = NWPCollector()

        nwp_rows = []
        for i in range(min(7, len(ens_data['dates']))):
            date_str = ens_data['dates'][i]
            max_stats = nwp.ensemble_stats(ens_data, i, 'max')
            min_stats = nwp.ensemble_stats(ens_data, i, 'min')
            max_probs = nwp.ensemble_to_bucket_probs(ens_data, i, DEFAULT_HIGH_TEMP_BUCKETS, 'max')
            min_probs = nwp.ensemble_to_bucket_probs(ens_data, i, DEFAULT_LOW_TEMP_BUCKETS, 'min')

            if max_stats and min_stats and max_probs and min_probs:
                best_h = max(max_probs, key=max_probs.get)
                best_l = max(min_probs, key=min_probs.get)
                nwp_rows.append({
                    'Date': date_str,
                    'High Mean': f"{max_stats['mean']:.1f}",
                    'High Std': f"{max_stats['std']:.2f}",
                    'Best High': f"{best_h} ({max_probs[best_h]:.0%})",
                    'Low Mean': f"{min_stats['mean']:.1f}",
                    'Low Std': f"{min_stats['std']:.2f}",
                    'Best Low': f"{best_l} ({min_probs[best_l]:.0%})",
                })

        st.dataframe(pd.DataFrame(nwp_rows), use_container_width=True, hide_index=True)
        st.caption(f"{ens_data.get('n_ecmwf', 50)} ECMWF + {ens_data.get('n_gfs', 30)} GFS = {ens_data.get('n_total', 80)} ensemble members")

        # NWP bucket probability chart for selected day
        if len(ens_data['dates']) > 0:
            sel_nwp = st.selectbox("NWP detailed view", range(len(ens_data['dates'])),
                                   format_func=lambda i: ens_data['dates'][i], key='nwp_sel')
            max_p = nwp.ensemble_to_bucket_probs(ens_data, sel_nwp, DEFAULT_HIGH_TEMP_BUCKETS, 'max')
            min_p = nwp.ensemble_to_bucket_probs(ens_data, sel_nwp, DEFAULT_LOW_TEMP_BUCKETS, 'min')

            if max_p and min_p:
                nc1, nc2 = st.columns(2)
                with nc1:
                    st.markdown(f"**NWP High Temp** — {ens_data['dates'][sel_nwp]}")
                    st.bar_chart(pd.DataFrame({'Probability': list(max_p.values())}, index=list(max_p.keys())))
                with nc2:
                    st.markdown(f"**NWP Low Temp** — {ens_data['dates'][sel_nwp]}")
                    st.bar_chart(pd.DataFrame({'Probability': list(min_p.values())}, index=list(min_p.keys())))

    except Exception as e:
        st.warning(f"NWP ensemble forecast unavailable: {e}")

    # ── Probability charts for selected day ─────────────────────────
    st.divider()
    sel = st.selectbox("Select day for detailed probabilities", range(len(wf)),
                       format_func=lambda i: wf[i].get("forecastDate", ""))
    day = wf[sel]
    max_t = float(day.get("forecastMaxtemp", {}).get("value", 0))
    min_t = float(day.get("forecastMintemp", {}).get("value", 0))

    high_buckets = parse_buckets(DEFAULT_HIGH_TEMP_BUCKETS)
    low_buckets = parse_buckets(DEFAULT_LOW_TEMP_BUCKETS)
    high_probs = compute_bucket_probabilities(max_t, 1.0, high_buckets)
    low_probs = compute_bucket_probabilities(min_t, 0.8, low_buckets)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**High Temperature** — Forecast: {max_t}°C")
        prob_df = pd.DataFrame({
            "Bucket": list(high_probs.keys()),
            "Probability": list(high_probs.values()),
        })
        st.bar_chart(prob_df.set_index("Bucket"))

    with col_b:
        st.markdown(f"**Low Temperature** — Forecast: {min_t}°C")
        prob_df = pd.DataFrame({
            "Bucket": list(low_probs.keys()),
            "Probability": list(low_probs.values()),
        })
        st.bar_chart(prob_df.set_index("Bucket"))


# ====================================================================
# PAGE: Betting Analysis
# ====================================================================
elif page == "Betting Analysis":
    st.markdown("# Polymarket Betting Analysis")
    st.caption("Fetch live Polymarket prices or enter them manually to identify +EV bets")

    # ── Fetch forecast ──────────────────────────────────────────────
    try:
        forecast = fetch_hko_forecast()
    except Exception as e:
        st.error(f"Cannot reach HKO API: {e}")
        st.stop()

    wf = forecast.get("weatherForecast", [])
    if not wf:
        st.warning("No forecast data.")
        st.stop()

    # ── Day selector ────────────────────────────────────────────────
    date_labels = []
    for d in wf:
        try:
            dt = datetime.strptime(d["forecastDate"], "%Y%m%d")
            date_labels.append(dt.strftime("%a %b %d"))
        except (ValueError, KeyError):
            date_labels.append(d.get("forecastDate", "?"))

    sel_idx = st.selectbox("Select date", range(len(wf)), format_func=lambda i: date_labels[i])
    day = wf[sel_idx]
    max_t = float(day["forecastMaxtemp"]["value"])
    min_t = float(day["forecastMintemp"]["value"])
    date_str = day["forecastDate"]
    try:
        date_iso = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        date_iso = date_str

    # ── Confidence slider ───────────────────────────────────────────
    confidence = st.slider("Model confidence", 0.5, 1.0, 0.75, 0.05)

    # Load saved market prices
    saved_prices = load_market_prices()
    high_key = f"{date_iso}_high"
    low_key = f"{date_iso}_low"

    # ── Live Polymarket fetch ────────────────────────────────────────
    st.divider()
    col_fetch, col_info = st.columns([1, 3])
    with col_fetch:
        fetch_clicked = st.button("🔄 Fetch live prices", use_container_width=True)
    with col_info:
        st.caption("Ambil harga YES share langsung dari Polymarket untuk tanggal yang dipilih.")

    if fetch_clicked:
        with st.spinner(f"Fetching Polymarket prices for {date_iso}..."):
            try:
                live = fetch_live_polymarket_prices(date_iso)
                n_high = len(live.get("high") or {})
                n_low = len(live.get("low") or {})
                if n_high == 0 and n_low == 0:
                    st.warning(
                        f"Tidak ada market aktif untuk {date_iso} "
                        f"(events scanned: {live.get('events_found', 0)}). "
                        "Coba tanggal lain atau isi manual."
                    )
                else:
                    if n_high:
                        saved_prices[high_key] = {
                            **saved_prices.get(high_key, {}),
                            **live["high"],
                        }
                    if n_low:
                        saved_prices[low_key] = {
                            **saved_prices.get(low_key, {}),
                            **live["low"],
                        }
                    save_market_prices(saved_prices)
                    st.success(
                        f"Live prices loaded — High: {n_high} buckets, Low: {n_low} buckets"
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"Gagal fetch Polymarket: {e}")

    # ── High Temp Market ────────────────────────────────────────────
    st.divider()
    st.markdown(f"## High Temperature Market — {date_iso}")
    st.metric("HKO Forecast High", f"{max_t}°C")

    high_buckets = parse_buckets(DEFAULT_HIGH_TEMP_BUCKETS)
    high_probs = compute_bucket_probabilities(max_t, 1.0, high_buckets)

    saved_high = saved_prices.get(high_key, {})

    st.markdown("**Market Prices (YES share price, 0.00 - 1.00):**")
    hcols = st.columns(len(DEFAULT_HIGH_TEMP_BUCKETS))
    high_market_prices = {}
    for i, (label, _, _) in enumerate(DEFAULT_HIGH_TEMP_BUCKETS):
        default_val = float(saved_high.get(label, 0.14))
        default_val = min(0.99, max(0.01, default_val))
        high_market_prices[label] = hcols[i].number_input(
            label, min_value=0.01, max_value=0.99, value=default_val, step=0.01, key=f"hp_{sel_idx}_{i}"
        )

    # Analyze
    high_analysis = analyze_market(
        date=date_iso, market_type="high",
        predicted_mean=max_t, predicted_std=1.0,
        market_prices=high_market_prices,
        confidence=confidence,
    )

    # Results table
    rows = []
    for bet in high_analysis.bets:
        rows.append({
            "Bucket": bet.bucket_label,
            "Model Prob": f"{bet.model_prob:.1%}",
            "Market Price": f"${bet.market_price:.2f}",
            "Edge": f"{bet.edge:+.1%}",
            "EV": f"{bet.ev:+.1%}",
            "Kelly %": f"{bet.kelly_fraction:.1%}",
            "Status": pricing_status(bet.edge),
            "Risk": bet_risk(
                bet.market_price, bet.model_prob, bet.edge,
                bet.kelly_fraction, forecast_temp=max_t,
                bucket_label=bet.bucket_label,
            ),
            "Action": bet.recommendation,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Visualization
    chart_data = pd.DataFrame({
        "Bucket": list(high_analysis.bucket_probs.keys()),
        "Model Probability": list(high_analysis.bucket_probs.values()),
        "Market Price": [high_market_prices.get(k, 0) for k in high_analysis.bucket_probs.keys()],
    }).set_index("Bucket")
    st.bar_chart(chart_data)

    if high_analysis.best_bet:
        st.success(
            f"**BEST BET: {high_analysis.best_bet.bucket_label}** — "
            f"Edge {high_analysis.best_bet.edge:+.1%} | "
            f"EV {high_analysis.best_bet.ev:+.1%} | "
            f"Kelly {high_analysis.best_bet.kelly_fraction:.1%}"
        )
    else:
        st.info("No +EV bets found. Market appears well-priced.")

    # Save prices
    saved_prices[high_key] = high_market_prices

    # ── Low Temp Market ─────────────────────────────────────────────
    st.divider()
    st.markdown(f"## Low Temperature Market — {date_iso}")
    st.metric("HKO Forecast Low", f"{min_t}°C")

    low_buckets = parse_buckets(DEFAULT_LOW_TEMP_BUCKETS)
    low_probs = compute_bucket_probabilities(min_t, 0.8, low_buckets)

    saved_low = saved_prices.get(low_key, {})

    st.markdown("**Market Prices (YES share price, 0.00 - 1.00):**")
    lcols = st.columns(len(DEFAULT_LOW_TEMP_BUCKETS))
    low_market_prices = {}
    for i, (label, _, _) in enumerate(DEFAULT_LOW_TEMP_BUCKETS):
        default_val = float(saved_low.get(label, 0.14))
        default_val = min(0.99, max(0.01, default_val))
        low_market_prices[label] = lcols[i].number_input(
            label, min_value=0.01, max_value=0.99, value=default_val, step=0.01, key=f"lp_{sel_idx}_{i}"
        )

    low_analysis = analyze_market(
        date=date_iso, market_type="low",
        predicted_mean=min_t, predicted_std=0.8,
        market_prices=low_market_prices,
        bucket_defs=DEFAULT_LOW_TEMP_BUCKETS,
        confidence=confidence,
    )

    rows = []
    for bet in low_analysis.bets:
        rows.append({
            "Bucket": bet.bucket_label,
            "Model Prob": f"{bet.model_prob:.1%}",
            "Market Price": f"${bet.market_price:.2f}",
            "Edge": f"{bet.edge:+.1%}",
            "EV": f"{bet.ev:+.1%}",
            "Kelly %": f"{bet.kelly_fraction:.1%}",
            "Status": pricing_status(bet.edge),
            "Risk": bet_risk(
                bet.market_price, bet.model_prob, bet.edge,
                bet.kelly_fraction, forecast_temp=min_t,
                bucket_label=bet.bucket_label,
            ),
            "Action": bet.recommendation,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    chart_data = pd.DataFrame({
        "Bucket": list(low_analysis.bucket_probs.keys()),
        "Model Probability": list(low_analysis.bucket_probs.values()),
        "Market Price": [low_market_prices.get(k, 0) for k in low_analysis.bucket_probs.keys()],
    }).set_index("Bucket")
    st.bar_chart(chart_data)

    if low_analysis.best_bet:
        st.success(
            f"**BEST BET: {low_analysis.best_bet.bucket_label}** — "
            f"Edge {low_analysis.best_bet.edge:+.1%} | "
            f"EV {low_analysis.best_bet.ev:+.1%} | "
            f"Kelly {low_analysis.best_bet.kelly_fraction:.1%}"
        )
    else:
        st.info("No +EV bets found. Market appears well-priced.")

    saved_prices[low_key] = low_market_prices

    # Save all prices
    save_market_prices(saved_prices)

    # ── Summary ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Quick Summary")

    sc1, sc2 = st.columns(2)

    high_event_url = build_polymarket_event_url(date_iso, "highest")
    low_event_url = build_polymarket_event_url(date_iso, "lowest")

    with sc1:
        if high_analysis.best_bet:
            bb = high_analysis.best_bet
            status = pricing_status(bb.edge)
            st.metric(
                "High Temp Best Bet",
                bb.bucket_label,
                delta=f"{bb.edge:+.1%} edge",
            )
            st.markdown(
                f"**Status:** {status}  \n"
                f"**Action:** BUY YES `{bb.bucket_label}`  \n"
                f"Model {bb.model_prob:.0%} · Market {bb.market_price:.0%} · "
                f"Kelly {bb.kelly_fraction:.1%}  \n"
                f"🔗 [Open High Temp event]({high_event_url})"
            )
        else:
            st.metric("High Temp", "No +EV bets")
            st.markdown(f"🔗 [Open High Temp event]({high_event_url})")

    with sc2:
        if low_analysis.best_bet:
            bb = low_analysis.best_bet
            status = pricing_status(bb.edge)
            st.metric(
                "Low Temp Best Bet",
                bb.bucket_label,
                delta=f"{bb.edge:+.1%} edge",
            )
            st.markdown(
                f"**Status:** {status}  \n"
                f"**Action:** BUY YES `{bb.bucket_label}`  \n"
                f"Model {bb.model_prob:.0%} · Market {bb.market_price:.0%} · "
                f"Kelly {bb.kelly_fraction:.1%}  \n"
                f"🔗 [Open Low Temp event]({low_event_url})"
            )
        else:
            st.metric("Low Temp", "No +EV bets")
            st.markdown(f"🔗 [Open Low Temp event]({low_event_url})")


# ====================================================================
# PAGE: Bankroll Management
# ====================================================================
elif page == "Bankroll":
    st.markdown("# Bankroll Management")
    st.caption("Track P&L, model performance, and bias corrections")

    from model_tracker import get_performance_stats, PERFORMANCE_LOG
    from bias_corrector import get_bias_report, BIAS_LOG

    # ── Performance Stats ──────────────────────────────────────────
    st.markdown("## Model Performance")

    period = st.selectbox("Period", [7, 14, 30, 90], format_func=lambda x: f"Last {x} days", index=2)
    stats = get_performance_stats(days=period)

    if stats:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Brier Score", f"{stats['brier_score']:.3f}" if stats['brier_score'] else "N/A",
                  help="Lower is better. 0=perfect, 0.25=no skill")
        c2.metric("Win Rate", f"{stats['win_rate']:.1%}" if stats['win_rate'] else "N/A")
        c3.metric("ROI", f"{stats['roi']:+.1%}" if stats['roi'] is not None else "N/A")
        c4.metric("Total Bets", stats['total_bets'])
        c5.metric("Resolved Days", stats['resolved_days'])
    else:
        st.info("No performance data yet. Predictions will be tracked after the first Telegram alert run.")

    st.divider()

    # ── Prediction Log ─────────────────────────────────────────────
    st.markdown("## Prediction History")

    if PERFORMANCE_LOG.exists():
        import json as _json
        entries = []
        for line in PERFORMANCE_LOG.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                entries.append(_json.loads(line))

        if entries:
            # Build table
            log_rows = []
            for e in entries[-30:]:  # Last 30 entries
                bets = e.get("recommended_bets", {})
                main_bucket = bets.get("main", {}).get("bucket", "-") if bets.get("main") else "-"
                main_prob = f"{bets.get('main', {}).get('model', 0):.0%}" if bets.get("main") else "-"
                main_edge = f"{bets.get('main', {}).get('edge', 0):+.1%}" if bets.get("main") else "-"
                lottery_bucket = bets.get("lottery", {}).get("bucket", "-") if bets.get("lottery") else "-"

                # Model leader: stored field, else from bucket_probs
                model_leader = e.get("model_leader")
                if not model_leader:
                    probs = e.get("bucket_probs") or {}
                    if probs:
                        model_leader = max(probs, key=probs.get)
                model_leader = model_leader or "-"

                # Market leader: stored field, else from market_prices map
                market_leader = e.get("market_leader")
                market_leader_price = e.get("market_leader_price")
                if not market_leader:
                    mkt = e.get("market_prices") or {}
                    if mkt:
                        market_leader = max(mkt, key=mkt.get)
                        market_leader_price = mkt.get(market_leader)
                if market_leader and market_leader_price is not None:
                    try:
                        market_leader_disp = f"{market_leader} ({float(market_leader_price):.0%})"
                    except (TypeError, ValueError):
                        market_leader_disp = str(market_leader)
                else:
                    market_leader_disp = market_leader or "-"

                status = "Resolved" if e.get("resolved") else "Pending"
                actual = f"{e.get('actual_temp', '?')}\u00b0C" if e.get("actual_temp") is not None else "-"

                log_rows.append({
                    "Date": e["target_date"],
                    "Type": e["temp_type"].upper(),
                    "Model Leader": model_leader,
                    "Market Leader": market_leader_disp,
                    "Main Bet": main_bucket,
                    "Model Prob": main_prob,
                    "Edge": main_edge,
                    "Lottery": lottery_bucket,
                    "Actual": actual,
                    "Status": status,
                })

            st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Market Leader = bucket with highest Polymarket YES price at log time. "
                "Old rows may show '-' until the next telegram_alert /predict run."
            )
        else:
            st.info("No predictions logged yet.")
    else:
        st.info("No prediction log file found. Run telegram_alert.py to start tracking.")

    st.divider()

    # ── Bias Correction ────────────────────────────────────────────
    st.markdown("## Bias Correction (Multi-Day Learning)")
    st.caption("Tracks if model consistently over/under-predicts and applies corrections")

    bias_report = get_bias_report()

    bc1, bc2 = st.columns(2)
    for col, temp_type in [(bc1, "high"), (bc2, "low")]:
        with col:
            b = bias_report[temp_type]
            label = "HIGH Temperature" if temp_type == "high" else "LOW Temperature"
            st.markdown(f"### {label}")

            if b["n_samples"] == 0:
                st.info("No bias data yet. Will start learning after first resolved event.")
            else:
                direction_emoji = "\u2b06\ufe0f" if b["mean_error"] > 0 else "\u2b07\ufe0f" if b["mean_error"] < 0 else "\u2705"
                c_b1, c_b2, c_b3 = st.columns(3)
                c_b1.metric("Direction", f"{direction_emoji} {b['direction']}")
                c_b2.metric("Mean Error", f"{b['mean_error']:+.2f}\u00b0C")
                c_b3.metric("Correction", f"{b['correction']:+.2f}\u00b0C")

                st.caption(f"Samples: {b['n_samples']} | EWMA: {b['ewma_error']:+.2f}\u00b0C | Confidence: {b['confidence']}")

                # Visual indicator
                if abs(b["correction"]) > 1.0:
                    st.warning("Large bias detected — model may need recalibration")
                elif abs(b["correction"]) > 0.5:
                    st.info("Moderate bias — correction being applied")
                else:
                    st.success("Model well-calibrated")

    st.divider()

    # ── Kelly Sizing Reference ─────────────────────────────────────
    st.markdown("## Kelly Criterion Reference")
    st.caption("Optimal bet sizing based on edge and probability")

    bankroll = st.number_input("Current bankroll ($)", value=100.0, step=10.0, min_value=1.0)

    kelly_rows = []
    for prob in [0.20, 0.30, 0.40, 0.50, 0.60]:
        for market_price in [0.10, 0.15, 0.20, 0.25, 0.30]:
            odds = 1.0 / market_price
            b = odds - 1
            p = prob
            q = 1 - p
            kelly_pct = (b * p - q) / b
            if kelly_pct > 0:
                stake = bankroll * kelly_pct * 0.25  # Quarter-Kelly
                stake = min(stake, bankroll * 0.10)  # Cap at 10%
                kelly_rows.append({
                    "Model Prob": f"{prob:.0%}",
                    "Market Price": f"${market_price:.2f}",
                    "Edge": f"{prob - market_price:+.1%}",
                    "Kelly %": f"{kelly_pct:.1%}",
                    "Stake": f"${stake:.2f}",
                })

    if kelly_rows:
        st.dataframe(pd.DataFrame(kelly_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No +EV scenarios at current bankroll level.")

    st.divider()

    # ── Market Efficiency Analysis ─────────────────────────────────
    st.markdown("## Market Efficiency Analysis")
    st.caption("Analyze how Polymarket prices compare to historical frequencies")

    from market_analyzer import analyze_bucket_accuracy, fetch_hka_data, detect_market_patterns

    if st.button("Run Market Analysis (fetches live data)", type="primary", key="btn_market_analysis"):
        with st.spinner("Analyzing market efficiency..."):
            try:
                data = fetch_hka_data()
                recent_cutoff = data.index.max() - pd.DateOffset(years=5)
                recent = data[data.index >= recent_cutoff]

                # Summer months (market season)
                summer = data[data.index.month.isin([5, 6, 7, 8, 9, 10])]
                summer_recent = summer[summer.index >= recent_cutoff]

                me1, me2 = st.columns(2)

                for col, bucket_defs, temp_col, label in [
                    (me1, DEFAULT_HIGH_TEMP_BUCKETS, "max_temp", "HIGH"),
                    (me2, DEFAULT_LOW_TEMP_BUCKETS, "min_temp", "LOW"),
                ]:
                    with col:
                        st.markdown(f"### {label} Temperature")

                        # Historical frequency (summer)
                        summer_data = recent[recent.index.month.isin([5, 6, 7, 8, 9, 10])]
                        freq = analyze_bucket_accuracy(bucket_defs, summer_data, temp_col)

                        # Show bucket frequencies
                        freq_rows = []
                        for blabel, stats in sorted(freq.items(), key=lambda x: x[1]["frequency"], reverse=True):
                            freq_rows.append({
                                "Bucket": blabel,
                                "Frequency": f"{stats['frequency']:.1%}",
                                "Count": stats["count"],
                            })
                        st.dataframe(pd.DataFrame(freq_rows), use_container_width=True, hide_index=True)

                        # Stats
                        temps = summer_data[temp_col]
                        st.caption(f"Summer mean: {temps.mean():.1f}\u00b0C \u00b1 {temps.std():.1f}\u00b0C (n={len(summer_data)})")

                # Pattern detection
                st.divider()
                st.markdown("### Detected Patterns")
                patterns = detect_market_patterns()
                for s in patterns.get("summary", []):
                    st.info(s)

            except Exception as e:
                st.error(f"Market analysis failed: {e}")


# ====================================================================
# PAGE: Historical Data
# ====================================================================
elif page == "Historical Data":
    st.markdown("# Historical Temperature Data")
    st.caption(
        "Primary: **HKA / VHHH Airport** (Polymarket resolution source) · "
        "Compare: HKO Headquarters"
    )

    def _temp_to_bucket_label(temp, bucket_defs):
        for label, lo, hi in bucket_defs:
            if lo <= -900:
                if temp < hi:
                    return label
            elif hi >= 900:
                if temp >= lo:
                    return label
            else:
                if lo <= temp < hi:
                    return label
        return bucket_defs[-1][0]

    def _fetch_station_temps_fallback(collector, station: str) -> pd.DataFrame:
        """Fallback if local data_collector is older (no fetch_station_temps)."""
        import io
        base = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
        urls = {
            "max": f"{base}?dataType=CLMMAXT&rformat=csv&station={station}",
            "min": f"{base}?dataType=CLMMINT&rformat=csv&station={station}",
        }

        def parse_one(url):
            resp = collector.session.get(url, timeout=60)
            resp.raise_for_status()
            text = resp.content.decode("utf-8-sig")
            lines = text.strip().split("\n")
            header_idx = 0
            for i, line in enumerate(lines):
                if "Year" in line or "year" in line:
                    header_idx = i
                    break
            clean = []
            for line in lines[header_idx:]:
                s = line.strip().strip('"')
                if not s or s.startswith("***") or s.startswith("#") or s.startswith("C "):
                    continue
                clean.append(line)
            df = pd.read_csv(io.StringIO("\n".join(clean)))
            cols = df.columns.tolist()
            yc = next(c for c in cols if "year" in c.lower())
            mc = next(c for c in cols if "month" in c.lower())
            dc = next(c for c in cols if "day" in c.lower())
            vc = next(
                c for c in cols
                if "value" in c.lower() or (
                    c not in (yc, mc, dc) and "year" not in c.lower()
                )
            )
            out = pd.DataFrame({
                "date": pd.to_datetime(
                    df[[yc, mc, dc]].rename(columns={yc: "Year", mc: "Month", dc: "Day"}),
                    errors="coerce",
                ),
                "value": pd.to_numeric(df[vc], errors="coerce"),
            }).dropna()
            return out.set_index("date")["value"]

        mx = parse_one(urls["max"])
        mn = parse_one(urls["min"])
        return pd.DataFrame({"max_temp": mx, "min_temp": mn}).dropna()

    try:
        hko = HKODataCollector()

        def _load_station(station: str) -> pd.DataFrame:
            if hasattr(hko, "fetch_station_temps"):
                return hko.fetch_station_temps(station)
            return _fetch_station_temps_fallback(hko, station)

        with st.spinner("Fetching HKA (Airport / Polymarket resolution)..."):
            hka = _load_station("HKA")
        with st.spinner("Fetching HKO Headquarters (comparison)..."):
            try:
                hq = _load_station("HKO")
            except Exception:
                hq = None

        st.success(
            f"HKA loaded: {len(hka)} days "
            f"({hka.index.min().date()} → {hka.index.max().date()})"
            + (f" · HKO HQ: {len(hq)} days" if hq is not None else "")
        )

        n_years = st.slider("Years to show", 3, 40, 10, key="hist_years")
        cutoff = hka.index.max() - pd.DateOffset(years=n_years)
        hka_f = hka[hka.index >= cutoff].copy()
        hq_f = hq[hq.index >= cutoff].copy() if hq is not None else None

        # ── Metrics ───────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("HKA Mean Max", f"{hka_f['max_temp'].mean():.1f}°C")
        m2.metric("HKA Mean Min", f"{hka_f['min_temp'].mean():.1f}°C")
        if hq_f is not None and len(hq_f):
            aligned = hka_f[["max_temp"]].join(
                hq_f[["max_temp"]].rename(columns={"max_temp": "hq_max"}),
                how="inner",
            )
            if len(aligned):
                offset = (aligned["hq_max"] - aligned["max_temp"]).mean()
                m3.metric("HQ − HKA (Max)", f"{offset:+.2f}°C")
            else:
                m3.metric("HQ − HKA (Max)", "n/a")
        else:
            m3.metric("HQ − HKA (Max)", "n/a")
        m4.metric("Days", f"{len(hka_f):,}")

        # ── Chart: HKA primary, HQ secondary ──────────────────────
        st.markdown("### Daily temperatures (resolution source)")
        chart = pd.DataFrame({
            "HKA Max (resolution)": hka_f["max_temp"],
            "HKA Min (resolution)": hka_f["min_temp"],
        })
        if hq_f is not None and len(hq_f):
            chart["HKO HQ Max"] = hq_f["max_temp"]
            chart["HKO HQ Min"] = hq_f["min_temp"]
        st.line_chart(chart)

        # ── Monthly averages ──────────────────────────────────────
        st.markdown("### Monthly average — HKA Max")
        tmp = hka_f.copy()
        tmp["month"] = tmp.index.month
        monthly_avg = tmp.groupby("month")["max_temp"].mean()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_chart = pd.DataFrame({
            "Month": [month_names[i - 1] for i in monthly_avg.index],
            "HKA Max (°C)": monthly_avg.values,
        }).set_index("Month")
        st.bar_chart(monthly_chart)

        # ── Bucket frequency (links to Polymarket structure) ──────
        st.markdown("### Bucket frequency — HKA daily max (Polymarket-style)")
        st.caption("How often each high-temp bucket would have won, based on HKA actuals.")

        season = st.selectbox(
            "Season filter",
            ["All months", "Summer (May–Oct)", "Winter (Nov–Apr)"],
            key="hist_season",
        )
        freq_df = hka_f.copy()
        if season == "Summer (May–Oct)":
            freq_df = freq_df[freq_df.index.month.isin([5, 6, 7, 8, 9, 10])]
        elif season == "Winter (Nov–Apr)":
            freq_df = freq_df[freq_df.index.month.isin([11, 12, 1, 2, 3, 4])]

        if len(freq_df):
            labels = [
                _temp_to_bucket_label(t, DEFAULT_HIGH_TEMP_BUCKETS)
                for t in freq_df["max_temp"]
            ]
            counts = pd.Series(labels).value_counts()
            # Preserve bucket order from config
            order = [b[0] for b in DEFAULT_HIGH_TEMP_BUCKETS]
            counts = counts.reindex(order).fillna(0).astype(int)
            pct = (counts / counts.sum() * 100).round(1)

            freq_table = pd.DataFrame({
                "Bucket": counts.index,
                "Days": counts.values,
                "Frequency %": pct.values,
            })
            st.dataframe(freq_table, use_container_width=True, hide_index=True)
            st.bar_chart(freq_table.set_index("Bucket")["Frequency %"])
        else:
            st.info("No days in selected filter.")

        with st.expander("Raw HKA sample"):
            st.dataframe(hka_f.tail(30), use_container_width=True)

    except Exception as e:
        st.error(f"Could not fetch historical data: {e}")
        st.info(
            "Check network access to data.weather.gov.hk. "
            "Synthetic demo is disabled on this page — fix the source instead."
        )


# ====================================================================
# PAGE: Model Training
# ====================================================================
elif page == "Model Training":
    st.markdown("# Model Training")

    st.markdown("### Model Status")
    if models_exist():
        st.success("Trained models found in `models/` directory")
    else:
        st.warning("No trained models found. Train with synthetic data below or run `python main.py run` in terminal.")

    st.divider()

    # ── Train with synthetic data ───────────────────────────────────
    st.markdown("### Quick Train (Synthetic Data)")
    st.caption("Train models on synthetic HK weather data for demo/testing purposes.")

    if st.button("Train Models (takes ~30s)", type="primary"):
        from features import build_feature_matrix
        from model import HKWeatherEnsemble

        progress = st.progress(0, text="Generating synthetic data...")

        raw_df = generate_synthetic_data()
        progress.progress(20, text="Building features...")

        features = build_feature_matrix(raw_df)
        progress.progress(40, text="Training high temperature model...")

        ensemble = HKWeatherEnsemble()
        max_metrics = ensemble.max_predictor.fit(features, "wu_max_temp_c")
        progress.progress(70, text="Training low temperature model...")

        min_metrics = ensemble.min_predictor.fit(features, "wu_min_temp_c")
        progress.progress(90, text="Saving models...")

        ensemble.save()
        progress.progress(100, text="Done!")

        st.success("Models trained and saved!")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**High Temperature Model**")
            st.metric("MAE", f"{max_metrics['mae']:.2f}°C")
            st.metric("RMSE", f"{max_metrics['rmse']:.2f}°C")
            st.metric("R²", f"{max_metrics['r2']:.4f}")
        with c2:
            st.markdown("**Low Temperature Model**")
            st.metric("MAE", f"{min_metrics['mae']:.2f}°C")
            st.metric("RMSE", f"{min_metrics['rmse']:.2f}°C")
            st.metric("R²", f"{min_metrics['r2']:.4f}")

    # ── Predict with trained model ─────────────────────────────────
    st.divider()
    st.markdown("### Predict with Trained Model")

    if models_exist():
        if st.button("Run Prediction"):
            from features import build_feature_matrix
            from model import HKWeatherEnsemble

            with st.spinner("Loading model and generating features..."):
                raw_df = generate_synthetic_data()
                features = build_feature_matrix(raw_df)

                ensemble = HKWeatherEnsemble()
                ensemble.load()

                recent = features.tail(7)
                preds = ensemble.predict(recent)

            st.markdown("**Last 7 Days Predictions:**")
            pred_df = pd.DataFrame({
                "Date": recent.index.strftime("%Y-%m-%d"),
                "Predicted High (°C)": preds["max_temp"]["mean"].round(1).values,
                "High Std": preds["max_temp"]["std"].round(2).values,
                "Predicted Low (°C)": preds["min_temp"]["mean"].round(1).values,
                "Low Std": preds["min_temp"]["std"].round(2).values,
            })
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
    else:
        st.info("Train models first to enable predictions.")

    # ── Auto-Retrain Pipeline ─────────────────────────────────────
    st.divider()
    st.markdown("### Auto-Retrain Pipeline (v5 Bucket Classifier)")
    st.caption("Retrain the direct bucket classifier with latest HKA data. Deploys only if improved.")

    from auto_retrain import get_model_status, get_retrain_history, run_full_retrain

    # Show current model status
    model_status = get_model_status()
    ms1, ms2 = st.columns(2)
    for col, tt in [(ms1, "high"), (ms2, "low")]:
        with col:
            s = model_status[tt]
            if s["exists"]:
                st.success(f"**{tt.upper()}** model trained {s['trained_at'][:10]}")
                st.metric("Validation Score", f"{s['val_score']:.1%}" if s['val_score'] else "N/A")
                st.caption(f"Training data: {s['train_days']} days")
            else:
                st.warning(f"**{tt.upper()}** no retrained model yet")

    retrain_years = st.slider("Training window (years)", 3, 10, 5, key="retrain_years")

    if st.button("Run Retraining (fetches live data)", type="primary", key="btn_retrain"):
        with st.spinner("Retraining models with latest data..."):
            try:
                results = run_full_retrain(train_years=retrain_years, deploy_if_better=True)
                st.success("Retraining complete!")

                rc1, rc2 = st.columns(2)
                for col, tt in [(rc1, "high"), (rc2, "low")]:
                    with col:
                        r = results[tt]
                        old = f"{r['old_val_score']:.1%}" if r['old_val_score'] else "N/A"
                        delta = f"{r['val_score'] - (r['old_val_score'] or 0):+.1%}"
                        status = "DEPLOYED" if r["deployed"] else "KEPT OLD"
                        st.metric(f"{tt.upper()} Validation", f"{r['val_score']:.1%}", delta=delta)
                        st.caption(f"Old: {old} | {status}")
                st.rerun()
            except Exception as e:
                st.error(f"Retrain failed: {e}")

    # Show retrain history
    history = get_retrain_history()
    if history:
        st.markdown("**Retrain History:**")
        hist_rows = []
        for h in history[-10:]:
            hist_rows.append({
                "Date": h["timestamp"][:10],
                "Type": h["temp_type"].upper(),
                "Val Score": f"{h['val_score']:.1%}",
                "Old Score": f"{h['old_val_score']:.1%}" if h.get("old_val_score") else "N/A",
                "Deployed": "Yes" if h["deployed"] else "No",
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)


# ====================================================================
# PAGE: Settings
# ====================================================================
elif page == "Settings":
    st.markdown("# Settings")

    st.markdown("### Polymarket Bucket Definitions")
    st.caption("These define the temperature ranges for each market bucket.")

    st.markdown("**High Temperature Buckets:**")
    high_df = pd.DataFrame([
        {"Label": l, "Lower (°C)": lo, "Upper (°C)": hi}
        for l, lo, hi in DEFAULT_HIGH_TEMP_BUCKETS
    ])
    st.dataframe(high_df, use_container_width=True, hide_index=True)

    st.markdown("**Low Temperature Buckets:**")
    low_df = pd.DataFrame([
        {"Label": l, "Lower (°C)": lo, "Upper (°C)": hi}
        for l, lo, hi in DEFAULT_LOW_TEMP_BUCKETS
    ])
    st.dataframe(low_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Data Sources")
    st.markdown("""
    | Source | Description |
    |--------|-------------|
    | **HKO Open Data API** | 9-day forecast, current weather, historical data |
    | **DATA.GOV.HK** | Historical CSVs (1884-present) |
    | **Weather Underground VHHH** | Polymarket resolution source |
    """)

    st.divider()
    st.markdown("### How It Works")
    st.markdown("""
    1. **Data Collection** — Fetches HKO historical + forecast data and Weather Underground VHHH readings
    2. **Feature Engineering** — 250+ features: temporal cycles, rolling stats, lag features, same-day historicals, anomaly z-scores
    3. **ML Prediction** — LightGBM + XGBoost ensemble with quantile regression (p10/p25/p50/p75/p90) for uncertainty estimation
    4. **Bucket Probability** — Converts predicted distribution into Polymarket bucket probabilities
    5. **EV Analysis** — Compares model probabilities vs market prices, identifies +EV bets using Kelly criterion

    **Key edge:** HKO's official forecast is highly accurate but Polymarket prices can lag behind forecast updates.
    """)

    st.divider()
    st.markdown("### Saved Market Prices")
    prices = load_market_prices()
    if prices:
        st.json(prices)
        if st.button("Clear saved prices"):
            save_market_prices({})
            st.rerun()
    else:
        st.info("No saved market prices yet.")


# ====================================================================
# PAGE: Backtest
# ====================================================================
elif page == "Backtest":
    st.markdown("# Backtest Results")
    st.caption("v5: Direct Bucket Classifier + Empirical Blend (HKA Airport data, 5-year backtest)")

    # Load v5 results (primary)
    bt_v5_path = PROCESSED_DATA_DIR / "backtest_v5_results.csv"
    has_v5 = bt_v5_path.exists()

    if not has_v5:
        st.warning("No v5 backtest results found. Run `python backtest_v5.py --years 5` first.")
        if st.button("Run Backtest Now (takes ~2 min)", type="primary"):
            with st.spinner("Running v5 backtest..."):
                import subprocess
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).parent / "backtest_v5.py"), "--years", "5"],
                    capture_output=True, text=True, cwd=str(Path(__file__).parent)
                )
            if result.returncode == 0:
                st.success("Backtest complete!")
                st.rerun()
            else:
                st.error(f"Backtest failed: {result.stderr[-500:]}")
        st.stop()

    bt = pd.read_csv(bt_v5_path)
    bt['date'] = pd.to_datetime(bt['date'])

    # ── Version comparison banner ──────────────────────────────────
    st.markdown("## Model Version Comparison")
    st.caption("5-year backtest on HKA Airport data (matches Polymarket VHHH resolution)")

    v3_high_path = PROCESSED_DATA_DIR / 'backtest_v3_high.csv'
    v4_high_path = PROCESSED_DATA_DIR / 'backtest_v4_high.csv'

    comp_data = []
    for months, label in [([6,7,8], "Summer"), (list(range(1,13)), "Full Year")]:
        sub = bt[bt['month'].isin(months)]
        if len(sub) == 0:
            continue
        v5_h = sub['max_correct'].mean()
        v5_l = sub['min_correct'].mean()
        v5_h_t2 = sub['max_top2'].mean()
        v5_l_t2 = sub['min_top2'].mean()

        v3_h, v3_l, v4_h, v4_l = 0, 0, 0, 0
        if v3_high_path.exists():
            v3h = pd.read_csv(v3_high_path)
            v3h_sub = v3h[v3h['month'].isin(months)]
            v3_h = v3h_sub['correct'].mean() if len(v3h_sub) > 0 else 0
            v3l = pd.read_csv(PROCESSED_DATA_DIR / 'backtest_v3_low.csv')
            v3l_sub = v3l[v3l['month'].isin(months)]
            v3_l = v3l_sub['correct'].mean() if len(v3l_sub) > 0 else 0
        if v4_high_path.exists():
            v4h = pd.read_csv(v4_high_path)
            v4h_sub = v4h[v4h['month'].isin(months)]
            v4_h = v4h_sub['correct'].mean() if len(v4h_sub) > 0 else 0
            v4l = pd.read_csv(PROCESSED_DATA_DIR / 'backtest_v4_low.csv')
            v4l_sub = v4l[v4l['month'].isin(months)]
            v4_l = v4l_sub['correct'].mean() if len(v4l_sub) > 0 else 0

        comp_data.append({
            "Season": label,
            "v3 (HKO HQ)": f"{v3_h:.1%}" if v3_h > 0 else "N/A",
            "v4 (HKA+Bias)": f"{v4_h:.1%}" if v4_h > 0 else "N/A",
            "v5 (Classifier)": f"{v5_h:.1%}",
            "v5 Top-2": f"{v5_h_t2:.1%}",
            "v5 Top-3": f"{sub['max_top3'].mean():.1%}",
            "Delta": f"{v5_h - max(v3_h, v4_h):+.1%}",
        })

    st.markdown("**High Temperature Win Rate:**")
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    comp_low = []
    for months, label in [([6,7,8], "Summer"), (list(range(1,13)), "Full Year")]:
        sub = bt[bt['month'].isin(months)]
        if len(sub) == 0:
            continue
        v5_l = sub['min_correct'].mean()
        v5_l_t2 = sub['min_top2'].mean()

        v3_l, v4_l = 0, 0
        if v3_high_path.exists():
            v3l = pd.read_csv(PROCESSED_DATA_DIR / 'backtest_v3_low.csv')
            v3l_sub = v3l[v3l['month'].isin(months)]
            v3_l = v3l_sub['correct'].mean() if len(v3l_sub) > 0 else 0
        if v4_high_path.exists():
            v4l = pd.read_csv(PROCESSED_DATA_DIR / 'backtest_v4_low.csv')
            v4l_sub = v4l[v4l['month'].isin(months)]
            v4_l = v4l_sub['correct'].mean() if len(v4l_sub) > 0 else 0

        comp_low.append({
            "Season": label,
            "v3 (HKO HQ)": f"{v3_l:.1%}" if v3_l > 0 else "N/A",
            "v4 (HKA+Bias)": f"{v4_l:.1%}" if v4_l > 0 else "N/A",
            "v5 (Classifier)": f"{v5_l:.1%}",
            "v5 Top-2": f"{v5_l_t2:.1%}",
            "v5 Top-3": f"{sub['min_top3'].mean():.1%}",
            "Delta": f"{v5_l - max(v3_l, v4_l):+.1%}",
        })

    st.markdown("**Low Temperature Win Rate:**")
    st.dataframe(pd.DataFrame(comp_low), use_container_width=True, hide_index=True)

    st.divider()

    # ── Season selector ────────────────────────────────────────────
    season_options = {
        "Summer (Jun-Aug)": [6, 7, 8],
        "Polymarket (May-Oct)": [5, 6, 7, 8, 9, 10],
        "Full Year": list(range(1, 13)),
    }
    sel_season = st.selectbox("Season", list(season_options.keys()), index=0)
    months = season_options[sel_season]
    sub = bt[bt['month'].isin(months)]

    if len(sub) == 0:
        st.warning("No data for selected season.")
        st.stop()

    n = len(sub)

    # ── Summary metrics ────────────────────────────────────────────
    st.markdown("## v5 Performance Summary")

    for label, temp_col in [("HIGH TEMPERATURE", "max"), ("LOW TEMPERATURE", "min")]:
        wr = sub[f'{temp_col}_correct'].mean()
        t2 = sub[f'{temp_col}_top2'].mean()
        t3 = sub[f'{temp_col}_top3'].mean()
        clf_t1 = sub[f'clf_{temp_col}_correct'].mean()
        clf_t2 = sub[f'clf_{temp_col}_top2'].mean()

        st.markdown(f"### {label}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("v5 Win Rate", f"{wr:.1%}", delta=f"{wr/(1/7):.2f}x vs random")
        c2.metric("Top-2 Accuracy", f"{t2:.1%}")
        c3.metric("Top-3 Accuracy", f"{t3:.1%}")
        c4.metric("Classifier Only", f"{clf_t1:.1%}")
        c5.metric("Days Tested", f"{n}")

        st.divider()

    # ── Bucket breakdown ────────────────────────────────────────────
    st.markdown("## Bucket Analysis")

    tab_h, tab_l = st.tabs(["High Temperature", "Low Temperature"])

    for tab, temp_col, bucket_defs, tlabel in [
        (tab_h, "max", DEFAULT_HIGH_TEMP_BUCKETS, "High"),
        (tab_l, "min", DEFAULT_LOW_TEMP_BUCKETS, "Low"),
    ]:
        with tab:
            summer = bt[bt['month'].isin([6, 7, 8])]
            if len(summer) == 0:
                continue

            rows = []
            for blabel, lo, hi in bucket_defs:
                actual_n = (summer[f'{temp_col}_actual_bucket'] == blabel).sum()
                freq = actual_n / len(summer) if len(summer) > 0 else 0
                model_picks = (summer[f'{temp_col}_best'] == blabel).sum()
                model_correct = summer[(summer[f'{temp_col}_best'] == blabel) & (summer[f'{temp_col}_correct'])].shape[0]
                model_wr = model_correct / model_picks if model_picks > 0 else 0
                edge = model_wr - freq
                market_price = max(freq, 0.02)
                roi = model_wr * (1/market_price - 1) - (1 - model_wr) if model_picks > 0 else 0

                rows.append({
                    "Bucket": blabel,
                    "Actual Count": actual_n,
                    "Hist. Freq": f"{freq:.1%}",
                    "Model Picks": model_picks,
                    "Model WR": f"{model_wr:.0%}",
                    "Edge": f"{edge:+.1%}",
                    "ROI/bet": f"{roi:+.2f}",
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Chart
            chart_rows = []
            for blabel, lo, hi in bucket_defs:
                actual_freq = (summer[f'{temp_col}_actual_bucket'] == blabel).sum() / len(summer)
                model_freq = (summer[f'{temp_col}_best'] == blabel).sum() / len(summer)
                chart_rows.append({"Bucket": blabel, "Actual Frequency": actual_freq, "Model Picks": model_freq})
            chart_df = pd.DataFrame(chart_rows).set_index("Bucket")
            st.bar_chart(chart_df)

    # ── Key insights ────────────────────────────────────────────────
    st.divider()
    st.markdown("## Key Insights for Polymarket Betting")
    st.markdown("""
    **v5 Model Performance:**
    
    | Metric | Summer Value | Interpretation |
    |--------|-------------|----------------|
    | Best bucket win rate | ~36-39% | **2.5x better than random** (14.3%) |
    | Top-2 accuracy | ~57-60% | Excellent for multi-bucket strategies |
    | Top-3 accuracy | ~72-77% | Strong coverage |
    | Classifier only | ~36-40% | Direct bucket prediction works well |
    
    **Where the edge comes from:**
    1. **Direct bucket classification** — GBM predicts bucket directly instead of temperature → bucket
    2. **Class-balanced training** — Fixes underrepresented buckets (30-31°C, 31-32°C)
    3. **Ensemble blend** — 50% classifier + 50% empirical combines strengths
    4. **HKA Airport data** — Matches Polymarket VHHH resolution (not HKO HQ)
    
    **Every bucket has positive edge:**
    - Previously blind spots (30-31°C, 31-32°C) now have +11-13% edge
    - Extreme buckets (<30°C, 35+°C) have +35-40% edge
    - Common buckets (33-34°C, 28-29°C) still have +17% edge
    
    **Recommended strategy:**
    - Multi-bucket approach: bet top-2 buckets when combined probability >50%
    - Focus on extreme buckets where edge is largest (+35% on 35+°C)
    - Use Kelly sizing: small bets (1-5% of bankroll) on high-conviction picks
    """)
