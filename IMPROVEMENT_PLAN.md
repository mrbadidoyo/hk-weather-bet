# Improvement Plan: HK Temperature Prediction Accuracy
## Data-Driven Analysis & Actionable Steps

**Date:** August 5, 2026
**Current summer win rate:** 23.2% (high) / 22.6% (low) vs 14.3% random

---

## 1. Root Cause Analysis

### 1.1 The Model Has a Systematic Cold Bias

Error analysis on summer (JJA) backtest reveals the **single biggest problem**:

#### High Temperature Errors

| Actual Temp Range | Days | Win Rate | Model Usually Predicts | Diagnosis |
|---|---|---|---|---|
| 28-30°C | 62 | **64.5%** | `<30` (correct) | Works well |
| 30-32°C | 116 | **0.9%** | `<30` (wrong!) | **Blind spot** |
| 32-34°C | 191 | 26.7% | `32-33` or `<30` | Partially OK |
| 34-36°C | 76 | **3.9%** | `32-33` (underpredicts) | Misses heat |
| 36-40°C | 2 | 0% | `33-34` (way off) | Rare events |

**Key finding:** When actual is 30-32°C (116 days!), the model predicts `<30` in 59% of cases. This is a **systematic cold bias** — the model "defaults" to predicting `<30` even when the actual temperature is moderate.

#### Low Temperature Errors

| Actual Temp Range | Days | Win Rate | Model Usually Predicts | Diagnosis |
|---|---|---|---|---|
| 24-26°C | 77 | 18.2% | `26-27` or `<25` | Partially OK |
| 26-27°C | 87 | **50.6%** | `26-27` (correct) | Works well |
| 27-28°C | 101 | **40.6%** | `26-27` or `27-28` | OK |
| **28-29°C** | **152** | **1.3%** | `27-28` (underpredicts) | **BIGGEST blind spot** |
| 29-31°C | 40 | **0%** | `27-28` (way off) | Never predicts |

**Key finding:** 28-29°C is the **most common** summer low temp (33% of days!), but the model achieves only 1.3% accuracy on it. The model predicts `26-27°C` instead.

### 1.2 Why This Happens

1. **Station mismatch (HKO vs VHHH):** HKO HQ is ~0.5-1°C warmer than VHHH Airport. The model trains on HKO data but the distribution was calibrated against HKO's own station, creating an offset when applied to VHHH-equivalent predictions.

2. **Blending weights favor historical mean:** The 40% historical + 45% roll7 + 15% roll3 blend is too anchored to the long-term average. When summer heat waves occur, the 7-day rolling average lags behind.

3. **KDE bandwidth too smooth:** Silverman bandwidth smoothing may over-smooth bimodal distributions (hot vs rainy days in summer).

4. **No external forecast signal:** The model has NO access to NWP forecasts (GFS/ECMWF) which would directly predict temperature anomalies 1-3 days ahead.

### 1.3 Prediction Uncertainty Paradox

| Predicted Std | Days | Win Rate | MAE |
|---|---|---|---|
| 0-1.5 (confident) | 67 | 13.4% | 1.55°C |
| 1.5-2.0 (moderate) | 325 | 24.3% | 1.55°C |
| 2.0-2.5 (uncertain) | 69 | **27.5%** | 1.64°C |

**Counterintuitive:** When the model is MORE confident (lower std), it performs WORSE. This means the model's confidence signal is **inverted** — it's confidently wrong on cold-bias predictions.

### 1.4 Bucket Distance When Wrong

When the model misses, how far off is it?

| Distance | Count | Cumulative |
|---|---|---|
| 1 bucket off | 126 | 36% |
| 2 buckets off | 106 | 66% |
| 3 buckets off | 75 | 87% |
| 4+ buckets off | 47 | 100% |

**Implication:** 36% of errors are off by just 1 bucket. If we could shift the prediction by 1 bucket in the right direction, we'd recover a huge portion of losses. This points to **bias correction** as the highest-impact fix.

---

## 2. Improvement Plan (Ranked by Expected Impact)

### Priority 1: Switch to VHHH/HKA Data (Expected: +5-8% win rate)

**Problem:** Model trains on HKO HQ but market resolves on VHHH Airport.
**Fix:** Use HKO's Airport station data directly.

**Data source (confirmed available):**
```
https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKA
https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMINT&rformat=csv&station=HKA
```

**Impact rationale:**
- HKO HQ is ~0.5-1°C warmer than Airport in summer
- Current model predicts `<30°C` too often because it's calibrated to warmer HKO readings
- Using Airport data directly aligns training with resolution source
- Expected to fix 30-50% of the 30-32°C blind spot (116 days at 0.9% WR)

**Implementation:**
```python
# In config.py, change:
HKO_HISTORICAL_CSVS = {
    "daily_max_temp": "...?dataType=CLMMAXT&rformat=csv&station=HKA",  # was HKO
    "daily_min_temp": "...?dataType=CLMMINT&rformat=csv&station=HKA",  # was HKO
}
```

---

### Priority 2: Add NWP Ensemble Forecast Features (Expected: +8-15% win rate)

**Problem:** Model has no access to weather forecast data — purely statistical.
**Fix:** Add GFS/ECMWF ensemble forecast as input features.

**Data source (confirmed free, no API key):**
```python
# Open-Meteo Ensemble API (50 ECMWF + 30 GFS members)
url = "https://ensemble-api.open-meteo.com/v1/ensemble"
params = {
    "latitude": 22.31, "longitude": 113.93,
    "daily": "temperature_2m_max,temperature_2m_min",
    "models": "ecmwf_ifs025,gfs_seamless",
    "timezone": "Asia/Hong_Kong"
}
# Returns: 50+30 individual member forecasts per day
```

**Impact rationale:**
- ECMWF IFS has MAE ~0.8°C for 1-day ahead HK forecast
- GFS ensemble spread directly estimates forecast uncertainty
- 80 ensemble members give a robust probability distribution
- This is the single biggest signal the model is currently missing

**Implementation plan:**
```python
# New module: nwp_collector.py
class NWPCollector:
    def fetch_ensemble_forecast(self, date, lead_days=7):
        """Fetch GFS+ECMWF ensemble forecasts for HK"""
        # Returns: DataFrame with 80 members × lead_days rows
        # Columns: member, model, date, lead_day, tmax, tmin
    
    def ensemble_to_bucket_probs(self, date, bucket_defs):
        """Convert ensemble members directly to bucket probabilities"""
        # Count how many members fall in each bucket
        # Much more accurate than KDE from historical data
```

**Expected bucket probability improvement:**
- Current KDE approach: ~23% best-bucket accuracy
- Ensemble approach (80 members): potentially 35-40%+ for 1-day ahead
- Ensemble spread = natural uncertainty estimate (no calibration needed)

---

### Priority 3: Bias Correction Layer (Expected: +3-5% win rate)

**Problem:** Model has systematic cold bias, especially in 30-32°C and 28-29°C ranges.
**Fix:** Add a learned bias correction using cross-validated residuals.

**Implementation:**
```python
class BiasCorrector:
    def __init__(self):
        self.bias_by_doy = {}      # bias per day-of-year
        self.bias_by_month = {}     # bias per month
        self.bias_by_pred_bucket = {}  # bias per predicted bucket
    
    def fit(self, predictions, actuals):
        """Learn bias from cross-validated predictions"""
        residuals = actuals - predictions
        
        # Per-month bias
        for month in range(1, 13):
            mask = actuals.index.month == month
            self.bias_by_month[month] = residuals[mask].mean()
        
        # Per predicted-bucket bias (most impactful)
        for bucket_label in all_buckets:
            mask = predicted_buckets == bucket_label
            if mask.sum() > 10:
                self.bias_by_pred_bucket[bucket_label] = residuals[mask].mean()
    
    def correct(self, pred_mean, pred_bucket, month):
        """Apply bias correction"""
        correction = (
            0.5 * self.bias_by_month.get(month, 0) +
            0.5 * self.bias_by_pred_bucket.get(pred_bucket, 0)
        )
        return pred_mean + correction
```

**Why this works:**
- When model predicts `<30°C`, actual is typically ~1.2°C higher → shift prediction up
- When model predicts `32-33°C`, actual is typically ~0.3°C lower → shift prediction down
- This alone could fix ~36% of 1-bucket-off errors

---

### Priority 4: Direct Bucket Classification (Expected: +3-7% win rate)

**Problem:** Current approach predicts temperature THEN converts to buckets. Errors compound.
**Fix:** Train a classifier directly on bucket labels.

**Implementation:**
```python
from sklearn.ensemble import GradientBoostingClassifier

class BucketClassifier:
    def __init__(self, bucket_defs):
        self.buckets = bucket_defs
        self.classifier = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1
        )
    
    def fit(self, X, y_temp):
        """Convert temperatures to bucket labels, then train"""
        y_buckets = [self._temp_to_bucket(t) for t in y_temp]
        self.classifier.fit(X, y_buckets)
        # Apply class_weight or SMOTE for rare buckets
    
    def predict_proba(self, X):
        """Direct bucket probabilities"""
        return self.classifier.predict_proba(X)
```

**Advantages:**
- Directly optimizes the metric we care about (bucket accuracy)
- Can use class_weight='balanced' to fix underrepresented buckets
- No distribution assumption (Gaussian, KDE, etc.)
- Handles asymmetric bucket widths naturally

**Key features to use:**
1. NWP ensemble mean/spread (from Priority 2)
2. Same-day historical distribution stats
3. Recent rolling averages
4. HKO official forecast
5. Month/DOY features

---

### Priority 5: Multi-Bucket Betting Strategy (Expected: +2-4% effective ROI)

**Problem:** Currently bets only the single best bucket. Top-2 accuracy is ~41-49%.
**Fix:** Bet on top-2 buckets when combined probability is high.

**Analysis from backtest data:**
- Top-1 accuracy: 23.2%
- Top-2 accuracy: ~41% (estimated from bucket distance data)
- Top-3 accuracy: ~58%

**Strategy:**
```python
def multi_bucket_strategy(probs, market_prices, bankroll):
    """
    Multi-bucket Kelly strategy.
    Bet on top-2 buckets when:
    1. Combined model probability > 50%
    2. Both buckets have positive EV
    """
    sorted_buckets = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top2 = sorted_buckets[:2]
    combined_prob = top2[0][1] + top2[1][1]
    
    bets = []
    for label, prob in top2:
        price = market_prices[label]
        edge = prob - price
        if edge > 0.03:  # 3% minimum edge
            kelly = edge / (1/price - 1)
            kelly = min(kelly, 0.05)  # cap at 5% bankroll
            bets.append({
                'bucket': label,
                'prob': prob,
                'price': price,
                'edge': edge,
                'bet_size': kelly * bankroll
            })
    
    return bets
```

---

### Priority 6: Typhoon Day Detection (Expected: +2-3% edge on tail events)

**Problem:** Typhoon days have dramatically cooler temperatures (2-5°C drop). Model misses these.
**Fix:** Add typhoon warning signals as features.

**Data sources:**
- HKO typhoon warnings: `https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=wtc` (warning codes)
- JTWC best track: `https://www.metoc.navy.mil/jtwc/jtwc.html`
- Historical typhoon dates can be derived from HKO pressure/wind data

**Impact:**
- ~15-20 typhoon days per year in HK summer
- These days often fall in `<30°C` high temp bucket
- Current model doesn't detect them → misses easy `<30°C` bets
- Typhoon signal = low pressure + high wind → strong predictor

---

### Priority 7: Polymarket Historical Price Scraping (Expected: enables true ROI measurement)

**Problem:** We simulate ROI using historical frequencies, not actual market prices.
**Fix:** Scrape real Polymarket prices for true PnL backtesting.

**API (confirmed free, no key for reads):**
```python
# Step 1: Find HK weather markets
import requests
events = requests.get("https://gamma-api.polymarket.com/events", 
    params={"tag": "weather", "active": "true"}).json()

# Step 2: Get price history
for market in events:
    condition_id = market['condition_id']
    prices = requests.get("https://clob.polymarket.com/prices-history",
        params={"market": condition_id, "interval": "all", "fidelity": "60"}).json()
    # Store for backtesting
```

---

## 3. Implementation Priority & Expected Impact Summary

| # | Improvement | Expected Impact | Difficulty | Data Available? |
|---|---|---|---|---|
| 1 | **Switch to HKA/VHHH data** | +5-8% win rate | Easy (1 file change) | **YES** — HKO API `station=HKA` |
| 2 | **Add NWP ensemble forecasts** | +8-15% win rate | Medium (new module) | **YES** — Open-Meteo free API |
| 3 | **Bias correction layer** | +3-5% win rate | Easy (post-processing) | Can derive from backtest |
| 4 | **Direct bucket classifier** | +3-7% win rate | Medium (new model) | Uses existing features |
| 5 | **Multi-bucket Kelly strategy** | +2-4% ROI | Easy (strategy change) | No new data needed |
| 6 | **Typhoon detection** | +2-3% edge on tails | Medium | **YES** — HKO warnings API |
| 7 | **Polymarket price scraping** | Enables true ROI | Easy | **YES** — Polymarket CLOB API |

**Combined expected improvement:** Summer win rate from **23% → 35-45%** (estimated)

---

## 4. Quick Wins (Can Implement Now)

### 4.1 Switch to Airport Data (5 minutes of work)

```python
# In config.py:
HKO_HISTORICAL_CSVS = {
    "daily_max_temp_hka": "...opendata.php?dataType=CLMMAXT&rformat=csv&station=HKA",
    "daily_min_temp_hka": "...opendata.php?dataType=CLMMINT&rformat=csv&station=HKA",
}
```

### 4.2 Add Open-Meteo Ensemble Forecast (30 minutes of work)

```python
import requests

def fetch_ensemble_forecast(lat=22.31, lon=113.93):
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "models": "ecmwf_ifs025,gfs_seamless",
        "timezone": "Asia/Hong_Kong"
    }
    data = requests.get(url, params=params).json()
    
    # Extract ensemble members
    # data['ecmwf_ifs025']['daily']['temperature_2m_max'] → list of 50 members × 16 days
    # Convert to bucket probabilities by counting members per bucket
    
    return data
```

### 4.3 Bias Correction from Backtest Data (10 minutes of work)

```python
# Run backtest, compute residuals, apply correction
import pandas as pd

bt = pd.read_csv('data/processed/backtest_v3_high.csv')
bt['residual'] = bt['actual'] - bt['pred_mean']

# Per-month bias
bias_by_month = bt.groupby('month')['residual'].mean()
print(bias_by_month)
# Apply: corrected_pred = pred_mean + bias_by_month[month]
```

---

## 5. Architecture for v2 System

```
┌──────────────────────────────────────────────────────────────┐
│                    v2 Architecture                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Data Sources:                                                │
│  ├── HKO HKA (Airport) CSV ─── historical max/min ──────────┐│
│  ├── Open-Meteo Ensemble API ── 80 NWP members ────────────┐││
│  ├── HKO Forecast (fnd/flw) ─── official prediction ──────┐│││
│  ├── Polymarket CLOB API ────── real market prices ───────┐││││
│  └── HKO Warnings API ────────── typhoon signals ─────────┐│││││
│                                                             ││││││
│  Feature Layer:                                              ││││││
│  ├── Same-day empirical distribution (KDE)                   ││││││
│  ├── NWP ensemble mean, spread, percentiles                 ││││││
│  ├── HKO forecast blend                                     ││││││
│  ├── Recent rolling + trend features                        ││││││
│  ├── Typhoon indicator (binary)                             ││││││
│  └── Bias-corrected prediction                              ││││││
│                                                              ││││││
│  Model Layer:                                                 ││││││
│  ├── Option A: Ensemble bucket counting (NWP members)       ││││││
│  ├── Option B: Direct bucket classifier (GBM)               ││││││
│  └── Option C: Blended (NWP probs × 0.5 + empirical × 0.5) ││││││
│                                                              ││││││
│  Strategy Layer:                                              ││││││
│  ├── Multi-bucket Kelly sizing                                ││││││
│  ├── Edge threshold filter (min 3%)                           ││││││
│  ├── Bankroll management                                      ││││││
│  └── Real Polymarket price comparison                         ││││││
│                                                               ││││││
│  Output:                                                      ││││││
│  ├── Dashboard: probabilities + recommendations               ││││││
│  └── Auto-bet API (optional, future)                          ││││││
└──────────────────────────────────────────────────────────────┘││││││
                                                                ││││││
  ──────────────────────────────────────────────────────────────┘│││││
                                                                 │││││
  ───────────────────────────────────────────────────────────────┘││││
                                                                  ││││
  ────────────────────────────────────────────────────────────────┘│││
                                                                   │││
  ─────────────────────────────────────────────────────────────────┘││
                                                                    ││
  ──────────────────────────────────────────────────────────────────┘│
                                                                     │
  ───────────────────────────────────────────────────────────────────┘
```

---

## 6. Validation Plan

After implementing improvements, validate with:

1. **Out-of-sample backtest:** Last 2 years (2024-2026), not used in training
2. **Per-bucket calibration:** Predicted prob should match actual frequency (within 5%)
3. **Brier score comparison:** v1 vs v2 vs NWP-only baseline
4. **Simulated PnL:** $1000 bankroll, Kelly sizing, realistic market prices from Polymarket API
5. **Paper trading:** Run model daily for 30 days, record predictions, compare to actual VHHH

---

## 7. Key Open-Meteo API Endpoints (Confirmed Working)

| API | URL | Use Case |
|---|---|---|
| Historical Archive | `https://archive-api.open-meteo.com/v1/archive?latitude=22.31&longitude=113.93&daily=temperature_2m_max,temperature_2m_min&timezone=Asia/Hong_Kong` | Historical VHHH data (ERA5 reanalysis) |
| Forecast | `https://api.open-meteo.com/v1/forecast?latitude=22.31&longitude=113.93&daily=temperature_2m_max,temperature_2m_min&models=ecmwf_ifs025&timezone=Asia/Hong_Kong` | Deterministic forecast (up to 16 days) |
| Ensemble | `https://ensemble-api.open-meteo.com/v1/ensemble?latitude=22.31&longitude=113.93&daily=temperature_2m_max,temperature_2m_min&models=ecmwf_ifs025,gfs_seamless&timezone=Asia/Hong_Kong` | 80-member ensemble for probability estimation |

All endpoints: **Free, no API key, JSON response.**

---

## 8. Key Polymarket API Endpoints (Confirmed Working)

| API | URL | Use Case |
|---|---|---|
| Gamma (events) | `https://gamma-api.polymarket.com/events?tag=weather&active=true` | Find HK weather markets |
| CLOB (prices) | `https://clob.polymarket.com/prices-history?market={id}&interval=all&fidelity=60` | Historical market prices |
| CLOB (current) | `https://clob.polymarket.com/price?token_id={id}` | Current market price |
| Python client | `pip install py-clob-client` | Official SDK |

All read-only endpoints: **Free, no API key.**

---

*Document generated August 2026. All API endpoints verified working.*
