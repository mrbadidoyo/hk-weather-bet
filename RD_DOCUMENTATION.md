# HK Weather Temperature Prediction System
## Research & Development Document

**Project:** Hong Kong Daily High/Low Temperature Prediction for Polymarket Weather Betting
**Date:** August 2026
**Status:** Working prototype with calibrated backtest results
**Language:** Python 3.14, Streamlit

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement & Market Context](#2-problem-statement--market-context)
3. [Data Sources](#3-data-sources)
4. [System Architecture](#4-system-architecture)
5. [Module Reference](#5-module-reference)
6. [Feature Engineering](#6-feature-engineering)
7. [Prediction Models](#7-prediction-models)
8. [Polymarket Strategy](#8-polymarket-strategy)
9. [Backtest Results](#9-backtest-results)
10. [Key Findings & Insights](#10-key-findings--insights)
11. [Known Limitations](#11-known-limitations)
12. [Research Directions for Improvement](#12-research-directions-for-improvement)
13. [Technical Notes & Pitfalls](#13-technical-notes--pitfalls)
14. [How to Run](#14-how-to-run)
15. [File Index](#15-file-index)

---

## 1. Project Overview

This system predicts daily high and low temperatures in Hong Kong and converts predictions into betting probabilities for **Polymarket weather markets**. Polymarket runs multi-outcome bucket markets for Hong Kong temperature, where each bucket represents a temperature range (e.g., "32-33°C"). The market resolves on **Weather Underground VHHH** (Hong Kong International Airport) daily temperature data.

**Core value proposition:** Identify mispriced temperature buckets where our model has an edge over the market, then use Kelly criterion for position sizing.

**Current results (5-year backtest, real HKO data):**
- Summer win rate (best bucket): **23.2%** (high) / **22.6%** (low) vs 14.3% random baseline
- Top-2 accuracy: **40.8%** (high) / **48.8%** (low)
- MAE: **1.56°C** (high) / **1.11°C** (low)
- ROI vs realistic market prices: **+0.21** (high) / **+0.30** (low) per $1 bet

---

## 2. Problem Statement & Market Context

### 2.1 Polymarket Weather Markets

Polymarket operates prediction markets including weather markets. The HK temperature market structure:

- **Market type:** Multi-outcome (7 buckets)
- **Resolution source:** Weather Underground (Wunderground) daily data for VHHH station
- **VHHH URL:** `https://www.wunderground.com/history/daily/hk/hong-kong/VHHH`
- **Bucket format:** Temperature ranges (e.g., `<30°C`, `30-31°C`, `31-32°C`, ..., `35+°C`)
- **Resolution:** Daily at ~midnight HKT, uses Wunderground's reported max/min temperature

### 2.2 Key Challenge

The market is semi-efficient. For summer months in Hong Kong:
- Historical distribution: 32-34°C accounts for ~43% of all summer days
- The market prices these common buckets efficiently
- Edge exists primarily in **tail events** (unusually hot or cold days)

### 2.3 Station Discrepancy

**Critical:** HKO Headquarters (HKO station) ≠ VHHH Airport. There is a systematic temperature offset:
- HKO HQ is in urban Tsim Sha Tsui (urban heat island effect)
- VHHH Airport is at Chek Lap Kok (coastal, windier, slightly cooler)
- Typical offset: HKO max is ~0.5-1.0°C higher than VHHH

The current model uses HKO data for training but should ideally use VHHH data for Polymarket resolution.

---

## 3. Data Sources

### 3.1 HKO Open Data API (Primary)

| Data Type | URL Pattern | Coverage |
|-----------|-------------|----------|
| 9-Day Forecast (JSON) | `data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd` | Rolling 9 days |
| Current Weather (JSON) | `...weather.php?dataType=rhrread` | Real-time |
| Local Forecast (JSON) | `...weather.php?dataType=flw` | Today/tomorrow |
| Historical Max Temp (CSV) | `...opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKO` | **1884-present** |
| Historical Min Temp (CSV) | `...opendata/opendata.php?dataType=CLMMINT&rformat=csv&station=HKO` | **1884-present** |
| Historical Mean Temp (CSV) | `...opendata/opendata.php?dataType=CLMTEMP&rformat=csv&station=HKO` | 1884-present |
| Historical Rainfall (CSV) | `...opendata/opendata.php?dataType=CLMRF&rformat=csv&station=HKO` | 1884-present |
| Historical RH (CSV) | `...opendata/opendata.php?dataType=CLMRH&rformat=csv&station=HKO` | 1884-present |
| Historical Wind Speed (CSV) | `...opendata/opendata.php?dataType=CLMWS&rformat=csv&station=HKO` | 1884-present |
| Historical Pressure (CSV) | `...opendata/opendata.php?dataType=CLMP&rformat=csv&station=HKO` | 1884-present |

**Dataset size:** ~49,490 daily records (1884-2026)

**CSV Format Warning:** HKO CSVs use UTF-8 BOM encoding, bilingual headers (English + Chinese), and footer lines starting with `***` or `#`. Must be cleaned before parsing.

### 3.2 Weather Underground (Resolution Source)

- **URL:** `https://www.wunderground.com/history/daily/hk/hong-kong/VHHH/date/YYYY-M-D`
- **Data:** Daily max/min/avg temperature for VHHH station
- **Used for:** Polymarket market resolution, model validation
- **Access:** Web scraping (BeautifulSoup), rate-limited

### 3.3 Additional Data Sources (Not Yet Integrated)

| Source | URL | Potential Use |
|--------|-----|---------------|
| ECMWF (ERA5) | `cds.climate.copernicus.eu` | Reanalysis, ensemble forecast |
| GFS/NCEP | `nomads.ncep.noaa.gov` | Medium-range forecast models |
| JMA | `jma.go.jp` | Regional typhoon/monsoon info |
| HKO Radar | `weather.gov.hk` | Convective activity detection |
| ENSO Index | `psl.noaa.gov/enso` | Seasonal temperature predictor |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     HK Weather Prediction System                  │
├─────────────┬──────────────┬───────────────┬────────────────────┤
│  Data Layer │ Feature Layer│  Model Layer  │  Strategy Layer    │
├─────────────┼──────────────┼───────────────┼────────────────────┤
│ config.py   │ features.py  │ model.py      │ polymarket_        │
│ data_       │ 250+ features│ LightGBM +    │  strategy.py       │
│ collector.py│ Rolling stats│ XGBoost       │ Bucket probs       │
│             │ Lag features │ Quantile      │ Kelly criterion    │
│             │ Same-day hist│ regression    │ EV calculation     │
│             │ Cyclical enc │               │                    │
├─────────────┼──────────────┼───────────────┼────────────────────┤
│ backtest_v3.py (Calibrated empirical distribution + KDE)         │
├─────────────┴──────────────┴───────────────┴────────────────────┤
│ app.py (Streamlit GUI) │ main.py (CLI)                           │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
HKO API (CSV/JSON) ──→ data_collector.py ──→ features.py ──→ model.py
                                                              │
Wunderground (VHHH) ──→ data_collector.py ────────────────────┘
                                                              │
                                                polymarket_strategy.py
                                                              │
                                                      app.py (GUI)
```

---

## 5. Module Reference

### 5.1 config.py (149 lines)
Central configuration: API endpoints, station codes, bucket definitions, model hyperparameters, file paths.

**Key constants:**
```python
HKO_API_BASE = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
RESOLUTION_STATION = "VHHH"  # Polymarket resolves on this

DEFAULT_HIGH_TEMP_BUCKETS = [
    ("<30", -999, 30), ("30-31", 30, 31), ("31-32", 31, 32),
    ("32-33", 32, 33), ("33-34", 33, 34), ("34-35", 34, 35), ("35+", 35, 999),
]
DEFAULT_LOW_TEMP_BUCKETS = [
    ("<25", -999, 25), ("25-26", 25, 26), ("26-27", 26, 27),
    ("27-28", 27, 28), ("28-29", 28, 29), ("29-30", 29, 30), ("30+", 30, 999),
]
```

### 5.2 data_collector.py (327 lines)
Two collector classes:
- `HKODataCollector` — HKO Open Data API (JSON forecasts, CSV historical)
- `WundergroundCollector` — VHHH scraping for resolution data
- `build_combined_dataset()` — merges both sources
- `collect_all_data()` — orchestrates full pipeline

### 5.3 features.py (272 lines)
Generates 250+ features:
- **Temporal:** day_of_year, month, cyclical sin/cos encoding, season indicators
- **Rolling:** mean/std/min/max/median for windows [3, 7, 14, 30] days + EWM [7, 14, 30]
- **Lag:** lag [1, 2, 3, 7] days, diff(1), diff(7), pct_change(1)
- **Same-day historical:** mean/std/min/max/median per day-of-year across all years, anomaly z-scores
- **Interaction:** temp_range (max-min), heat_index_approx

### 5.4 model.py (346 lines)
- `TemperaturePredictor` — LightGBM + XGBoost ensemble
  - 5 quantile models per algorithm: p10, p25, p50, p75, p90
  - Ensemble weight: 60% LightGBM + 40% XGBoost
  - `predict_distribution()` → mean, std, p10-p90 quantiles
- `HKWeatherEnsemble` — wraps max + min predictors
  - Blends ML prediction with HKO official forecast (60% ML + 40% HKO)

### 5.5 polymarket_strategy.py (393 lines)
- `Bucket` dataclass — temperature range with `contains()` method
- `MarketBet` dataclass — bucket label, model prob, market price, edge, EV, Kelly fraction
- `compute_bucket_probabilities()` — two methods:
  - **Normal:** Gaussian CDF from predicted mean/std
  - **Empirical:** Piecewise linear CDF from quantile regression
- `evaluate_bets()` — compares model probs vs market prices, computes Kelly criterion
- `analyze_market()` — full analysis pipeline

### 5.6 backtest_v3.py (495 lines) — **PRIMARY BACKTEST MODULE**
- `EmpiricalDistribution` — KDE-based probability estimation from same-day historical data
- `CalibratedPredictor` — blends empirical distribution + recent rolling average
  - 40% historical same-day + 45% 7-day rolling + 15% 3-day rolling
  - Distribution shift based on recent anomaly from historical mean
  - Volatility adjustment: 70% historical std + 30% recent std
- Bucket probability: 70% empirical KDE + 30% Gaussian (smoothing)
- Realistic ROI calculation using historical bucket frequencies as market prices

### 5.7 app.py (838 lines)
Streamlit GUI with 6 pages:
1. **Dashboard** — Live HKO data, 9-day forecast, probability bars
2. **Betting Analysis** — Market price input, EV calculation, recommendations
3. **Backtest** — Seasonal results, bucket breakdown, calibration check
4. **Historical Data** — HKO CSV viewer with year slider
5. **Model Training** — One-click training with metrics display
6. **Settings** — Bucket config, data source docs, saved prices

### 5.8 main.py (430 lines)
CLI interface with commands:
- `python main.py collect` — fetch all data
- `python main.py train` — train models
- `python main.py predict --date YYYY-MM-DD` — predict for date
- `python main.py run` — full pipeline
- `python main.py quick` — HKO forecast only (no training needed)

---

## 6. Feature Engineering

### 6.1 Feature Categories (250+ features)

| Category | Count | Description |
|----------|-------|-------------|
| Temporal | ~10 | DOY, month, sin/cos cycles, season dummies |
| Rolling statistics | ~60 | Mean/std/min/max/median × windows [3,7,14,30] |
| EWM features | ~12 | Exponential weighted mean for spans [7,14,30] |
| Lag features | ~12 | Lag [1,2,3,7], diff(1), diff(7), pct_change |
| Same-day historical | ~20 | Per-DOY stats across all years + z-scores |
| Interaction | ~4 | Range, heat index approximation |

### 6.2 Key Feature Insights

**Most predictive features (from LightGBM feature importance):**
1. `same_day_hist_mean` — same-day historical mean is the strongest signal
2. `roll7_mean` — 7-day rolling average captures weather regime
3. `doy_sin` / `doy_cos` — cyclical seasonality
4. `roll3_mean` — short-term momentum
5. `same_day_hist_std` — day-of-year variability (higher std = more uncertain)

**Feature importance pattern:** For summer months, recent rolling features become more important than historical features because summer temperature is more influenced by short-term weather systems (typhoons, monsoon surges).

---

## 7. Prediction Models

### 7.1 ML Ensemble (model.py)

```
LightGBM (60% weight)     XGBoost (40% weight)
  ├── Quantile p10          ├── Quantile p10
  ├── Quantile p25          ├── Quantile p25
  ├── Quantile p50          ├── Quantile p50
  ├── Quantile p75          ├── Quantile p75
  └── Quantile p90          └── Quantile p90

         ↓ Ensemble blend → predict_distribution()
         ↓ Returns: mean, std, p10, p25, p50, p75, p90
```

**Hyperparameters:**
```python
LightGBM: n_estimators=500, lr=0.05, max_depth=8, num_leaves=63, subsample=0.8
XGBoost:  n_estimators=500, lr=0.05, max_depth=8, subsample=0.8
```

### 7.2 Calibrated Empirical Model (backtest_v3.py) — **PRODUCTION MODEL**

This is the model that produced the backtest results. Key innovation: **uses empirical same-day distribution, not Gaussian assumption.**

```python
class EmpiricalDistribution:
    """KDE-based distribution from same-day historical values"""
    # Uses scipy.stats.gaussian_kde with Silverman bandwidth
    # Integrates KDE over each bucket range for probability

class CalibratedPredictor:
    """Blended prediction:
       mean = 0.40 * hist_mean + 0.45 * roll7_mean + 0.15 * roll3_mean
       std = 0.70 * hist_std + 0.30 * recent_14d_std
       
       bucket_probs = 0.70 * empirical_kde_probs + 0.30 * gaussian_probs
    """
```

**Why empirical > Gaussian for HK summer:**
- HK summer temperatures are right-skewed (long tail toward 35°C+)
- Gaussian underestimates probability of extreme heat
- Same-day empirical distribution captures the actual shape, including multimodality
- Typhoon days create bimodal distributions (hot before, cool during)

### 7.3 HKO Forecast Blend (model.py)

```python
# Final prediction blends ML with HKO official forecast:
final = 0.60 * ml_prediction + 0.40 * hko_forecast
```

HKO's 1-day forecast is extremely accurate (MAE ~0.5°C). The blend adds significant edge when HKO forecast is available (1-2 days ahead).

---

## 8. Polymarket Strategy

### 8.1 Bucket Probability Calculation

Two methods for converting temperature distribution to bucket probabilities:

**Method 1 — Normal Distribution (simple):**
```python
P(bucket) = Φ((upper - mean) / std) - Φ((lower - mean) / std)
```

**Method 2 — Empirical CDF (production):**
```python
# Shift empirical distribution by recent anomaly
shifted_buckets = [(label, lo - shift, hi - shift) for label, lo, hi in buckets]
probs = base_dist.bucket_probabilities(shifted_buckets)  # KDE integration

# Blend with Gaussian for smoothing
final = 0.70 * empirical + 0.30 * gaussian
```

### 8.2 Expected Value & Kelly Criterion

```python
edge = model_probability - market_price
EV = (model_prob * (1/market_price - 1)) - (1 - model_prob)
kelly_fraction = edge / (1/market_price - 1)  # for multi-outcome
kelly_bet = kelly_fraction * confidence_multiplier * bankroll
```

### 8.3 Betting Decision Matrix

| Condition | Action |
|-----------|--------|
| Edge > 5% AND Kelly > 1% | BUY (positive EV) |
| model_prob < market_price - 10% | SELL (overpriced) |
| Edge < 5% or Kelly < 1% | SKIP (no edge) |

---

## 9. Backtest Results

### 9.1 Backtest v3 — Final Results (5-year, real HKO data)

**Test period:** July 2021 - June 2026 (1,827 days)
**Training data:** 1884-2021 (47,663 days)

#### Seasonal Win Rates (Best Bucket Prediction)

| Season | High Temp | Low Temp | Random |
|--------|-----------|----------|--------|
| **Summer (JJA)** | **23.2%** | **22.6%** | 14.3% |
| Polymarket (May-Oct) | 33.9% | 32.0% | 14.3% |
| Autumn (SON) | 56.7% | 55.2% | 14.3% |
| Winter (DJF) | 100.0% | 100.0% | 14.3% |
| Spring (MAM) | 80.4% | 77.6% | 14.3% |
| Full Year | 64.9% | 63.7% | 14.3% |

#### Multi-Bucket Accuracy (Summer)

| Metric | High Temp | Low Temp |
|--------|-----------|----------|
| Best bucket | 23.2% | 22.6% |
| Top-2 | 40.8% | 48.8% |
| Top-3 | 57.5% | 68.5% |
| MAE | 1.56°C | 1.11°C |

#### Summer Bucket Edge Analysis (High Temperature)

| Bucket | Actual Freq | Model WR | Edge | ROI/bet |
|--------|-------------|----------|------|---------|
| `<30°C` | 16.5% | 25% | **+8.5%** | +0.52 |
| `30-31°C` | 13.9% | 0% | -13.9% | 0.00 |
| `31-32°C` | 11.3% | 3% | -7.9% | -0.70 |
| `32-33°C` | 21.3% | 22% | +0.8% | +0.04 |
| `33-34°C` | 20.2% | 27% | **+6.8%** | +0.34 |
| `34-35°C` | 12.1% | 0% | -12.1% | 0.00 |
| `35+°C` | 4.8% | 50% | **+45.2%** | +9.48 |

#### Summer Bucket Edge Analysis (Low Temperature)

| Bucket | Actual Freq | Model WR | Edge | ROI/bet |
|--------|-------------|----------|------|---------|
| `<25°C` | 4.3% | 22% | **+17.2%** | +3.97 |
| `25-26°C` | 13.2% | 40% | **+26.8%** | +2.02 |
| `26-27°C` | 18.9% | 22% | +3.1% | +0.17 |
| `27-28°C` | 21.9% | 22% | -0.1% | 0.00 |
| `28-29°C` | 33.0% | 29% | -4.4% | -0.13 |
| `29-30°C` | 8.7% | 0% | -8.7% | 0.00 |

### 9.2 Calibration Assessment

| Predicted Prob Range | Count | Actual Win Rate | Calibrated? |
|---------------------|-------|-----------------|-------------|
| 10-20% (High) | 15 | 20.0% | Yes |
| 20-30% (High) | 326 | 19.6% | Yes |
| 30-50% (High) | 89 | 28.1% | Overconfident |
| >50% (High) | 31 | 48.4% | Overconfident |

**Interpretation:** Model is well-calibrated in the 10-30% range (most common for bucket prediction). Overconfident above 30% — needs calibration adjustment for high-confidence picks.

### 9.3 Realistic ROI Summary

| Metric | High Temp | Low Temp |
|--------|-----------|----------|
| ROI vs uniform prices (1/7) | +3.54 | +3.46 |
| ROI vs realistic prices (historical freq) | **+0.21** | **+0.30** |

**Key insight:** Edge is real but small. Profitable only with consistent execution and proper bankroll management.

---

## 10. Key Findings & Insights

### 10.1 What Works

1. **Empirical same-day distribution** is significantly better than Gaussian for HK summer temperatures. The actual distribution is right-skewed and sometimes bimodal.

2. **Recent trend blending** (40% hist + 45% roll7 + 15% roll3) captures short-term weather regime changes that pure climatology misses.

3. **Tail events** are where the edge is concentrated. The model correctly identifies unusually cold days (`<30°C` high in summer) and unusually hot days (`35+°C`).

4. **HKO's official forecast** is extremely accurate for 1-day ahead (MAE ~0.5°C). Blending with HKO forecast significantly improves predictions.

5. **Class imbalance is the #1 problem.** Winter/autumn days where `<30°C` is trivially correct inflate overall accuracy to 64%, masking poor summer performance (23%).

### 10.2 What Doesn't Work

1. **Common bucket prediction.** The model struggles with the most common summer buckets (32-34°C high, 27-29°C low). These are also the hardest to profit from because markets price them efficiently.

2. **Gaussian assumption.** A normal distribution underestimates extreme temperatures in HK summer. The empirical distribution fixes this but needs more data for rare buckets.

3. **HKO vs VHHH offset.** Training on HKO HQ data but resolving on VHHH Airport introduces systematic bias (~0.5-1°C offset in max temp).

4. **No NWP integration.** The model doesn't use numerical weather prediction (GFS, ECMWF) outputs, which are the gold standard for temperature forecasting.

### 10.3 Seasonal Patterns

- **Winter:** Trivial prediction (<30°C high, <25°C low always correct). Not useful for Polymarket.
- **Spring:** Good accuracy but market may not be active.
- **Summer (Jun-Aug):** Most relevant for Polymarket. Win rate 23%, edge in tails.
- **Autumn:** Moderate accuracy, market may be active through October.

---

## 11. Known Limitations

### 11.1 Critical Limitations

| # | Limitation | Impact | Fix Difficulty |
|---|-----------|--------|---------------|
| 1 | **HKO ≠ VHHH station offset** | Systematic ~0.5-1°C bias in max temp | Medium — need VHHH historical data |
| 2 | **No historical Polymarket prices** | Cannot compute true ROI | Hard — need to scrape/archive market data |
| 3 | **No NWP forecast integration** | Missing the best signal source | Medium — need GFS/ECMWF API access |
| 4 | **Overconfident calibration** | High-confidence picks underperform | Easy — isotonic regression calibration |
| 5 | **Common bucket blind spot** | 0% accuracy on 30-31°C, 34-35°C | Hard — needs better model architecture |

### 11.2 Data Limitations

- Wunderground VHHH data requires web scraping (fragile, rate-limited)
- HKO CSV format changes occasionally (BOM encoding, bilingual headers)
- No access to real-time Polymarket order book / prices
- Typhoon events are underrepresented in training data

### 11.3 Model Limitations

- LightGBM/XGBoost ensemble trained on synthetic data, not fully retrained on real data
- No ensemble weather forecast (NWP) integration
- Quantile regression doesn't account for day-of-year varying skewness
- Kelly criterion assumes known true probability (which we estimate imperfectly)

---

## 12. Research Directions for Improvement

### 12.1 High Priority (Expected Large Impact)

#### A. VHHH Station Data Integration
- **Goal:** Train/validate on VHHH data (Polymarket resolution source), not HKO HQ
- **Approach:** Scrape Wunderground VHHH daily data for full history, or find HKO's Airport station CSV
- **Expected impact:** Eliminate ~0.5-1°C systematic bias → improve all bucket predictions

#### B. NWP Forecast Integration (GFS/ECMWF)
- **Goal:** Use numerical weather prediction model outputs as features
- **Data sources:**
  - GFS: `https://nomads.ncep.noaa.gov/` (free, 0.25° resolution, 4x daily)
  - ECMWF: `https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels` (free historical)
  - HKO WRF: `https://www.hko.gov.hk/en/wxinfo/model/` (HK-specific, high-res)
- **Expected impact:** NWP forecasts have MAE ~1°C for 1-day ahead, significantly better than climatology alone

#### C. Historical Polymarket Price Scraping
- **Goal:** Build dataset of actual market prices for true backtesting
- **Approach:** Polymarket API or Gamma Markets API, archive daily
- **Expected impact:** Enable true PnL backtesting, identify which market conditions offer edge

#### D. Calibration Fix
- **Goal:** Apply isotonic regression or Platt scaling to fix overconfidence
- **Approach:** Use backtest predictions as training data for calibration layer
- **Expected impact:** Better Kelly sizing, improved ROI on high-confidence bets

### 12.2 Medium Priority

#### E. Multi-Bucket Betting Strategy
- **Current:** Bet on best single bucket
- **Proposed:** Bet on top-2 or top-3 buckets when combined probability >50%
- **Rationale:** Top-3 accuracy is 58-69% in summer — much better than single bucket
- **Implementation:** Optimize Kelly sizing across multiple correlated outcomes

#### F. Typhoon Detection
- **Goal:** Identify days likely affected by typhoons (cooler, wetter)
- **Data:** HKO typhoon warnings, JTWC best track data
- **Impact:** Typhoon days are extreme outliers in temperature — detecting them early gives huge edge on `<30°C` bucket

#### G. Time-of-Day Analysis
- **Current:** Uses daily max/min only
- **Proposed:** Analyze hourly temperature patterns
- **Rationale:** Morning temperature trajectory predicts afternoon max better than daily features alone
- **Data:** HKO hourly API (`dataType=rhrread` gives current readings)

#### H. Ensemble Weather Model Post-Processing
- **Goal:** Statistical post-processing (MOS — Model Output Statistics) of GFS/ECMWF ensemble forecasts
- **Approach:** Train regression on ensemble spread/mean → observed temperature
- **Impact:** Gold standard approach used by professional weather forecasters

### 12.3 Experimental / Long-Term

#### I. Transformer / LSTM for Time Series
- Replace LightGBM with sequence model (Temporal Fusion Transformer, LSTM)
- Capture multi-day weather pattern evolution
- Requires more data and compute

#### J. Reinforcement Learning for Betting
- Agent learns optimal betting strategy from historical Polymarket data
- State: model probabilities, market prices, bankroll, days remaining
- Action: bet amounts per bucket
- Reward: cumulative PnL

#### K. Alternative Markets
- Expand beyond HK temperature to other Polymarket weather markets
- Tokyo, Singapore, NYC temperature markets
- Rainfall markets (HKO has rainfall data)

#### L. Market Making Strategy
- Instead of directional bets, provide liquidity on both sides
- Earn bid-ask spread using model's uncertainty estimate
- Requires API access for automated trading

---

## 13. Technical Notes & Pitfalls

### 13.1 HKO CSV Parsing

HKO historical CSVs have several quirks:
```python
# 1. UTF-8 BOM encoding
text = resp.content.decode('utf-8-sig')

# 2. Header line detection (search for 'Year' column)
for i, line in enumerate(lines):
    if 'Year' in line:
        header_idx = i
        break

# 3. Footer lines starting with *** or #
clean_lines = [l for l in lines if not l.strip().startswith('***') 
               and not l.strip().startswith('#')]
```

### 13.2 Pandas Duplicate Index Handling

When using `.loc[date]` on a DatetimeIndex that may have duplicates:
```python
# WRONG: raises ValueError if date appears multiple times
value = df.loc[date, column]

# CORRECT: handle DataFrame return
row = df.loc[date]
if isinstance(row, pd.DataFrame):
    row = row.iloc[-1]
```

### 13.3 NumPy Version Compatibility

```python
# NumPy 2.0+ removed np.trapz
# Use np.trapezoid instead:
integral = np.trapezoid(y, x)  # not np.trapz(y, x)
```

### 13.4 Pandas 3.0+ Deprecations

```python
# WRONG (deprecated in pandas 3.0+):
df.fillna(method='ffill')

# CORRECT:
df.ffill().bfill()
```

### 13.5 LightGBM eval_set Deprecation

```python
# WRONG (deprecated):
model.fit(X, y, eval_set=[(X_val, y_val)])

# CORRECT:
model.fit(X, y, eval_X=X_val, eval_y=y_val)
```

### 13.6 Windows PowerShell Issues

- Use `python -m streamlit run app.py` (not `streamlit run app.py`)
- Use semicolons `;` instead of `&&` for command chaining
- Avoid Unicode characters (emoji, box-drawing) in console output

### 13.7 Streamlit Deprecation Warnings

```python
# Deprecated:
st.dataframe(df, use_container_width=True)

# Correct (Streamlit 2025+):
st.dataframe(df, width='stretch')
```

---

## 14. How to Run

### 14.1 Prerequisites

```bash
cd hk_weather_bet
pip install -r requirements.txt
# Also install: pip install streamlit
```

**Python version:** 3.14+ tested (should work on 3.10+)

### 14.2 Quick Start (No Training Required)

```bash
# Use HKO's official forecast directly
python main.py quick
```

This fetches HKO's 9-day forecast and shows temperature probability distributions. No model training needed.

### 14.3 Full Pipeline

```bash
# 1. Collect data
python main.py collect

# 2. Train models (takes ~1 min)
python main.py train

# 3. Predict for a specific date
python main.py predict --date 2026-08-06

# 4. Or run full pipeline
python main.py run
```

### 14.4 Backtest

```bash
# Run calibrated backtest (5 years, ~2 min)
python backtest_v3.py --years 5

# Results saved to data/processed/backtest_v3_high.csv and backtest_v3_low.csv
```

### 14.5 Streamlit GUI

```bash
python -m streamlit run app.py --server.port 8501
# Open http://localhost:8501
```

Pages:
- **Dashboard** — Live HKO data + 9-day forecast
- **Betting Analysis** — Input market prices, get EV analysis
- **Backtest** — View backtest results with interactive charts
- **Historical Data** — Browse HKO historical records
- **Model Training** — Train/retrain models
- **Settings** — Configure buckets, data sources

### 14.6 Test with Synthetic Data

```bash
python test_pipeline.py
# Runs full pipeline with generated data (no API calls needed)
```

---

## 15. File Index

```
hk_weather_bet/
├── config.py                  # Central configuration (149 lines)
├── data_collector.py          # HKO + Wunderground data fetching (327 lines)
├── features.py                # 250+ feature engineering (272 lines)
├── model.py                   # LightGBM + XGBoost ensemble (346 lines)
├── polymarket_strategy.py     # Bucket probs, EV, Kelly (393 lines)
├── backtest.py                # v1 backtest (deprecated) (420 lines)
├── backtest_v2.py             # v2 seasonal backtest (416 lines)
├── backtest_v3.py             # v3 calibrated empirical model (495 lines) ★ USE THIS
├── main.py                    # CLI runner (430 lines)
├── app.py                     # Streamlit GUI (838 lines)
├── test_pipeline.py           # Synthetic data test (144 lines)
├── _check_api.py              # API diagnostic script
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── raw/
│   │   ├── hko_max_temp_sample.csv    # Sample HKO data
│   │   └── market_prices.json         # Saved market prices
│   └── processed/
│       ├── backtest_v3_high.csv       # ★ Final high temp backtest results
│       ├── backtest_v3_low.csv        # ★ Final low temp backtest results
│       ├── backtest_v2_high.csv       # v2 results (for comparison)
│       ├── backtest_v2_low.csv
│       └── backtest_results.csv       # v1 results (deprecated)
│
└── models/
    ├── temp_predictor_max_temp.joblib  # Trained max temp model
    └── temp_predictor_min_temp.joblib  # Trained min temp model
```

---

## Appendix A: Polymarket HK Temperature Market Structure

Based on research (July 2026), Polymarket HK weather markets typically have:

**High Temperature Buckets (summer example):**
```
<30°C  |  30-31°C  |  31-32°C  |  32-33°C  |  33-34°C  |  34-35°C  |  35+°C
```

**Low Temperature Buckets (summer example):**
```
<25°C  |  25-26°C  |  26-27°C  |  27-28°C  |  28-29°C  |  29-30°C  |  30+°C
```

**Resolution:** Weather Underground daily reported temperature for VHHH. Markets close at midnight HKT, resolve within 24 hours.

**Typical pricing pattern (summer):**
- Common buckets (32-34°C high): 20-25% each
- Uncommon buckets (<30°C, 35+°C): 5-15% each
- Total always sums to ~100%

**Market efficiency observations:**
- Prices closely track HKO forecast when available
- Lag behind typhoon events (temperature drops not priced in fast enough)
- Weekend markets sometimes mispriced due to lower liquidity

---

## Appendix B: HK Climate Summary

### Hong Kong Temperature Climatology (1991-2020 normals)

| Month | Avg Max | Avg Min | Std Max | Std Min |
|-------|---------|---------|---------|---------|
| Jan | 18.6 | 14.5 | 2.3 | 2.5 |
| Feb | 19.6 | 15.4 | 2.5 | 2.6 |
| Mar | 22.1 | 17.8 | 2.3 | 2.3 |
| Apr | 25.8 | 21.3 | 2.2 | 2.1 |
| May | 28.8 | 24.2 | 1.8 | 1.7 |
| Jun | 30.7 | 26.5 | 1.6 | 1.5 |
| Jul | 31.9 | 27.3 | 1.4 | 1.3 |
| Aug | 31.6 | 27.1 | 1.5 | 1.4 |
| Sep | 30.5 | 26.1 | 1.8 | 1.7 |
| Oct | 28.3 | 23.7 | 2.1 | 2.2 |
| Nov | 24.8 | 19.8 | 2.5 | 2.6 |
| Dec | 20.5 | 15.9 | 2.4 | 2.5 |

### Summer Temperature Distribution (JJA, HKO station)

```
High temp histogram:
  <30°C : 16.5%  (rainy/typhoon days)
  30-31°C: 13.9%
  31-32°C: 11.3%
  32-33°C: 21.3%  ← peak
  33-34°C: 20.2%  ← near peak
  34-35°C: 12.1%
  35+°C  :  4.8%  (extreme heat)

Low temp histogram:
  <25°C :  4.3%
  25-26°C: 13.2%
  26-27°C: 18.9%
  27-28°C: 21.9%  ← peak
  28-29°C: 33.0%  ← dominant bucket
  29-30°C:  8.7%
```

**Key observation:** High temp distribution is roughly normal but right-skewed. Low temp distribution is right-skewed with 28-29°C dominating (33%).

---

## Appendix C: Prompt Templates for Further Research

### For Improving the Model

```
I have a Hong Kong temperature prediction system for Polymarket betting.
Current summer win rate: 23% (vs 14.3% random).

The model uses:
1. Empirical same-day historical distribution (KDE) from HKO data (1884-2026)
2. Recent rolling average blend (40% hist + 45% roll7 + 15% roll3)
3. 7 buckets per market, resolved on Weather Underground VHHH

Key problems:
- HKO HQ ≠ VHHH Airport (0.5-1°C systematic offset)
- Model can't predict common summer buckets (32-34°C)
- No NWP forecast integration
- Overconfident calibration above 30%

Please help me [integrate GFS forecast data / build a calibration layer / 
design a multi-bucket Kelly strategy / ...]
```

### For NWP Integration

```
I need to integrate GFS numerical weather prediction forecasts into my
HK temperature model. GFS data is available at:
https://nomads.ncep.noaa.gov/

Requirements:
- Download 0.25° resolution GFS forecasts for HK (22.3°N, 114.2°E)
- Extract T2M (2m temperature) forecast for lead times 1-7 days
- Use as features alongside existing empirical distribution
- Need to handle GFS 6-hourly model runs (00Z, 06Z, 12Z, 18Z)

Please help me [build the GFS data pipeline / create MOS post-processing / ...]
```

### For Backtest Improvement

```
My HK weather backtest currently shows 23% summer win rate (7-bucket market).
The model uses empirical same-day distributions with KDE.

Backtest results by bucket:
- <30°C: edge +8.5%, ROI +0.52 (overpredicts, but when correct, high ROI)
- 33-34°C: edge +6.8%, ROI +0.34 (most common bucket, slight edge)
- 35+°C: edge +45.2%, ROI +9.48 (rare but huge edge when correct)
- 30-31°C, 34-35°C: 0% accuracy (model never picks these)

I need to:
1. Fix the blind spots (buckets model never predicts)
2. Better calibrate high-confidence predictions
3. Test multi-bucket strategies (bet top-2 instead of top-1)

Please help me [redesign the prediction approach / build multi-bucket optimizer / ...]
```

---

*Document generated August 2026. All backtest results based on real HKO data.*
