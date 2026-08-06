# HK Temperature Prediction System

Sistem prediksi suhu Hong Kong untuk betting di Polymarket weather markets.

## Overview

Proyek ini membangun model prediksi untuk market suhu harian HK di Polymarket:
- **Highest Temperature** — 11 bucket (27°C or below → 37°C or higher)
- **Lowest Temperature** — 11 bucket (21°C or below → 31°C or higher)

Resolution source: HKO "Absolute Daily Max/Min" dari [Daily Extract](https://www.weather.gov.hk/en/cis/climat.htm)

## Features

- **Data Collection**: HKA station data dari HKO API + Open-Meteo ensemble NWP
- **Feature Engineering**: Rolling stats, seasonal features, lag features
- **Prediction Models**:
  - v5: Direct bucket classifier (GBM) + empirical distribution blend
  - Ensemble blend: 50% GBM + 50% KDE-based empirical distribution
- **Polymarket Scraper**: Live price scraper via Gamma API
- **Telegram Alert**: Hourly prediction alerts with market price comparison
- **Streamlit GUI**: Interactive dashboard untuk monitoring

### v2 Improvements

- **Dynamic Bucket Ranges**: Bucket ranges extracted live from Polymarket (shift daily)
- **Kelly Criterion**: Optimal bet sizing with quarter-Kelly (0.25 fraction), capped at 10% bankroll
- **Model Performance Tracking**: JSONL log of predictions vs outcomes, Brier score, win rate, ROI
- **Ensemble NWP Integration**: 80 members (50 ECMWF + 30 GFS) blended 60% HKO + 40% NWP
- **Multi-Day Bias Correction**: EWMA-based learning from prediction errors, auto-corrects forecast drift
- **Bankroll Dashboard**: P&L tracking, prediction history, bias visualization, Kelly reference table
- **Auto-Retrain Pipeline**: Weekly GBM classifier retraining with validation comparison, auto-deploy if improved
- **Market Efficiency Analysis**: Historical frequency analysis, pattern detection, mispricing identification

## Project Structure

```
hk_weather_bet/
├── config.py              # Konfigurasi bucket, paths, constants
├── data_collector.py      # HKO + Open-Meteo data fetcher
├── features.py            # Feature engineering
├── model.py               # Model training utilities
├── backtest_v5.py         # Backtest v5 (direct bucket classifier)
├── polymarket_scraper.py  # Polymarket price scraper
├── telegram_alert.py      # Hourly Telegram prediction alerts (v2)
├── model_tracker.py       # Model performance tracking
├── bias_corrector.py      # Multi-day bias correction (EWMA)
├── auto_retrain.py        # Auto-retrain pipeline (weekly retraining)
├── market_analyzer.py     # Market efficiency analysis
├── app.py                 # Streamlit GUI
├── nwp_collector.py       # Open-Meteo ensemble collector
├── data/
│   ├── raw/               # Raw downloaded data
│   └── processed/         # Processed data, backtest results, performance log
└── models/                # Trained model files
```

## Installation

```bash
pip install pandas numpy scipy scikit-learn lightgbm requests streamlit plotly
```

## Usage

### Run Streamlit Dashboard
```bash
python -m streamlit run app.py
```

### Run Telegram Alert (manual)
```bash
python telegram_alert.py
```

### Run Backtest
```bash
python backtest_v5.py
```

### Scrape Polymarket Prices
```bash
python polymarket_scraper.py
```

## Backtest Results (v5)

| Season | Win Rate | Top-2 | Top-3 |
|--------|----------|-------|-------|
| Summer (High) | 36.2% | 56.6% | 72.0% |
| Summer (Low) | 38.8% | 60.1% | 77.2% |

## Telegram Bot

Hourly alerts sent to Telegram with:
- Model predictions vs Polymarket prices
- Value bets (edge >5%) with MAIN + LOTTERY recommendations
- Kelly Criterion bet sizing suggestions
- Resolution status
- Ensemble NWP uncertainty quantification

## Notes

- Markets aktif Mei–Oktober setiap tahun
- HKO forecast sangat akurat untuk 1-3 hari ahead
- Model menggunakan 60% HKO forecast + 40% ensemble NWP (80 members)
- Predictions logged for performance tracking (Brier score, win rate, ROI)
