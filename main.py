"""
HK Weather Prediction for Polymarket Betting
Main CLI runner - orchestrates data collection, model training, and bet analysis
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    DEFAULT_HIGH_TEMP_BUCKETS, DEFAULT_LOW_TEMP_BUCKETS,
)
from data_collector import HKODataCollector, WundergroundCollector, build_combined_dataset
from features import build_feature_matrix
from model import HKWeatherEnsemble
from polymarket_strategy import (
    analyze_market, format_analysis, format_multi_analysis, parse_buckets,
    compute_bucket_probabilities,
)

logger = logging.getLogger(__name__)


def cmd_collect(args):
    """Collect and prepare all weather data"""
    logger.info("=" * 70)
    logger.info("  STEP 1: Data Collection")
    logger.info("=" * 70)

    hko = HKODataCollector()

    # Fetch HKO historical data
    logger.info("\nFetching HKO historical datasets...")
    hko_datasets = hko.fetch_all_historical()
    hko.save_historical_data(hko_datasets)

    # Fetch HKO forecast
    logger.info("\nFetching HKO 9-day forecast...")
    try:
        forecast = hko.fetch_9day_forecast()
        forecast_path = RAW_DATA_DIR / "hko_9day_forecast.json"
        with open(forecast_path, "w") as f:
            json.dump(forecast, f, indent=2)
        logger.info(f"  Saved forecast -> {forecast_path}")
    except Exception as e:
        logger.warning(f"  Forecast fetch failed: {e}")
        forecast = None

    # Fetch Wunderground VHHH data
    logger.info("\nFetching Weather Underground VHHH data...")
    wu = WundergroundCollector()
    wu_path = RAW_DATA_DIR / "wunderground_vhhh.csv"

    if wu_path.exists() and not args.force:
        existing = pd.read_csv(wu_path, parse_dates=["date"])
        last_date = existing["date"].max()
        if (datetime.now() - last_date).days < 3:
            logger.info(f"  WU data up to date (last: {last_date.date()})")
            wu_df = existing
        else:
            start = last_date + timedelta(days=1)
            new_df = wu.fetch_date_range(start, datetime.now())
            wu.save_data(new_df)
            wu_df = pd.read_csv(wu_path, parse_dates=["date"])
    else:
        end = datetime.now()
        start = end - timedelta(days=args.days)
        wu_df = wu.fetch_date_range(start, end)
        wu.save_data(wu_df)

    # Build combined dataset
    logger.info("\nBuilding combined dataset...")
    combined = build_combined_dataset(hko_datasets, wu_df)
    combined_path = PROCESSED_DATA_DIR / "combined.csv"
    combined.to_csv(combined_path)
    logger.info(f"  Saved combined dataset -> {combined_path} ({len(combined)} rows)")

    print(f"\n✓ Data collection complete. {len(combined)} records.")
    return combined


def cmd_train(args):
    """Train prediction models"""
    logger.info("=" * 70)
    logger.info("  STEP 2: Model Training")
    logger.info("=" * 70)

    # Load combined dataset
    combined_path = PROCESSED_DATA_DIR / "combined.csv"
    if not combined_path.exists():
        logger.error("No combined dataset found. Run 'collect' first.")
        return None

    raw_df = pd.read_csv(combined_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(raw_df)} records with {len(raw_df.columns)} features")

    # Build feature matrix
    features = build_feature_matrix(raw_df)
    features_path = PROCESSED_DATA_DIR / "features.csv"
    features.to_csv(features_path)

    target_max = features.attrs.get("target_col_max", "wu_max_temp_c")
    target_min = features.attrs.get("target_col_min", "wu_min_temp_c")

    # Train ensemble
    ensemble = HKWeatherEnsemble()
    metrics = ensemble.fit(features, target_max, target_min)

    # Save models
    ensemble.save()

    print(f"\n✓ Models trained successfully!")
    print(f"  High temp MAE: {metrics['max_temp']['mae']:.2f}°C")
    print(f"  Low temp MAE: {metrics['min_temp']['mae']:.2f}°C")

    return ensemble, metrics


def cmd_predict(args):
    """Generate predictions and betting analysis"""
    logger.info("=" * 70)
    logger.info("  STEP 3: Prediction & Betting Analysis")
    logger.info("=" * 70)

    # Load models
    ensemble = HKWeatherEnsemble()
    try:
        ensemble.load()
    except FileNotFoundError:
        logger.error("No trained models found. Run 'train' first.")
        return None

    # Load feature data
    features_path = PROCESSED_DATA_DIR / "features.csv"
    if features_path.exists():
        features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    else:
        logger.error("No features found. Run 'collect' and 'train' first.")
        return None

    # Load HKO forecast
    forecast_path = RAW_DATA_DIR / "hko_9day_forecast.json"
    hko_forecast = None
    if forecast_path.exists():
        with open(forecast_path) as f:
            hko_forecast = json.load(f)

    # Predict for target dates
    target_dates = _get_target_dates(args)
    logger.info(f"Generating predictions for {len(target_dates)} dates...")

    from polymarket_strategy import MarketAnalysis

    all_analyses = []

    for target_date in target_dates:
        # Get features for this date
        if target_date in features.index:
            day_features = features.loc[[target_date]]
        else:
            # Use most recent features
            day_features = features.tail(1)

        # Get HKO forecast for this date
        hko_max, hko_min = _get_hko_forecast_for_date(hko_forecast, target_date)

        # Generate predictions
        preds = ensemble.predict(day_features, hko_max, hko_min)

        max_dist = preds["max_temp"]
        min_dist = preds["min_temp"]

        # Get market prices (placeholder - in production, fetch from Polymarket API)
        high_prices = _get_market_prices(target_date, "high", args)
        low_prices = _get_market_prices(target_date, "low", args)

        # Analyze high temp market
        if high_prices:
            high_analysis = analyze_market(
                date=target_date.strftime("%Y-%m-%d"),
                market_type="high",
                predicted_mean=max_dist["mean"].iloc[0],
                predicted_std=max_dist["std"].iloc[0],
                market_prices=high_prices,
                quantile_data={
                    "p10": max_dist["p10"].iloc[0],
                    "p25": max_dist["p25"].iloc[0],
                    "p50": max_dist["p50"].iloc[0],
                    "p75": max_dist["p75"].iloc[0],
                    "p90": max_dist["p90"].iloc[0],
                },
                confidence=0.75,
            )
            all_analyses.append(high_analysis)

        # Analyze low temp market
        if low_prices:
            low_analysis = analyze_market(
                date=target_date.strftime("%Y-%m-%d"),
                market_type="low",
                predicted_mean=min_dist["mean"].iloc[0],
                predicted_std=min_dist["std"].iloc[0],
                market_prices=low_prices,
                bucket_defs=DEFAULT_LOW_TEMP_BUCKETS,
                quantile_data={
                    "p10": min_dist["p10"].iloc[0],
                    "p25": min_dist["p25"].iloc[0],
                    "p50": min_dist["p50"].iloc[0],
                    "p75": min_dist["p75"].iloc[0],
                    "p90": min_dist["p90"].iloc[0],
                },
                confidence=0.75,
            )
            all_analyses.append(low_analysis)

    # Print results
    if all_analyses:
        print(format_multi_analysis(all_analyses))
    else:
        # Just show predictions without market comparison
        print("\n  PREDICTIONS (no market prices provided):")
        print(f"  {'Date':<12} {'High (°C)':>10} {'Low (°C)':>10} {'Uncertainty':>12}")
        print(f"  {'-' * 44}")
        for target_date in target_dates:
            if target_date in features.index:
                day_features = features.loc[[target_date]]
            else:
                day_features = features.tail(1)
            hko_max, hko_min = _get_hko_forecast_for_date(hko_forecast, target_date)
            preds = ensemble.predict(day_features, hko_max, hko_min)
            max_pred = preds["max_temp"]["mean"].iloc[0]
            max_std = preds["max_temp"]["std"].iloc[0]
            min_pred = preds["min_temp"]["mean"].iloc[0]
            print(f"  {target_date.strftime('%Y-%m-%d'):<12} {max_pred:>9.1f}±{max_std:.1f} {min_pred:>9.1f} {'±':>12}{max_std:.1f}")

    return all_analyses


def cmd_full_pipeline(args):
    """Run the complete pipeline: collect -> train -> predict"""
    print("\n" + "=" * 70)
    print("  HK WEATHER PREDICTION FOR POLYMARKET BETTING")
    print("  Full Pipeline Execution")
    print("=" * 70 + "\n")

    cmd_collect(args)
    cmd_train(args)
    cmd_predict(args)


def cmd_quick_predict(args):
    """Quick prediction using HKO forecast + historical patterns (no ML training)"""
    print("\n  Quick Temperature Prediction for Hong Kong")
    print("  " + "-" * 45)

    hko = HKODataCollector()

    # Get HKO forecast
    try:
        forecast = hko.fetch_9day_forecast()
    except Exception as e:
        print(f"  Failed to fetch HKO forecast: {e}")
        return

    today = datetime.now()
    print(f"\n  Date: {today.strftime('%Y-%m-%d')}")
    print(f"\n  HKO 9-Day Forecast:")
    print(f"  {'Date':<12} {'High (°C)':>10} {'Low (°C)':>10} {'Weather':<20}")
    print(f"  {'-' * 52}")

    for day in forecast.get("weatherForecast", []):
        date = day.get("forecastDate", "")
        max_t = day.get("forecastMaxtemp", {}).get("value", "N/A")
        min_t = day.get("forecastMintemp", {}).get("value", "N/A")
        weather = day.get("forecastWeather", "")[:20]
        print(f"  {date:<12} {str(max_t):>10} {str(min_t):>10} {weather:<20}")

    # Generate probability distributions based on forecast + historical uncertainty
    print(f"\n  Probability Distributions (for Polymarket):")
    print(f"\n  HIGH TEMPERATURE:")
    for day in forecast.get("weatherForecast", [])[:3]:  # next 3 days
        date = day.get("forecastDate", "")
        max_t = float(day.get("forecastMaxtemp", {}).get("value", 32))
        std = 1.0  # HKO forecast is usually accurate within ±1°C

        probs = compute_bucket_probabilities(max_t, std, parse_buckets(DEFAULT_HIGH_TEMP_BUCKETS))
        print(f"\n  {date} (forecast: {max_t}°C +/- {std}°C):")
        for label, prob in probs.items():
            bar = '#' * int(prob * 30)
            print(f"    {label:<8} {prob:>6.1%}  {bar}")

    print(f"\n  LOW TEMPERATURE:")
    for day in forecast.get("weatherForecast", [])[:3]:
        date = day.get("forecastDate", "")
        min_t = float(day.get("forecastMintemp", {}).get("value", 27))
        std = 0.8

        probs = compute_bucket_probabilities(min_t, std, parse_buckets(DEFAULT_LOW_TEMP_BUCKETS))
        print(f"\n  {date} (forecast: {min_t}°C +/- {std}°C):")
        for label, prob in probs.items():
            bar = '#' * int(prob * 30)
            print(f"    {label:<8} {prob:>6.1%}  {bar}")


def _get_target_dates(args) -> list:
    """Get list of target dates for prediction"""
    dates = []
    start = datetime.now()
    if args.date:
        start = datetime.strptime(args.date, "%Y-%m-%d")

    for i in range(args.days):
        dates.append(start + timedelta(days=i))

    return dates


def _get_hko_forecast_for_date(forecast: dict, target_date: datetime) -> tuple:
    """Extract HKO forecast for a specific date"""
    if forecast is None:
        return None, None

    date_str = target_date.strftime("%Y%m%d")
    for day in forecast.get("weatherForecast", []):
        if day.get("forecastDate") == date_str:
            try:
                return (
                    float(day["forecastMaxtemp"]["value"]),
                    float(day["forecastMintemp"]["value"]),
                )
            except (KeyError, TypeError, ValueError):
                pass
    return None, None


def _get_market_prices(target_date: datetime, market_type: str, args) -> dict | None:
    """
    Get Polymarket prices for a specific date and market type.
    In production, this would fetch from Polymarket API.
    For now, use mock prices or read from file.
    """
    # Check if prices file exists
    prices_path = RAW_DATA_DIR / "market_prices.json"
    if prices_path.exists():
        with open(prices_path) as f:
            all_prices = json.load(f)
        key = f"{target_date.strftime('%Y-%m-%d')}_{market_type}"
        if key in all_prices:
            return all_prices[key]

    # Return None if no prices available (user can add prices file)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="HK Weather Temperature Prediction for Polymarket Betting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py collect              # Collect HKO + WU data
  python main.py train                # Train prediction models
  python main.py predict              # Generate predictions
  python main.py run                  # Full pipeline
  python main.py quick                # Quick forecast check (no training)
  python main.py predict --date 2026-08-10 --days 5  # Predict specific dates
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect weather data")
    collect_parser.add_argument("--days", type=int, default=730, help="Days of WU history to fetch")
    collect_parser.add_argument("--force", action="store_true", help="Force re-fetch all data")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train prediction models")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Generate predictions")
    predict_parser.add_argument("--date", type=str, help="Start date (YYYY-MM-DD)")
    predict_parser.add_argument("--days", type=int, default=7, help="Number of days to predict")

    # Run (full pipeline) command
    run_parser = subparsers.add_parser("run", help="Full pipeline: collect -> train -> predict")
    run_parser.add_argument("--days", type=int, default=730, help="Days of WU history to fetch")
    run_parser.add_argument("--force", action="store_true", help="Force re-fetch")

    # Quick command
    quick_parser = subparsers.add_parser("quick", help="Quick forecast check")

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "run":
        cmd_full_pipeline(args)
    elif args.command == "quick":
        cmd_quick_predict(args)
    else:
        parser.print_help()
        print("\nQuick start: python main.py quick")


if __name__ == "__main__":
    main()
