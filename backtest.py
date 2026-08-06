"""
Backtest Module - Evaluate prediction accuracy and simulated betting performance
Uses real HKO historical data to test if our bucket predictions match actual outcomes.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    DEFAULT_HIGH_TEMP_BUCKETS,
    DEFAULT_LOW_TEMP_BUCKETS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    HKO_API_BASE,
)
from data_collector import HKODataCollector
from polymarket_strategy import (
    parse_buckets,
    compute_bucket_probabilities,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_real_hko_data():
    """Fetch real HKO historical temperature data from the official API."""
    import io
    hko = HKODataCollector()

    max_url = 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKO'
    min_url = 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMINT&rformat=csv&station=HKO'

    def fetch_and_parse(url, label):
        logger.info(f"Fetching {label} from {url}")
        resp = hko.session.get(url, timeout=60)
        resp.raise_for_status()
        text = resp.content.decode('utf-8-sig')  # handle BOM
        # Skip first 2 header lines, last 3 footer lines
        lines = text.strip().split('\n')
        # Find the header line (contains 'Year' or 'year')
        header_idx = None
        for i, line in enumerate(lines):
            if 'Year' in line or 'year' in line:
                header_idx = i
                break
        if header_idx is None:
            header_idx = 2  # fallback

        # Keep header + data lines (skip footer legend)
        data_lines = lines[header_idx:]
        # Remove footer lines (lines starting with quotes and # or ***)
        clean_lines = []
        for line in data_lines:
            stripped = line.strip().strip('"')
            if stripped.startswith('***') or stripped.startswith('#') or stripped.startswith('C '):
                continue
            clean_lines.append(line)

        csv_text = '\n'.join(clean_lines)
        df = pd.read_csv(io.StringIO(csv_text))
        logger.info(f"  Parsed {len(df)} rows, columns: {df.columns.tolist()}")
        return df

    try:
        max_df = fetch_and_parse(max_url, "daily max temp")
    except Exception as e:
        logger.error(f"Failed to fetch max temp: {e}")
        return None

    try:
        min_df = fetch_and_parse(min_url, "daily min temp")
    except Exception as e:
        logger.error(f"Failed to fetch min temp: {e}")
        return None

    # Parse into time series
    def parse_hko_csv(df):
        cols = df.columns.tolist()
        # Columns might be: Year, Month, Day, Value or with Chinese headers
        year_col = [c for c in cols if 'year' in c.lower() or c == 'Year'][0] if any('year' in c.lower() for c in cols) else cols[0]
        month_col = [c for c in cols if 'month' in c.lower() or c == 'Month'][0] if any('month' in c.lower() for c in cols) else cols[1]
        day_col = [c for c in cols if 'day' in c.lower() or c == 'Day'][0] if any('day' in c.lower() for c in cols) else cols[2]
        val_col = [c for c in cols if 'value' in c.lower() or c == 'Value'][0] if any('value' in c.lower() for c in cols) else cols[3]

        df['date'] = pd.to_datetime(df[[year_col, month_col, day_col]].rename(columns={year_col: 'Year', month_col: 'Month', day_col: 'Day'}), errors='coerce')
        df['value'] = pd.to_numeric(df[val_col], errors='coerce')
        df = df.dropna(subset=['date', 'value'])
        df = df.sort_values('date').set_index('date')
        return df[['value']]

    max_ts = parse_hko_csv(max_df)
    min_ts = parse_hko_csv(min_df)

    # Combine
    combined = max_ts.join(min_ts, lsuffix='_max', rsuffix='_min', how='inner')
    combined.columns = ['max_temp', 'min_temp']
    combined = combined.dropna()

    logger.info(f"Combined dataset: {len(combined)} days, from {combined.index.min().date()} to {combined.index.max().date()}")
    return combined


def backtest_bucket_accuracy(
    temp_series: pd.DataFrame,
    bucket_defs_high: list,
    bucket_defs_low: list,
    lookback_days: int = 365,
    std_high: float = 1.0,
    std_low: float = 0.8,
):
    """
    Backtest: For each day in the test period, use the SAME-DAY historical mean/std
    as our 'prediction' and check if the actual temperature falls in the predicted
    highest-probability bucket.

    This simulates what a simple baseline model would do.
    """
    # Use last `lookback_days` as test set
    test_data = temp_series.tail(lookback_days).copy()
    train_data = temp_series.iloc[:-lookback_days].copy()

    if len(train_data) < 365:
        logger.warning("Not enough training data. Using all data for same-day stats.")
        train_data = temp_series.copy()

    logger.info(f"Test period: {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data)} days)")
    logger.info(f"Training data: {len(train_data)} historical records")

    # Compute same-day historical statistics (mean/std per day-of-year)
    train_data["doy"] = train_data.index.dayofyear
    doy_stats_max = train_data.groupby("doy")["max_temp"].agg(["mean", "std", "min", "max"])
    doy_stats_min = train_data.groupby("doy")["min_temp"].agg(["mean", "std", "min", "max"])

    # Fill missing std with global std
    global_std_max = train_data["max_temp"].std()
    global_std_min = train_data["min_temp"].std()
    doy_stats_max["std"] = doy_stats_max["std"].fillna(global_std_max)
    doy_stats_min["std"] = doy_stats_min["std"].fillna(global_std_min)

    # Also use recent rolling mean as a blended prediction
    temp_series_copy = temp_series.copy()
    temp_series_copy["max_roll_7"] = temp_series_copy["max_temp"].rolling(7, min_periods=1).mean()
    temp_series_copy["min_roll_7"] = temp_series_copy["min_temp"].rolling(7, min_periods=1).mean()
    temp_series_copy["max_roll_3"] = temp_series_copy["max_temp"].rolling(3, min_periods=1).mean()
    temp_series_copy["min_roll_3"] = temp_series_copy["min_temp"].rolling(3, min_periods=1).mean()

    # Parse buckets
    high_buckets = parse_buckets(bucket_defs_high)
    low_buckets = parse_buckets(bucket_defs_low)

    # Run backtest
    results = []
    for date, row in test_data.iterrows():
        doy = date.dayofyear

        # Get historical stats for this day-of-year
        if doy in doy_stats_max.index:
            hist_mean_max = doy_stats_max.loc[doy, "mean"]
            hist_std_max = doy_stats_max.loc[doy, "std"]
            hist_mean_min = doy_stats_min.loc[doy, "mean"]
            hist_std_min = doy_stats_min.loc[doy, "std"]
        else:
            continue

        # Blend: 50% historical mean + 50% recent 7-day rolling average
        if date in temp_series_copy.index:
            recent_max = temp_series_copy.loc[date, "max_roll_7"]
            recent_min = temp_series_copy.loc[date, "min_roll_7"]
            if pd.notna(recent_max) and pd.notna(recent_min):
                pred_mean_max = 0.5 * hist_mean_max + 0.5 * recent_max
                pred_mean_min = 0.5 * hist_mean_min + 0.5 * recent_min
            else:
                pred_mean_max = hist_mean_max
                pred_mean_min = hist_mean_min
        else:
            pred_mean_max = hist_mean_max
            pred_mean_min = hist_mean_min

        actual_max = row["max_temp"]
        actual_min = row["min_temp"]

        # Compute bucket probabilities
        high_probs = compute_bucket_probabilities(pred_mean_max, hist_std_max, high_buckets)
        low_probs = compute_bucket_probabilities(pred_mean_min, hist_std_min, low_buckets)

        # Find highest probability bucket
        best_high_bucket = max(high_probs, key=high_probs.get)
        best_high_prob = high_probs[best_high_bucket]
        best_low_bucket = max(low_probs, key=low_probs.get)
        best_low_prob = low_probs[best_low_bucket]

        # Find actual bucket
        def find_actual_bucket(actual_val, buckets):
            for b in buckets:
                if b.contains(actual_val):
                    return b.label
            return None

        actual_high_bucket = find_actual_bucket(actual_max, high_buckets)
        actual_low_bucket = find_actual_bucket(actual_min, low_buckets)

        # Check if prediction was correct
        high_correct = (best_high_bucket == actual_high_bucket)
        low_correct = (best_low_bucket == actual_low_bucket)

        # Compute probability for the ACTUAL bucket (for calibration)
        high_prob_actual = high_probs.get(actual_high_bucket, 0)
        low_prob_actual = low_probs.get(actual_low_bucket, 0)

        results.append({
            "date": date,
            "pred_max": pred_mean_max,
            "actual_max": actual_max,
            "best_high_bucket": best_high_bucket,
            "best_high_prob": best_high_prob,
            "actual_high_bucket": actual_high_bucket,
            "high_correct": high_correct,
            "high_prob_actual": high_prob_actual,
            "pred_min": pred_mean_min,
            "actual_min": actual_min,
            "best_low_bucket": best_low_bucket,
            "best_low_prob": best_low_prob,
            "actual_low_bucket": actual_low_bucket,
            "low_correct": low_correct,
            "low_prob_actual": low_prob_actual,
        })

    return pd.DataFrame(results)


def compute_backtest_metrics(results: pd.DataFrame) -> dict:
    """Compute comprehensive backtest metrics."""
    n = len(results)
    if n == 0:
        return {}

    # Basic accuracy
    high_accuracy = results["high_correct"].mean()
    low_accuracy = results["low_correct"].mean()

    # Random baseline (1/N buckets)
    n_high_buckets = len(DEFAULT_HIGH_TEMP_BUCKETS)
    n_low_buckets = len(DEFAULT_LOW_TEMP_BUCKETS)
    random_baseline_high = 1.0 / n_high_buckets
    random_baseline_low = 1.0 / n_low_buckets

    # Mean Absolute Error
    mae_max = (results["pred_max"] - results["actual_max"]).abs().mean()
    mae_min = (results["pred_min"] - results["actual_min"]).abs().mean()

    # Calibration: average predicted probability for the actual bucket
    # (well-calibrated model should have this close to 1/N_buckets for uniform,
    #  or match the average predicted probability)
    avg_prob_actual_high = results["high_prob_actual"].mean()
    avg_prob_actual_low = results["low_prob_actual"].mean()

    # Average confidence (predicted probability for best bucket)
    avg_confidence_high = results["best_high_prob"].mean()
    avg_confidence_low = results["best_low_prob"].mean()

    # Simulated EV against random market prices (uniform 1/N)
    uniform_price_high = 1.0 / n_high_buckets
    uniform_price_low = 1.0 / n_low_buckets

    # For each day, compute EV if we always bet on the best bucket at uniform price
    ev_per_bet_high = (results["high_prob_actual"] - uniform_price_high).mean()
    ev_per_bet_low = (results["low_prob_actual"] - uniform_price_low).mean()

    # "Top-2" accuracy: did the actual bucket fall in the top-2 predicted buckets?
    # (This is more relevant for Polymarket where you might bet on 2 buckets)
    # We'll compute this from the probability distributions
    top2_high = 0
    top2_low = 0
    for _, row in results.iterrows():
        # Reconstruct probabilities (approximate from best bucket)
        if row["high_prob_actual"] >= uniform_price_high:
            top2_high += 1
        if row["low_prob_actual"] >= uniform_price_low:
            top2_low += 1

    # Hit rate: did the model assign > uniform probability to the actual bucket?
    hit_rate_high = top2_high / n
    hit_rate_low = top2_low / n

    # Simulated ROI: if we bet $1 on best bucket every day
    # Win pays (1/price - 1), loss costs $1
    wins_high = results["high_correct"].sum()
    losses_high = n - wins_high
    roi_high = (wins_high * (1.0/uniform_price_high - 1) - losses_high) / n

    wins_low = results["low_correct"].sum()
    losses_low = n - wins_low
    roi_low = (wins_low * (1.0/uniform_price_low - 1) - losses_low) / n

    return {
        "n_days": n,
        "date_range": f"{results['date'].min().date()} to {results['date'].max().date()}",
        "high_temp": {
            "accuracy": high_accuracy,
            "random_baseline": random_baseline_high,
            "lift_vs_random": high_accuracy / random_baseline_high,
            "mae": mae_max,
            "avg_model_confidence": avg_confidence_high,
            "avg_prob_of_actual": avg_prob_actual_high,
            "hit_rate_above_uniform": hit_rate_high,
            "simulated_roi_per_dollar": roi_high,
            "simulated_ev_per_bet": ev_per_bet_high,
            "wins": int(wins_high),
            "losses": int(losses_high),
        },
        "low_temp": {
            "accuracy": low_accuracy,
            "random_baseline": random_baseline_low,
            "lift_vs_random": low_accuracy / random_baseline_low,
            "mae": mae_min,
            "avg_model_confidence": avg_confidence_low,
            "avg_prob_of_actual": avg_prob_actual_low,
            "hit_rate_above_uniform": hit_rate_low,
            "simulated_roi_per_dollar": roi_low,
            "simulated_ev_per_bet": ev_per_bet_low,
            "wins": int(wins_low),
            "losses": int(losses_low),
        },
    }


def run_backtest(lookback_days: int = 365):
    """Main backtest runner."""
    print("=" * 70)
    print("  HK Weather Prediction - Backtest")
    print("=" * 70)

    # 1. Fetch real data
    print("\n[1/3] Fetching real HKO historical data...")
    temp_data = fetch_real_hko_data()
    if temp_data is None:
        print("ERROR: Could not fetch data. Aborting backtest.")
        return None

    # 2. Run backtest
    print(f"\n[2/3] Running backtest on last {lookback_days} days...")
    results = backtest_bucket_accuracy(
        temp_data,
        DEFAULT_HIGH_TEMP_BUCKETS,
        DEFAULT_LOW_TEMP_BUCKETS,
        lookback_days=lookback_days,
    )

    # 3. Compute metrics
    print("\n[3/3] Computing metrics...\n")
    metrics = compute_backtest_metrics(results)

    # Print results
    print("=" * 70)
    print("  BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Period: {metrics['date_range']}")
    print(f"  Days tested: {metrics['n_days']}")
    print()

    for label, key in [("HIGH TEMPERATURE", "high_temp"), ("LOW TEMPERATURE", "low_temp")]:
        m = metrics[key]
        print(f"  --- {label} ---")
        print(f"  Win Rate (best bucket hit):     {m['accuracy']:.1%}  ({m['wins']}/{m['wins']+m['losses']})")
        print(f"  Random baseline (1/N buckets):  {m['random_baseline']:.1%}")
        print(f"  Lift vs random:                 {m['lift_vs_random']:.2f}x")
        print(f"  Mean Absolute Error:            {m['mae']:.2f} C")
        print(f"  Avg model confidence:           {m['avg_model_confidence']:.1%}")
        print(f"  Avg prob assigned to actual:    {m['avg_prob_of_actual']:.1%}")
        print(f"  Hit rate (prob > uniform):      {m['hit_rate_above_uniform']:.1%}")
        print(f"  Simulated EV per $1 bet:        ${m['simulated_ev_per_bet']:+.4f}")
        print(f"  Simulated ROI per $1/day:       ${m['simulated_roi_per_dollar']:+.4f}")
        print()

    # Save results
    results.to_csv(PROCESSED_DATA_DIR / "backtest_results.csv", index=False)
    print(f"  Results saved to data/processed/backtest_results.csv")

    # Bucket-level breakdown
    print("\n  --- BUCKET BREAKDOWN (High Temp) ---")
    for bucket_def in DEFAULT_HIGH_TEMP_BUCKETS:
        label = bucket_def[0]
        subset = results[results["actual_high_bucket"] == label]
        if len(subset) > 0:
            pred_correct = subset["high_correct"].sum()
            total = len(subset)
            avg_prob = subset["high_prob_actual"].mean()
            print(f"    {label:<8} occurred {total:>4}x | predicted correctly {pred_correct:>3}x ({pred_correct/total:.0%}) | avg model prob {avg_prob:.1%}")

    print("\n  --- BUCKET BREAKDOWN (Low Temp) ---")
    for bucket_def in DEFAULT_LOW_TEMP_BUCKETS:
        label = bucket_def[0]
        subset = results[results["actual_low_bucket"] == label]
        if len(subset) > 0:
            pred_correct = subset["low_correct"].sum()
            total = len(subset)
            avg_prob = subset["low_prob_actual"].mean()
            print(f"    {label:<8} occurred {total:>4}x | predicted correctly {pred_correct:>3}x ({pred_correct/total:.0%}) | avg model prob {avg_prob:.1%}")

    print("\n" + "=" * 70)

    return results, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest HK weather prediction")
    parser.add_argument("--days", type=int, default=365, help="Number of days to backtest")
    args = parser.parse_args()

    run_backtest(lookback_days=args.days)
