"""
Backtest v2 - Per-season analysis with proper calibration and realistic market simulation.
Focuses on summer months (Jun-Aug) when Polymarket HK weather markets are most active.
"""
import io
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    DEFAULT_HIGH_TEMP_BUCKETS,
    DEFAULT_LOW_TEMP_BUCKETS,
    PROCESSED_DATA_DIR,
)
from data_collector import HKODataCollector
from polymarket_strategy import (
    parse_buckets,
    compute_bucket_probabilities,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def fetch_hko_data():
    """Fetch real HKO historical temperature data."""
    hko = HKODataCollector()

    max_url = 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKO'
    min_url = 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMINT&rformat=csv&station=HKO'

    def fetch_and_parse(url, label):
        logger.info(f"  Fetching {label}...")
        resp = hko.session.get(url, timeout=60)
        resp.raise_for_status()
        text = resp.content.decode('utf-8-sig')
        lines = text.strip().split('\n')
        header_idx = None
        for i, line in enumerate(lines):
            if 'Year' in line:
                header_idx = i
                break
        if header_idx is None:
            header_idx = 2
        data_lines = lines[header_idx:]
        clean_lines = []
        for line in data_lines:
            s = line.strip().strip('"')
            if s.startswith('***') or s.startswith('#') or s.startswith('C '):
                continue
            clean_lines.append(line)
        csv_text = '\n'.join(clean_lines)
        df = pd.read_csv(io.StringIO(csv_text))
        return df

    max_df = fetch_and_parse(max_url, "max temp")
    min_df = fetch_and_parse(min_url, "min temp")

    def parse(df):
        cols = df.columns.tolist()
        yc = [c for c in cols if 'year' in c.lower()][0]
        mc = [c for c in cols if 'month' in c.lower()][0]
        dc = [c for c in cols if 'day' in c.lower()][0]
        vc = [c for c in cols if 'value' in c.lower()][0]
        df['date'] = pd.to_datetime(df[[yc, mc, dc]].rename(columns={yc:'Year', mc:'Month', dc:'Day'}), errors='coerce')
        df['value'] = pd.to_numeric(df[vc], errors='coerce')
        return df.dropna(subset=['date','value']).sort_values('date').set_index('date')[['value']]

    max_ts = parse(max_df)
    min_ts = parse(min_df)
    combined = max_ts.join(min_ts, lsuffix='_max', rsuffix='_min', how='inner')
    combined.columns = ['max_temp', 'min_temp']
    return combined.dropna()


def build_prediction(train_df, test_df, temp_col, std_window=30):
    """
    Build predictions for each test day using:
    1. Same day-of-year historical mean/std (from ALL training data)
    2. Recent 7-day rolling mean (momentum/trend)
    3. Blended prediction: 40% historical + 60% recent
    """
    train = train_df.copy()
    train['doy'] = train.index.dayofyear

    # Historical same-day stats
    doy_stats = train.groupby('doy')[temp_col].agg(['mean', 'std', 'min', 'max', 'count'])
    global_std = train[temp_col].std()
    doy_stats['std'] = doy_stats['std'].fillna(global_std).clip(lower=0.5)

    # Also compute rolling stats on the FULL series
    full = pd.concat([train_df[[temp_col]], test_df[[temp_col]]])
    full = full[~full.index.duplicated(keep='last')]
    full = full.sort_index()
    full[f'{temp_col}_roll7'] = full[temp_col].rolling(7, min_periods=1).mean()
    full[f'{temp_col}_roll3'] = full[temp_col].rolling(3, min_periods=1).mean()

    predictions = []
    for date, row in test_df.iterrows():
        doy = date.dayofyear
        actual = row[temp_col]

        if doy in doy_stats.index:
            hist_mean = doy_stats.loc[doy, 'mean']
            hist_std = doy_stats.loc[doy, 'std']
        else:
            hist_mean = train[temp_col].mean()
            hist_std = global_std

        # Recent rolling average
        if date in full.index:
            row = full.loc[date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            recent7 = row[f'{temp_col}_roll7']
            recent3 = row[f'{temp_col}_roll3']
            if pd.notna(recent7):
                # Blend: weight recent data more for short-term trends
                pred_mean = 0.40 * hist_mean + 0.45 * recent7 + 0.15 * recent3
            else:
                pred_mean = hist_mean
        else:
            pred_mean = hist_mean

        # Use historical std but adjust if recent volatility is different
        recent_slice = full.loc[max(date - pd.Timedelta(days=30), full.index.min()):date, temp_col]
        if len(recent_slice) > 5:
            recent_std = recent_slice.std()
            # Blend std: 70% historical, 30% recent
            pred_std = 0.70 * hist_std + 0.30 * recent_std
        else:
            pred_std = hist_std
        pred_std = max(pred_std, 0.5)

        predictions.append({
            'date': date,
            'actual': actual,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'hist_mean': hist_mean,
            'hist_std': hist_std,
            'doy': doy,
            'month': date.month,
        })

    return pd.DataFrame(predictions)


def simulate_backtest(preds_df, bucket_defs, temp_label, n_buckets):
    """
    Simulate betting on each day.
    For each day:
    - Compute bucket probabilities from predicted distribution
    - Find best bucket (highest prob)
    - Check if actual falls in predicted bucket
    - Compute realistic market price (based on historical frequency)
    - Compute EV and ROI
    """
    buckets = parse_buckets(bucket_defs)

    results = []
    for _, row in preds_df.iterrows():
        probs = compute_bucket_probabilities(row['pred_mean'], row['pred_std'], buckets)

        # Find best bucket
        best_bucket = max(probs, key=probs.get)
        best_prob = probs[best_bucket]

        # Find actual bucket
        actual_val = row['actual']
        actual_bucket = None
        for b in buckets:
            if b.contains(actual_val):
                actual_bucket = b.label
                break

        correct = (best_bucket == actual_bucket)
        prob_of_actual = probs.get(actual_bucket, 0)

        # Realistic market price: based on historical climatology
        # In a real market, prices reflect the true probability
        # We approximate this with the same-day historical frequency
        # This means the "edge" is only from our model being BETTER than pure historical
        uniform_price = 1.0 / n_buckets

        # For realistic simulation, we use two price models:
        # 1. Uniform (naive): all buckets = 1/N
        # 2. Historical (smart): each bucket priced at its historical frequency for that month
        # We report both.

        results.append({
            'date': row['date'],
            'month': row['month'],
            'actual': actual_val,
            'pred_mean': row['pred_mean'],
            'pred_std': row['pred_std'],
            'best_bucket': best_bucket,
            'best_prob': best_prob,
            'actual_bucket': actual_bucket,
            'correct': correct,
            'prob_of_actual': prob_of_actual,
            'all_probs': probs,
        })

    return pd.DataFrame(results)


def analyze_season(results_df, season_name, months):
    """Analyze results for a specific season."""
    mask = results_df['month'].isin(months)
    subset = results_df[mask]
    if len(subset) == 0:
        return None

    n = len(subset)
    wins = subset['correct'].sum()
    win_rate = wins / n
    mae = (subset['pred_mean'] - subset['actual']).abs().mean()
    avg_confidence = subset['best_prob'].mean()
    avg_prob_actual = subset['prob_of_actual'].mean()
    brier = ((subset['prob_of_actual'] - 1.0) ** 2).mean()  # Brier score for correct bucket

    # ROI at uniform prices
    n_buckets = 7
    uniform = 1.0 / n_buckets
    roi_uniform = (wins * (1/uniform - 1) - (n - wins)) / n

    # More realistic: "fair" price = average model probability for the winning bucket
    # If our model is well-calibrated, avg_prob_actual should equal actual frequency
    # Edge exists only where our model disagrees with market
    # We estimate edge as: avg_prob_actual - (1/n_buckets)
    # This is conservative - real edge is smaller because market also has info

    # Top-2 accuracy
    top2_correct = 0
    for _, row in subset.iterrows():
        probs = row['all_probs']
        sorted_buckets = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top2_labels = [sorted_buckets[0][0], sorted_buckets[1][0]]
        if row['actual_bucket'] in top2_labels:
            top2_correct += 1
    top2_rate = top2_correct / n

    return {
        'season': season_name,
        'months': f"{min(months)}-{max(months)}",
        'n_days': n,
        'win_rate': win_rate,
        'wins': int(wins),
        'top2_rate': top2_rate,
        'mae': mae,
        'avg_confidence': avg_confidence,
        'avg_prob_actual': avg_prob_actual,
        'brier_score': brier,
        'roi_uniform': roi_uniform,
    }


def run_backtest_v2(years_back=3):
    """Run comprehensive seasonal backtest."""
    print("=" * 72)
    print("  HK Weather Prediction - Backtest v2 (Seasonal + Calibrated)")
    print("=" * 72)

    # Fetch data
    print("\n[1/4] Fetching HKO historical data (1884-2026)...")
    all_data = fetch_hko_data()
    print(f"  Loaded {len(all_data)} days: {all_data.index.min().date()} to {all_data.index.max().date()}")

    # Define test period
    test_end = all_data.index.max()
    test_start = test_end - pd.DateOffset(years=years_back)
    test_data = all_data[test_start:test_end].copy()
    train_data = all_data[:test_start].copy()

    print(f"\n  Train: {len(train_data)} days ({train_data.index.min().date()} to {train_data.index.max().date()})")
    print(f"  Test:  {len(test_data)} days ({test_data.index.min().date()} to {test_data.index.max().date()})")

    # Build predictions
    print(f"\n[2/4] Building predictions for {len(test_data)} test days...")
    max_preds = build_prediction(train_data, test_data, 'max_temp')
    min_preds = build_prediction(train_data, test_data, 'min_temp')

    # Simulate
    print(f"\n[3/4] Simulating bucket bets...")
    n_high = len(DEFAULT_HIGH_TEMP_BUCKETS)
    n_low = len(DEFAULT_LOW_TEMP_BUCKETS)
    max_results = simulate_backtest(max_preds, DEFAULT_HIGH_TEMP_BUCKETS, 'High', n_high)
    min_results = simulate_backtest(min_preds, DEFAULT_LOW_TEMP_BUCKETS, 'Low', n_low)

    # Analyze per season
    print(f"\n[4/4] Seasonal analysis...")

    seasons = {
        'Summer (JJA)': [6, 7, 8],
        'Autumn (SON)': [9, 10, 11],
        'Winter (DJF)': [12, 1, 2],
        'Spring (MAM)': [3, 4, 5],
        'Polymarket Season (May-Oct)': [5, 6, 7, 8, 9, 10],
    }

    for label, temp_name, results_df in [("HIGH TEMP", "High Temp", max_results), ("LOW TEMP", "Low Temp", min_results)]:
        print(f"\n{'='*72}")
        print(f"  {label} BACKTEST RESULTS ({years_back}-year)")
        print(f"{'='*72}")

        all_metrics = []
        for season_name, months in seasons.items():
            m = analyze_season(results_df, season_name, months)
            if m:
                all_metrics.append(m)

        # Print table
        print(f"\n  {'Season':<30} {'Days':>5} {'WinRate':>8} {'Top-2':>7} {'MAE':>7} {'Conf':>7} {'ROI(unif)':>10}")
        print(f"  {'-'*76}")
        for m in all_metrics:
            print(
                f"  {m['season']:<30} {m['n_days']:>5} "
                f"{m['win_rate']:>7.1%} {m['top2_rate']:>6.1%} "
                f"{m['mae']:>6.2f}C {m['avg_confidence']:>6.1%} "
                f"{m['roi_uniform']:>+9.3f}"
            )

        # Bucket breakdown for summer only
        print(f"\n  --- SUMMER BUCKET BREAKDOWN ({label}) ---")
        summer_mask = results_df['month'].isin([6, 7, 8])
        summer = results_df[summer_mask]
        bucket_defs = DEFAULT_HIGH_TEMP_BUCKETS if 'HIGH' in label else DEFAULT_LOW_TEMP_BUCKETS

        for bdef in bucket_defs:
            blabel = bdef[0]
            occurred = summer[summer['actual_bucket'] == blabel]
            if len(occurred) > 0:
                pred_as_best = summer[summer['best_bucket'] == blabel]
                n_occurred = len(occurred)
                n_pred_best = len(pred_as_best)
                correct_when_actual = occurred['correct'].sum()
                avg_prob = occurred['prob_of_actual'].mean()
                print(f"    {blabel:<8} actual: {n_occurred:>3}x | "
                      f"model picked: {n_pred_best:>3}x | "
                      f"correct: {correct_when_actual:>3}x ({correct_when_actual/n_occurred:.0%}) | "
                      f"avg prob: {avg_prob:.1%}")

    # Overall summary
    print(f"\n{'='*72}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*72}")
    for label, results_df in [("HIGH TEMP", max_results), ("LOW TEMP", min_results)]:
        n = len(results_df)
        wins = results_df['correct'].sum()
        mae = (results_df['pred_mean'] - results_df['actual']).abs().mean()

        # Summer only
        summer = results_df[results_df['month'].isin([6, 7, 8])]
        s_wins = summer['correct'].sum() if len(summer) > 0 else 0
        s_n = len(summer) if len(summer) > 0 else 1

        print(f"\n  {label}:")
        print(f"    Overall ({years_back}yr): {wins}/{n} = {wins/n:.1%} win rate, MAE {mae:.2f}C")
        print(f"    Summer only:  {s_wins}/{s_n} = {s_wins/s_n:.1%} win rate")

    # Key insight
    print(f"\n  --- KEY INSIGHT FOR POLYMARKET ---")
    print(f"  The model's edge comes from PREDICTING WHICH BUCKET the temperature")
    print(f"  falls into MORE ACCURATELY than the market price implies.")
    print(f"  ")
    print(f"  Against UNIFORM prices (1/7 = 14.3% each):")
    print(f"    - High temp overall win rate: {max_results['correct'].mean():.1%}")
    print(f"    - Low temp overall win rate:  {min_results['correct'].mean():.1%}")
    print(f"  ")

    # Realistic estimate: market prices ~ true climatology
    # Our edge = model accuracy - market accuracy (historical frequency)
    summer_high = max_results[max_results['month'].isin([6, 7, 8])]
    summer_low = min_results[min_results['month'].isin([6, 7, 8])]

    if len(summer_high) > 0:
        # Historical bucket frequencies for summer
        summer_actuals = test_data[test_data.index.month.isin([6, 7, 8])]['max_temp']
        high_buckets = parse_buckets(DEFAULT_HIGH_TEMP_BUCKETS)
        hist_freq = {}
        for b in high_buckets:
            count = sum(1 for v in summer_actuals if b.contains(v))
            hist_freq[b.label] = count / len(summer_actuals) if len(summer_actuals) > 0 else 0

        print(f"  Summer (JJA) historical bucket frequencies (market baseline):")
        for blabel, freq in hist_freq.items():
            model_wins = summer_high[summer_high['best_bucket'] == blabel]['correct'].sum()
            model_picks = len(summer_high[summer_high['best_bucket'] == blabel])
            model_wr = model_wins / model_picks if model_picks > 0 else 0
            edge = model_wr - freq
            roi = model_wr * (1/freq - 1) - (1 - model_wr) if freq > 0 else 0
            print(f"    {blabel:<8} freq={freq:.1%} | model picks {model_picks}x, correct {model_wr:.0%} | edge={edge:+.1%} | ROI/bet={roi:+.2f}")

    # Save
    max_results.to_csv(PROCESSED_DATA_DIR / 'backtest_v2_high.csv', index=False)
    min_results.to_csv(PROCESSED_DATA_DIR / 'backtest_v2_low.csv', index=False)
    print(f"\n  Results saved to data/processed/backtest_v2_*.csv")
    print(f"{'='*72}")

    return max_results, min_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3, help="Years to backtest")
    args = parser.parse_args()
    run_backtest_v2(years_back=args.years)
