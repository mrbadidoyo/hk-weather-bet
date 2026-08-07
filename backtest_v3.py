"""
Backtest v3 - Fixed calibration using empirical same-day distributions
+ HKO forecast simulation + realistic Polymarket pricing
"""
import io
import logging
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d

from config import (
    DEFAULT_HIGH_TEMP_BUCKETS,
    DEFAULT_LOW_TEMP_BUCKETS,
    PROCESSED_DATA_DIR,
)
from data_collector import HKODataCollector
from polymarket_strategy import parse_buckets, compute_bucket_probabilities

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def fetch_hko_data():
    """Fetch real HKO historical temperature data."""
    hko = HKODataCollector()
    max_url = 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKO'
    min_url = 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMINT&rformat=csv&station=HKO'

    def fetch_and_parse(url):
        resp = hko.session.get(url, timeout=60)
        resp.raise_for_status()
        text = resp.content.decode('utf-8-sig')
        lines = text.strip().split('\n')
        header_idx = 2
        for i, line in enumerate(lines):
            if 'Year' in line:
                header_idx = i
                break
        clean = []
        for line in lines[header_idx:]:
            s = line.strip().strip('"')
            if s.startswith('***') or s.startswith('#') or s.startswith('C '):
                continue
            clean.append(line)
        return pd.read_csv(io.StringIO('\n'.join(clean)))

    def parse(df):
        cols = df.columns.tolist()
        yc = [c for c in cols if 'year' in c.lower()][0]
        mc = [c for c in cols if 'month' in c.lower()][0]
        dc = [c for c in cols if 'day' in c.lower()][0]
        vc = [c for c in cols if 'value' in c.lower()][0]
        df['date'] = pd.to_datetime(df[[yc, mc, dc]].rename(columns={yc:'Year', mc:'Month', dc:'Day'}), errors='coerce')
        df['value'] = pd.to_numeric(df[vc], errors='coerce')
        return df.dropna(subset=['date','value']).sort_values('date').set_index('date')[['value']]

    return parse(fetch_and_parse(max_url)), parse(fetch_and_parse(min_url))


class EmpiricalDistribution:
    """
    Empirical CDF-based distribution for same-day historical temperatures.
    Much better than Gaussian for capturing the actual shape (skewness, multimodality).
    """
    def __init__(self, values, min_samples=10):
        self.values = np.sort(values)
        self.n = len(values)
        self.min_samples = min_samples

        if self.n >= min_samples:
            # KDE for smooth density estimation
            self.kde = stats.gaussian_kde(values, bw_method='silverman')
            self.mean = np.mean(values)
            self.std = np.std(values)
            self.median = np.median(values)
            # Empirical quantiles
            self.quantiles = np.percentile(values, [5, 10, 25, 50, 75, 90, 95])
        else:
            self.kde = None
            self.mean = np.mean(values) if self.n > 0 else 0
            self.std = np.std(values) if self.n > 1 else 2.0
            self.median = self.mean
            self.quantiles = [self.mean - 2*self.std, self.mean - 1.5*self.std,
                              self.mean - 0.7*self.std, self.mean,
                              self.mean + 0.7*self.std, self.mean + 1.5*self.std,
                              self.mean + 2*self.std]

    def bucket_probabilities(self, bucket_defs):
        """Compute probability for each bucket using the empirical distribution."""
        buckets = parse_buckets(bucket_defs)
        probs = {}

        if self.kde is not None and self.n >= self.min_samples:
            # Use KDE integration for smooth probabilities
            for b in buckets:
                if b.is_open_lower:
                    lo, hi = -100, b.upper
                elif b.is_open_upper:
                    lo, hi = b.lower, 100
                else:
                    lo, hi = b.lower, b.upper

                # Integrate KDE over bucket range
                xs = np.linspace(lo, hi, 200)
                if b.is_open_lower:
                    xs = np.linspace(self.quantiles[0] - 3*self.std, b.upper, 200)
                elif b.is_open_upper:
                    xs = np.linspace(b.lower, self.quantiles[-1] + 3*self.std, 200)

                ys = self.kde(xs)
                probs[b.label] = np.trapezoid(ys, xs)
        else:
            # Fallback to Gaussian
            for b in buckets:
                if b.is_open_lower:
                    probs[b.label] = stats.norm.cdf(b.upper, self.mean, self.std)
                elif b.is_open_upper:
                    probs[b.label] = 1.0 - stats.norm.cdf(b.lower, self.mean, self.std)
                else:
                    probs[b.label] = (stats.norm.cdf(b.upper, self.mean, self.std)
                                      - stats.norm.cdf(b.lower, self.mean, self.std))

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        return probs

    def predict_mean(self):
        return self.mean

    def predict_std(self):
        return self.std


class CalibratedPredictor:
    """
    Predicts temperature using a blend of:
    1. Empirical same-day historical distribution (climatological baseline)
    2. Recent rolling average (captures short-term weather patterns)
    3. Persistence forecast (yesterday's temp)
    
    The key insight: use the EMPIRICAL distribution shape, not Gaussian.
    Then shift the distribution mean based on recent conditions.
    """
    def __init__(self, train_df, temp_col, window_doy=5):
        self.temp_col = temp_col
        self.window_doy = window_doy
        self.train = train_df.copy()
        self.train['doy'] = self.train.index.dayofyear
        self.train['month'] = self.train.index.month

        # Pre-compute rolling stats on full series
        self.train[f'{temp_col}_roll7'] = self.train[temp_col].rolling(7, min_periods=1).mean()
        self.train[f'{temp_col}_roll3'] = self.train[temp_col].rolling(3, min_periods=1).mean()
        self.train[f'{temp_col}_lag1'] = self.train[temp_col].shift(1)
        self.train[f'{temp_col}_lag2'] = self.train[temp_col].shift(2)

        # Build empirical distributions per day-of-year (with window)
        self.doy_distributions = {}
        for doy in range(1, 367):
            # Window: doy ± window_doy days
            mask = (self.train['doy'] >= doy - window_doy) & (self.train['doy'] <= doy + window_doy)
            # Handle wrap-around for Jan/Dec
            if doy - window_doy < 1:
                mask |= (self.train['doy'] >= 366 + doy - window_doy)
            if doy + window_doy > 366:
                mask |= (self.train['doy'] <= doy + window_doy - 366)
            values = self.train.loc[mask, temp_col].values
            self.doy_distributions[doy] = EmpiricalDistribution(values)

        # Build monthly distributions for bucket frequency baseline
        self.monthly_distributions = {}
        for month in range(1, 13):
            values = self.train[self.train['month'] == month][temp_col].values
            self.monthly_distributions[month] = EmpiricalDistribution(values)

    def predict(self, date, recent_data=None):
        """
        Predict distribution for a given date.
        
        Returns:
            probs: dict of bucket -> probability
            pred_mean: predicted mean
            pred_std: predicted std
        """
        doy = date.timetuple().tm_yday

        # Base distribution from same-day historical
        base_dist = self.doy_distributions.get(doy, self.doy_distributions.get(180))

        # If we have recent data, shift the distribution mean
        shift = 0.0
        if recent_data is not None:
            recent_vals = recent_data.get(self.temp_col, [])
            if len(recent_vals) >= 3:
                recent_mean = np.mean(recent_vals[-7:])
                recent_3 = np.mean(recent_vals[-3:])
                # Shift = how much recent deviates from historical
                anomaly = recent_mean - base_dist.mean
                # Blend: trust recent data partially
                shift = 0.6 * anomaly
                # Also factor in persistence (last 3 days trend)
                trend = recent_3 - recent_mean
                shift += 0.2 * trend

        # Apply shift to get adjusted distribution
        adjusted_mean = base_dist.mean + shift
        adjusted_std = base_dist.std

        # Use recent volatility if available
        if recent_data is not None and len(recent_data.get(self.temp_col, [])) >= 7:
            recent_std = np.std(recent_data[self.temp_col][-14:])
            adjusted_std = 0.6 * base_dist.std + 0.4 * recent_std
            adjusted_std = max(adjusted_std, 0.5)

        return adjusted_mean, adjusted_std, base_dist

    def bucket_probs(self, date, bucket_defs, recent_data=None):
        """Get bucket probabilities for a date."""
        adj_mean, adj_std, base_dist = self.predict(date, recent_data)

        # Method 1: Shift the empirical distribution
        # Instead of recomputing KDE, shift the base distribution's bucket boundaries
        # P(X in [a,b] | shifted by s) = P(X-s in [a,b]) = P(X in [a-s, b-s])
        shift = adj_mean - base_dist.mean

        shifted_buckets = []
        for label, lo, hi in bucket_defs:
            shifted_buckets.append((label, lo - shift, hi - shift))

        probs = base_dist.bucket_probabilities(shifted_buckets)

        # Method 2: Also compute Gaussian probs for comparison
        gauss_buckets = parse_buckets(bucket_defs)
        gauss_probs = compute_bucket_probabilities(adj_mean, adj_std, gauss_buckets)

        # Blend: 70% empirical, 30% Gaussian (smooths out noise)
        blended = {}
        for label in probs:
            blended[label] = 0.70 * probs.get(label, 0) + 0.30 * gauss_probs.get(label, 0)

        # Normalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v/total for k, v in blended.items()}

        return blended, adj_mean, adj_std


def run_backtest_v3(years_back=5):
    """Run calibrated backtest with realistic market simulation."""
    print("=" * 72)
    print("  Backtest v3: Calibrated Empirical Distribution + Realistic Pricing")
    print("=" * 72)

    # Fetch data
    print("\n[1/5] Fetching HKO data...")
    max_ts, min_ts = fetch_hko_data()
    combined = max_ts.join(min_ts, lsuffix='_max', rsuffix='_min', how='inner')
    combined.columns = ['max_temp', 'min_temp']
    combined = combined.dropna()
    print(f"  {len(combined)} days: {combined.index.min().date()} to {combined.index.max().date()}")

    # Split
    test_end = combined.index.max()
    test_start = test_end - pd.DateOffset(years=years_back)
    train = combined[:test_start].copy()
    test = combined[test_start:test_end].copy()
    print(f"  Train: {len(train)} | Test: {len(test)}")

    # Build predictors
    print("\n[2/5] Building calibrated predictors...")
    max_predictor = CalibratedPredictor(train, 'max_temp', window_doy=5)
    min_predictor = CalibratedPredictor(train, 'min_temp', window_doy=5)

    # Run predictions
    print("\n[3/5] Running predictions on test set...")
    max_results = []
    min_results = []

    for date, row in test.iterrows():
        # Get recent data (last 14 days before this date)
        recent_start = date - pd.Timedelta(days=14)
        recent_slice = combined.loc[recent_start:date - pd.Timedelta(days=1)]

        recent_data = {
            'max_temp': recent_slice['max_temp'].tolist(),
            'min_temp': recent_slice['min_temp'].tolist(),
        }

        # High temp prediction
        h_probs, h_mean, h_std = max_predictor.bucket_probs(date, DEFAULT_HIGH_TEMP_BUCKETS, recent_data)
        h_best = max(h_probs, key=h_probs.get)

        actual_max = row['max_temp']
        h_actual_bucket = None
        for label, lo, hi in DEFAULT_HIGH_TEMP_BUCKETS:
            b = parse_buckets([(label, lo, hi)])[0]
            if b.contains(actual_max):
                h_actual_bucket = label
                break

        max_results.append({
            'date': date, 'month': date.month, 'actual': actual_max,
            'pred_mean': h_mean, 'pred_std': h_std,
            'best_bucket': h_best, 'best_prob': h_probs[h_best],
            'actual_bucket': h_actual_bucket,
            'correct': h_best == h_actual_bucket,
            'prob_actual': h_probs.get(h_actual_bucket, 0),
            'all_probs': h_probs,
        })

        # Low temp prediction
        l_probs, l_mean, l_std = min_predictor.bucket_probs(date, DEFAULT_LOW_TEMP_BUCKETS, recent_data)
        l_best = max(l_probs, key=l_probs.get)

        actual_min = row['min_temp']
        l_actual_bucket = None
        for label, lo, hi in DEFAULT_LOW_TEMP_BUCKETS:
            b = parse_buckets([(label, lo, hi)])[0]
            if b.contains(actual_min):
                l_actual_bucket = label
                break

        min_results.append({
            'date': date, 'month': date.month, 'actual': actual_min,
            'pred_mean': l_mean, 'pred_std': l_std,
            'best_bucket': l_best, 'best_prob': l_probs[l_best],
            'actual_bucket': l_actual_bucket,
            'correct': l_best == l_actual_bucket,
            'prob_actual': l_probs.get(l_actual_bucket, 0),
            'all_probs': l_probs,
        })

    max_df = pd.DataFrame(max_results)
    min_df = pd.DataFrame(min_results)

    # Analyze per season
    print("\n[4/5] Seasonal analysis with realistic market simulation...\n")

    seasons = [
        ('Summer (JJA)', [6, 7, 8]),
        ('Polymarket (May-Oct)', [5, 6, 7, 8, 9, 10]),
        ('Autumn (SON)', [9, 10, 11]),
        ('Winter (DJF)', [12, 1, 2]),
        ('Spring (MAM)', [3, 4, 5]),
        ('Full Year', list(range(1, 13))),
    ]

    for label, results_df in [("HIGH TEMP", max_df), ("LOW TEMP", min_df)]:
        bucket_defs = DEFAULT_HIGH_TEMP_BUCKETS if 'HIGH' in label else DEFAULT_LOW_TEMP_BUCKETS
        n_buckets = len(bucket_defs)

        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")

        print(f"\n  {'Season':<28} {'Days':>5} {'WinR':>6} {'Top2':>6} {'Top3':>6} {'MAE':>7} {'Brier':>7} {'ROI(u)':>8} {'ROI(r)':>8}")
        print(f"  {'-'*84}")

        for season_name, months in seasons:
            mask = results_df['month'].isin(months)
            sub = results_df[mask]
            if len(sub) == 0:
                continue

            n = len(sub)
            wins = sub['correct'].sum()
            wr = wins / n
            mae = (sub['pred_mean'] - sub['actual']).abs().mean()
            brier = ((sub['prob_actual'] - 1.0)**2).mean()

            # Top-2 and Top-3 accuracy
            top2 = 0
            top3 = 0
            for _, r in sub.iterrows():
                sorted_b = sorted(r['all_probs'].items(), key=lambda x: x[1], reverse=True)
                top_labels = [x[0] for x in sorted_b[:3]]
                if r['actual_bucket'] in top_labels[:2]:
                    top2 += 1
                if r['actual_bucket'] in top_labels[:3]:
                    top3 += 1

            # ROI at uniform prices
            uniform = 1.0 / n_buckets
            roi_uniform = (wins * (1/uniform - 1) - (n - wins)) / n

            # ROI at realistic prices (historical frequency for that month)
            # For each bet, the "market price" = historical frequency of that bucket in that month
            roi_realistic = 0
            for _, r in sub.iterrows():
                month = r['month']
                best_b = r['best_bucket']
                # Historical freq of this bucket for this month
                month_mask = results_df['month'] == month
                month_data = results_df[month_mask]
                freq = (month_data['actual_bucket'] == best_b).mean()
                market_price = max(freq, 0.02)  # floor at 2%
                if r['correct']:
                    roi_realistic += (1/market_price - 1) / n
                else:
                    roi_realistic -= 1.0 / n

            print(f"  {season_name:<28} {n:>5} {wr:>5.1%} {top2/n:>5.1%} {top3/n:>5.1%} "
                  f"{mae:>6.2f}C {brier:>6.3f} {roi_uniform:>+7.3f} {roi_realistic:>+7.3f}")

        # Summer bucket breakdown
        summer = results_df[results_df['month'].isin([6, 7, 8])]
        if len(summer) > 0:
            print(f"\n  --- SUMMER BUCKET DETAIL ({label}) ---")
            print(f"  {'Bucket':<10} {'Actual':>7} {'Freq':>6} {'Model':>7} {'M.WR':>6} {'Edge':>7} {'ROI(r)':>8}")
            print(f"  {'-'*55}")

            for label_b, lo, hi in bucket_defs:
                actual_n = (summer['actual_bucket'] == label_b).sum()
                freq = actual_n / len(summer) if len(summer) > 0 else 0
                model_picks = (summer['best_bucket'] == label_b).sum()
                model_correct = summer[(summer['best_bucket'] == label_b) & (summer['correct'])].shape[0]
                model_wr = model_correct / model_picks if model_picks > 0 else 0
                edge = model_wr - freq

                # ROI at realistic price
                market_price = max(freq, 0.02)
                roi = model_wr * (1/market_price - 1) - (1 - model_wr) if model_picks > 0 else 0

                print(f"  {label_b:<10} {actual_n:>7} {freq:>5.1%} {model_picks:>7} "
                      f"{model_wr:>5.0%} {edge:>+6.1%} {roi:>+7.2f}")

        # Calibration check: are predicted probabilities matching actual frequencies?
        print(f"\n  --- CALIBRATION CHECK ({label}, Summer) ---")
        prob_bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]
        print(f"  {'Pred Prob':>10} {'Count':>6} {'Actual Freq':>12} {'Calibration':>12}")
        for lo, hi in prob_bins:
            mask_prob = (summer['best_prob'] >= lo) & (summer['best_prob'] < hi)
            sub = summer[mask_prob]
            if len(sub) > 0:
                actual_freq = sub['correct'].mean()
                avg_pred = sub['best_prob'].mean()
                print(f"  {lo:.1f}-{hi:.1f}     {len(sub):>6} {actual_freq:>11.1%} "
                      f"{'well calibrated' if abs(actual_freq - avg_pred) < 0.1 else 'MISCALIBRATED':>15}")

    # Final summary
    print(f"\n{'='*72}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*72}")

    for label, results_df in [("HIGH TEMP", max_df), ("LOW TEMP", min_df)]:
        summer = results_df[results_df['month'].isin([6, 7, 8])]
        n = len(summer)
        if n == 0:
            continue
        wins = summer['correct'].sum()
        wr = wins / n
        mae = (summer['pred_mean'] - summer['actual']).abs().mean()

        top2 = sum(1 for _, r in summer.iterrows()
                   if r['actual_bucket'] in sorted(r['all_probs'].items(), key=lambda x: x[1], reverse=True)[:2][0][0]
                   or r['actual_bucket'] == sorted(r['all_probs'].items(), key=lambda x: x[1], reverse=True)[1][0])
        # Simpler top-2
        top2_count = 0
        for _, r in summer.iterrows():
            sb = sorted(r['all_probs'].items(), key=lambda x: x[1], reverse=True)
            if r['actual_bucket'] in [sb[0][0], sb[1][0]]:
                top2_count += 1

        print(f"\n  {label} (Summer):")
        print(f"    Best bucket win rate:  {wr:.1%} ({wins}/{n})")
        print(f"    Top-2 accuracy:        {top2_count/n:.1%}")
        print(f"    MAE:                   {mae:.2f}C")
        print(f"    vs Random (1/7):       {wr/(1/7):.2f}x better")

    print(f"\n{'='*72}")

    # Save
    max_df.to_csv(PROCESSED_DATA_DIR / 'backtest_v3_high.csv', index=False)
    min_df.to_csv(PROCESSED_DATA_DIR / 'backtest_v3_low.csv', index=False)

    return max_df, min_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()
    run_backtest_v3(years_back=args.years)
