"""
Backtest v4 — Full improvement pipeline:
1. HKA Airport data (matches Polymarket VHHH resolution)
2. Bias correction layer
3. Multi-bucket strategy (top-2 betting)
4. NWP ensemble blending (live mode) / simulated NWP signal (backtest mode)
"""
import io
import logging
import sys
from pathlib import Path
from collections import defaultdict

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


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────

def fetch_hka_data():
    """Fetch HKO Airport (HKA) historical temperature — matches VHHH."""
    hko = HKODataCollector()
    urls = {
        'max': 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKA',
        'min': 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMINT&rformat=csv&station=HKA',
    }

    def fetch(url, label):
        logger.info(f"  Fetching HKA {label}...")
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
        df['date'] = pd.to_datetime(df[[yc, mc, dc]].rename(columns={yc:'Year', mc:'Month', dc:'Day'}), errors='coerce')
        df['value'] = pd.to_numeric(df[vc], errors='coerce')
        return df.dropna(subset=['date','value']).sort_values('date').set_index('date')[['value']]

    max_df = parse(fetch(urls['max'], 'max'))
    min_df = parse(fetch(urls['min'], 'min'))

    combined = max_df.join(min_df, lsuffix='_max', rsuffix='_min', how='inner')
    combined.columns = ['max_temp', 'min_temp']
    return combined.dropna()


def fetch_hko_data():
    """Fetch HKO HQ historical data (for comparison)."""
    hko = HKODataCollector()
    urls = {
        'max': 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKO',
        'min': 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMINT&rformat=csv&station=HKO',
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
        df['date'] = pd.to_datetime(df[[yc, mc, dc]].rename(columns={yc:'Year', mc:'Month', dc:'Day'}), errors='coerce')
        df['value'] = pd.to_numeric(df[vc], errors='coerce')
        return df.dropna(subset=['date','value']).sort_values('date').set_index('date')[['value']]

    max_df = parse(fetch(urls['max']))
    min_df = parse(fetch(urls['min']))
    combined = max_df.join(min_df, lsuffix='_max', rsuffix='_min', how='inner')
    combined.columns = ['max_temp', 'min_temp']
    return combined.dropna()


# ──────────────────────────────────────────────────────────────
# Empirical Distribution (same as v3)
# ──────────────────────────────────────────────────────────────

class EmpiricalDistribution:
    def __init__(self, values, min_samples=10):
        self.values = np.sort(values)
        self.n = len(values)
        if self.n >= min_samples:
            self.kde = stats.gaussian_kde(values, bw_method='silverman')
            self.mean = np.mean(values)
            self.std = np.std(values)
            self.quantiles = np.percentile(values, [5, 10, 25, 50, 75, 90, 95])
        else:
            self.kde = None
            self.mean = np.mean(values) if self.n > 0 else 0
            self.std = np.std(values) if self.n > 1 else 2.0
            self.quantiles = [self.mean - 2*self.std] * 7

    def bucket_probabilities(self, bucket_defs):
        buckets = parse_buckets(bucket_defs)
        probs = {}
        if self.kde is not None and self.n >= 10:
            for b in buckets:
                if b.is_open_lower:
                    xs = np.linspace(self.quantiles[0] - 3*self.std, b.upper, 200)
                elif b.is_open_upper:
                    xs = np.linspace(b.lower, self.quantiles[-1] + 3*self.std, 200)
                else:
                    xs = np.linspace(b.lower, b.upper, 200)
                ys = self.kde(xs)
                probs[b.label] = np.trapezoid(ys, xs)
        else:
            for b in buckets:
                if b.is_open_lower:
                    probs[b.label] = stats.norm.cdf(b.upper, self.mean, self.std)
                elif b.is_open_upper:
                    probs[b.label] = 1.0 - stats.norm.cdf(b.lower, self.mean, self.std)
                else:
                    probs[b.label] = (stats.norm.cdf(b.upper, self.mean, self.std)
                                      - stats.norm.cdf(b.lower, self.mean, self.std))
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        return probs


# ──────────────────────────────────────────────────────────────
# Bias Correction Layer
# ──────────────────────────────────────────────────────────────

class BiasCorrector:
    """
    Learns systematic bias from cross-validated residuals and corrects predictions.
    """
    def __init__(self):
        self.bias_by_doy = {}       # bias per day-of-year (smoothed)
        self.bias_by_month = {}     # bias per month
        self.std_correction = {}    # std adjustment per month

    def fit(self, train_df, temp_col, window_doy=5):
        """
        Learn bias from training data using leave-one-year-out cross-validation.
        For each year in training, predict using the other years' data, compute residuals.
        """
        train = train_df.copy()
        train['doy'] = train.index.dayofyear
        train['year'] = train.index.year
        train['month'] = train.index.month

        all_residuals = []

        years = sorted(train['year'].unique())
        # Use last 5 years for bias estimation (faster)
        for test_year in years[-5:]:
            yr_train = train[train['year'] != test_year]
            yr_test = train[train['year'] == test_year]

            # Build DOY stats from training years
            doy_stats = yr_train.groupby('doy')[temp_col].agg(['mean', 'std'])
            doy_stats['std'] = doy_stats['std'].clip(lower=0.5)

            for date, row in yr_test.iterrows():
                doy = date.dayofyear
                if doy in doy_stats.index:
                    pred = doy_stats.loc[doy, 'mean']
                    residual = row[temp_col] - pred
                    all_residuals.append({
                        'date': date,
                        'doy': doy,
                        'month': date.month,
                        'residual': residual,
                        'pred': pred,
                        'actual': row[temp_col],
                    })

        residuals_df = pd.DataFrame(all_residuals)

        # Per-month bias
        for month in range(1, 13):
            mask = residuals_df['month'] == month
            subset = residuals_df[mask]
            if len(subset) > 20:
                self.bias_by_month[month] = subset['residual'].mean()
                self.std_correction[month] = subset['residual'].std() / max(subset['pred'].std(), 0.01)

        # Per-DOY bias (smoothed with window)
        for doy in range(1, 367):
            mask = (residuals_df['doy'] >= doy - window_doy) & (residuals_df['doy'] <= doy + window_doy)
            if doy - window_doy < 1:
                mask |= (residuals_df['doy'] >= 366 + doy - window_doy)
            subset = residuals_df[mask]
            if len(subset) > 15:
                self.bias_by_doy[doy] = subset['residual'].mean()

    def correct_mean(self, pred_mean, doy, month):
        """Apply bias correction to predicted mean."""
        doy_bias = self.bias_by_doy.get(doy, 0)
        month_bias = self.bias_by_month.get(month, 0)
        # Weighted blend: 60% DOY-specific, 40% monthly
        correction = 0.6 * doy_bias + 0.4 * month_bias
        return pred_mean + correction

    def correct_std(self, pred_std, month):
        """Adjust predicted std based on historical calibration."""
        factor = self.std_correction.get(month, 1.0)
        factor = np.clip(factor, 0.7, 1.5)  # don't over-correct
        return pred_std * factor


# ──────────────────────────────────────────────────────────────
# Improved Predictor
# ──────────────────────────────────────────────────────────────

class ImprovedPredictor:
    """
    Combines empirical distribution + recent trend + bias correction.
    """
    def __init__(self, train_df, temp_col, window_doy=5):
        self.temp_col = temp_col
        self.window_doy = window_doy
        self.train = train_df.copy()
        self.train['doy'] = self.train.index.dayofyear
        self.train['month'] = self.train.index.month

        # Pre-compute rolling
        self.train[f'{temp_col}_roll7'] = self.train[temp_col].rolling(7, min_periods=1).mean()
        self.train[f'{temp_col}_roll3'] = self.train[temp_col].rolling(3, min_periods=1).mean()

        # Build DOY distributions
        self.doy_distributions = {}
        for doy in range(1, 367):
            mask = (self.train['doy'] >= doy - window_doy) & (self.train['doy'] <= doy + window_doy)
            if doy - window_doy < 1:
                mask |= (self.train['doy'] >= 366 + doy - window_doy)
            if doy + window_doy > 366:
                mask |= (self.train['doy'] <= doy + window_doy - 366)
            values = self.train.loc[mask, temp_col].values
            self.doy_distributions[doy] = EmpiricalDistribution(values)

        # Build and fit bias corrector
        self.bias_corrector = BiasCorrector()
        self.bias_corrector.fit(train_df, temp_col, window_doy)

    def predict(self, date, recent_data=None, bucket_defs=None):
        """
        Returns blended, bias-corrected bucket probabilities + stats.
        """
        doy = date.timetuple().tm_yday
        month = date.month
        base_dist = self.doy_distributions.get(doy, self.doy_distributions.get(180))

        # Recent trend shift
        shift = 0.0
        if recent_data is not None and len(recent_data.get(self.temp_col, [])) >= 3:
            recent_vals = recent_data[self.temp_col]
            recent_mean = np.mean(recent_vals[-7:])
            recent_3 = np.mean(recent_vals[-3:])
            anomaly = recent_mean - base_dist.mean
            shift = 0.6 * anomaly
            trend = recent_3 - recent_mean
            shift += 0.2 * trend

        # Raw mean before bias correction
        raw_mean = base_dist.mean + shift

        # Apply bias correction
        corrected_mean = self.bias_corrector.correct_mean(raw_mean, doy, month)
        pred_std = base_dist.std
        pred_std = self.bias_corrector.correct_std(pred_std, month)

        # Blend with recent volatility
        if recent_data is not None and len(recent_data.get(self.temp_col, [])) >= 7:
            recent_std = np.std(recent_data[self.temp_col][-14:])
            pred_std = 0.6 * pred_std + 0.4 * recent_std
            pred_std = max(pred_std, 0.5)

        # Get bucket probabilities
        if bucket_defs:
            # Shift distribution
            distribution_shift = corrected_mean - base_dist.mean
            shifted_buckets = [(label, lo - distribution_shift, hi - distribution_shift)
                               for label, lo, hi in bucket_defs]
            emp_probs = base_dist.bucket_probabilities(shifted_buckets)

            # Also Gaussian for smoothing
            gauss_buckets = parse_buckets(bucket_defs)
            gauss_probs = compute_bucket_probabilities(corrected_mean, pred_std, gauss_buckets)

            # Blend 70% empirical + 30% Gaussian
            blended = {}
            for label in emp_probs:
                blended[label] = 0.70 * emp_probs.get(label, 0) + 0.30 * gauss_probs.get(label, 0)
            total = sum(blended.values())
            if total > 0:
                blended = {k: v/total for k, v in blended.items()}
            return blended, corrected_mean, pred_std

        return None, corrected_mean, pred_std


# ──────────────────────────────────────────────────────────────
# Multi-Bucket Strategy
# ──────────────────────────────────────────────────────────────

def multi_bucket_bet(probs, market_prices=None, n_buckets=7):
    """
    Multi-bucket betting: bet on top-2 buckets when combined prob > threshold.
    Returns list of (bucket, prob, action) tuples.
    """
    sorted_buckets = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    uniform_price = 1.0 / n_buckets

    bets = []
    for label, prob in sorted_buckets[:2]:
        price = market_prices.get(label, uniform_price) if market_prices else uniform_price
        edge = prob - price
        if edge > 0.03 and prob > price * 1.1:  # 3% min edge, 10% above market
            kelly = edge / (1/price - 1) if price > 0 else 0
            kelly = min(kelly, 0.08)  # cap at 8% bankroll
            bets.append({
                'bucket': label,
                'prob': prob,
                'price': price,
                'edge': edge,
                'kelly': kelly,
                'action': 'BUY',
            })

    return bets


# ──────────────────────────────────────────────────────────────
# Main Backtest
# ──────────────────────────────────────────────────────────────

def run_backtest_v4(years_back=5):
    print("=" * 72)
    print("  Backtest v4: Airport Data + Bias Correction + Multi-Bucket")
    print("=" * 72)

    # ── Fetch data ──
    print("\n[1/6] Fetching HKA Airport data...")
    try:
        hka_data = fetch_hka_data()
        print(f"  HKA (Airport): {len(hka_data)} days ({hka_data.index.min().date()} to {hka_data.index.max().date()})")
    except Exception as e:
        print(f"  ERROR fetching HKA data: {e}")
        print("  Falling back to HKO HQ data...")
        hka_data = fetch_hko_data()

    print("\n  Fetching HKO HQ data (for comparison)...")
    hko_data = fetch_hko_data()
    print(f"  HKO (HQ): {len(hko_data)} days ({hko_data.index.min().date()} to {hko_data.index.max().date()})")

    # ── Compare stations ──
    common = hka_data.index.intersection(hko_data.index)
    if len(common) > 100:
        hka_max = hka_data.loc[common, 'max_temp']
        hko_max = hko_data.loc[common, 'max_temp']
        offset = (hko_max - hka_max).mean()
        print(f"\n  Station offset (HKO-HKA): max temp {offset:+.2f}C (HKO is warmer)")

        # Summer offset
        summer_mask = common.month.isin([6,7,8])
        if summer_mask.sum() > 50:
            summer_offset = (hko_max[summer_mask] - hka_max[summer_mask]).mean()
            print(f"  Summer offset: {summer_offset:+.2f}C")

    # ── Split ──
    test_end = hka_data.index.max()
    test_start = test_end - pd.DateOffset(years=years_back)
    train = hka_data[:test_start].copy()
    test = hka_data[test_start:test_end].copy()
    print(f"\n  Train: {len(train)} | Test: {len(test)}")

    # ── Build predictors ──
    print("\n[2/6] Building improved predictors with bias correction...")
    max_pred = ImprovedPredictor(train, 'max_temp', window_doy=5)
    min_pred = ImprovedPredictor(train, 'min_temp', window_doy=5)

    print(f"  Bias correction learned:")
    print(f"    High temp monthly bias: ", end="")
    for m in [6,7,8]:
        print(f"  M{m}:{max_pred.bias_corrector.bias_by_month.get(m, 0):+.2f}C", end="")
    print()
    print(f"    Low temp monthly bias:  ", end="")
    for m in [6,7,8]:
        print(f"  M{m}:{min_pred.bias_corrector.bias_by_month.get(m, 0):+.2f}C", end="")
    print()

    # ── Run predictions ──
    print(f"\n[3/6] Predicting {len(test)} test days...")
    max_results = []
    min_results = []

    for date, row in test.iterrows():
        recent_start = date - pd.Timedelta(days=14)
        recent_slice = hka_data.loc[recent_start:date - pd.Timedelta(days=1)]
        recent_data = {'max_temp': recent_slice['max_temp'].tolist(), 'min_temp': recent_slice['min_temp'].tolist()}

        # High temp
        h_probs, h_mean, h_std = max_pred.predict(date, recent_data, DEFAULT_HIGH_TEMP_BUCKETS)
        h_best = max(h_probs, key=h_probs.get)
        actual_max = row['max_temp']
        h_actual_bucket = None
        for label, lo, hi in DEFAULT_HIGH_TEMP_BUCKETS:
            b = parse_buckets([(label, lo, hi)])[0]
            if b.contains(actual_max):
                h_actual_bucket = label
                break

        # Top-2
        sorted_h = sorted(h_probs.items(), key=lambda x: x[1], reverse=True)
        h_top2 = [sorted_h[0][0], sorted_h[1][0]]

        max_results.append({
            'date': date, 'month': date.month, 'actual': actual_max,
            'pred_mean': h_mean, 'pred_std': h_std,
            'best_bucket': h_best, 'best_prob': h_probs[h_best],
            'actual_bucket': h_actual_bucket,
            'correct': h_best == h_actual_bucket,
            'correct_top2': h_actual_bucket in h_top2,
            'prob_actual': h_probs.get(h_actual_bucket, 0),
            'all_probs': h_probs,
        })

        # Low temp
        l_probs, l_mean, l_std = min_pred.predict(date, recent_data, DEFAULT_LOW_TEMP_BUCKETS)
        l_best = max(l_probs, key=l_probs.get)
        actual_min = row['min_temp']
        l_actual_bucket = None
        for label, lo, hi in DEFAULT_LOW_TEMP_BUCKETS:
            b = parse_buckets([(label, lo, hi)])[0]
            if b.contains(actual_min):
                l_actual_bucket = label
                break

        sorted_l = sorted(l_probs.items(), key=lambda x: x[1], reverse=True)
        l_top2 = [sorted_l[0][0], sorted_l[1][0]]

        min_results.append({
            'date': date, 'month': date.month, 'actual': actual_min,
            'pred_mean': l_mean, 'pred_std': l_std,
            'best_bucket': l_best, 'best_prob': l_probs[l_best],
            'actual_bucket': l_actual_bucket,
            'correct': l_best == l_actual_bucket,
            'correct_top2': l_actual_bucket in l_top2,
            'prob_actual': l_probs.get(l_actual_bucket, 0),
            'all_probs': l_probs,
        })

    max_df = pd.DataFrame(max_results)
    min_df = pd.DataFrame(min_results)

    # ── Seasonal analysis ──
    print(f"\n[4/6] Seasonal analysis...")

    seasons = [
        ('Summer (JJA)', [6, 7, 8]),
        ('Polymarket (May-Oct)', [5, 6, 7, 8, 9, 10]),
        ('Full Year', list(range(1, 13))),
    ]

    for label, results_df in [("HIGH TEMP", max_df), ("LOW TEMP", min_df)]:
        bucket_defs = DEFAULT_HIGH_TEMP_BUCKETS if 'HIGH' in label else DEFAULT_LOW_TEMP_BUCKETS
        n_buckets = len(bucket_defs)

        print(f"\n{'='*72}")
        print(f"  {label} — v4 Results (HKA Airport + Bias Correction)")
        print(f"{'='*72}")

        print(f"\n  {'Season':<28} {'Days':>5} {'Top-1':>7} {'Top-2':>7} {'Top-3':>7} {'MAE':>7} {'ROI(u)':>8}")
        print(f"  {'-'*72}")

        for season_name, months in seasons:
            mask = results_df['month'].isin(months)
            sub = results_df[mask]
            if len(sub) == 0:
                continue

            n = len(sub)
            top1 = sub['correct'].sum() / n
            top2 = sub['correct_top2'].sum() / n
            mae = (sub['pred_mean'] - sub['actual']).abs().mean()

            # Top-3
            top3_count = 0
            for _, r in sub.iterrows():
                sb = sorted(r['all_probs'].items(), key=lambda x: x[1], reverse=True)
                if r['actual_bucket'] in [sb[0][0], sb[1][0], sb[2][0]]:
                    top3_count += 1
            top3 = top3_count / n

            uniform = 1.0 / n_buckets
            roi_uniform = (sub['correct'].sum() * (1/uniform - 1) - (n - sub['correct'].sum())) / n

            print(f"  {season_name:<28} {n:>5} {top1:>6.1%} {top2:>6.1%} {top3:>6.1%} "
                  f"{mae:>6.2f}C {roi_uniform:>+7.3f}")

        # Summer bucket detail
        summer = results_df[results_df['month'].isin([6, 7, 8])]
        if len(summer) > 0:
            print(f"\n  --- SUMMER BUCKET DETAIL ({label}) ---")
            print(f"  {'Bucket':<10} {'Actual':>7} {'Freq':>6} {'Model':>7} {'M.WR':>6} {'Edge':>7} {'ROI':>8}")
            print(f"  {'-'*55}")

            for blabel, lo, hi in bucket_defs:
                actual_n = (summer['actual_bucket'] == blabel).sum()
                freq = actual_n / len(summer) if len(summer) > 0 else 0
                model_picks = (summer['best_bucket'] == blabel).sum()
                model_correct = summer[(summer['best_bucket'] == blabel) & (summer['correct'])].shape[0]
                model_wr = model_correct / model_picks if model_picks > 0 else 0
                edge = model_wr - freq
                market_price = max(freq, 0.02)
                roi = model_wr * (1/market_price - 1) - (1 - model_wr) if model_picks > 0 else 0

                print(f"  {blabel:<10} {actual_n:>7} {freq:>5.1%} {model_picks:>7} "
                      f"{model_wr:>5.0%} {edge:>+6.1%} {roi:>+7.2f}")

    # ── Multi-bucket simulation ──
    print(f"\n[5/6] Multi-bucket strategy simulation...")

    for label, results_df in [("HIGH TEMP", max_df), ("LOW TEMP", min_df)]:
        summer = results_df[results_df['month'].isin([6, 7, 8])]
        if len(summer) == 0:
            continue

        bucket_defs = DEFAULT_HIGH_TEMP_BUCKETS if 'HIGH' in label else DEFAULT_LOW_TEMP_BUCKETS
        n_buckets = len(bucket_defs)

        # Simulate multi-bucket betting: bet top-2 every day
        total_pnl = 0
        n_bets = 0
        n_wins = 0

        for _, row in summer.iterrows():
            probs = row['all_probs']
            sorted_b = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top2 = sorted_b[:2]

            for bucket_label, prob in top2:
                # Simulate fair market price = historical frequency for this month
                month = row['month']
                month_mask = summer['month'] == month
                freq = (summer[month_mask]['actual_bucket'] == bucket_label).mean()
                price = max(freq, 0.05)

                n_bets += 1
                if row['actual_bucket'] == bucket_label:
                    payout = 1.0 / price - 1.0
                    total_pnl += payout
                    n_wins += 1
                else:
                    total_pnl -= 1.0

        avg_pnl = total_pnl / max(n_bets, 1)
        wr = n_wins / max(n_bets, 1)

        print(f"\n  {label} Multi-Bucket (Summer):")
        print(f"    Total bets: {n_bets}")
        print(f"    Win rate: {wr:.1%}")
        print(f"    Avg PnL per $1 bet: ${avg_pnl:+.3f}")
        print(f"    Total PnL ({len(summer)} days, 2 bets/day): ${total_pnl:+.1f}")

    # ── v3 vs v4 comparison ──
    print(f"\n[6/6] v3 vs v4 Comparison...")

    # Load v3 results for comparison
    v3_high_path = PROCESSED_DATA_DIR / 'backtest_v3_high.csv'
    v3_low_path = PROCESSED_DATA_DIR / 'backtest_v3_low.csv'

    if v3_high_path.exists() and v3_low_path.exists():
        v3_high = pd.read_csv(v3_high_path)
        v3_low = pd.read_csv(v3_low_path)

        print(f"\n  {'Metric':<25} {'v3 (HKO)':>12} {'v4 (HKA+Bias)':>15} {'Delta':>10}")
        print(f"  {'-'*64}")

        for label, v3_df, v4_df in [("High Temp", v3_high, max_df), ("Low Temp", v3_low, min_df)]:
            for months, season_label in [([6,7,8], "Summer"), (list(range(1,13)), "Full Year")]:
                v3_sub = v3_df[v3_df['month'].isin(months)]
                v4_sub = v4_df[v4_df['month'].isin(months)]

                v3_wr = v3_sub['correct'].mean() if len(v3_sub) > 0 else 0
                v4_wr = v4_sub['correct'].mean() if len(v4_sub) > 0 else 0

                v3_mae = (v3_sub['pred_mean'] - v3_sub['actual']).abs().mean() if len(v3_sub) > 0 else 0
                v4_mae = (v4_sub['pred_mean'] - v4_sub['actual']).abs().mean() if len(v4_sub) > 0 else 0

                delta_wr = v4_wr - v3_wr
                delta_mae = v4_mae - v3_mae

                print(f"  {season_label} {label} WR:   {v3_wr:>10.1%} {v4_wr:>13.1%} {delta_wr:>+9.1%}")
                print(f"  {season_label} {label} MAE:  {v3_mae:>10.2f}C {v4_mae:>12.2f}C {delta_mae:>+8.2f}C")

                # Top-2
                v4_t2 = v4_sub['correct_top2'].mean() if len(v4_sub) > 0 else 0
                print(f"  {season_label} {label} Top2: {'N/A':>10} {v4_t2:>13.1%}")
            print()

    # Save
    max_df.to_csv(PROCESSED_DATA_DIR / 'backtest_v4_high.csv', index=False)
    min_df.to_csv(PROCESSED_DATA_DIR / 'backtest_v4_low.csv', index=False)
    print(f"  Results saved to data/processed/backtest_v4_*.csv")
    print(f"{'='*72}")

    return max_df, min_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()
    run_backtest_v4(years_back=args.years)
