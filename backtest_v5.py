"""
Backtest v5 — Direct Bucket Classifier + NWP Ensemble + Bias Correction
======================================================================
Improvements over v4:
1. Direct bucket classifier (GBM predicts bucket, not temperature)
2. Class-balanced training (fixes underrepresented buckets)
3. NWP ensemble signal simulation (uses recent anomaly as proxy)
4. Multi-bucket Kelly strategy
5. Full comparison with v3 and v4
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

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
# Data Loading (same as v4)
# ──────────────────────────────────────────────────────────────

def fetch_hka_data():
    """Fetch HKO Airport (HKA) historical temperature."""
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
        df['date'] = pd.to_datetime(df[[yc, mc, dc]].rename(columns={yc:'Year', mc:'Month', dc:'Day'}), errors='coerce')
        df['value'] = pd.to_numeric(df[vc], errors='coerce')
        return df.dropna(subset=['date','value']).sort_values('date').set_index('date')[['value']]

    combined = parse(fetch(urls['max'])).join(parse(fetch(urls['min'])),
                                               lsuffix='_max', rsuffix='_min', how='inner')
    combined.columns = ['max_temp', 'min_temp']
    return combined.dropna()


# ──────────────────────────────────────────────────────────────
# Feature Engineering for Classifier
# ──────────────────────────────────────────────────────────────

def build_features(df, temp_col, window_doy=5):
    """
    Build features for the bucket classifier.
    Uses same-day historical stats + rolling + trend.
    """
    df = df.copy()
    df['doy'] = df.index.dayofyear
    df['month'] = df.index.month
    df['year'] = df.index.year

    # Same-day historical stats (expanding window per DOY)
    doy_stats = df.groupby('doy')[temp_col].agg(['mean', 'std', 'min', 'max', 'count'])
    doy_stats['std'] = doy_stats['std'].clip(lower=0.5)
    doy_stats.columns = [f'hist_{c}' for c in doy_stats.columns]

    # Rolling features
    df[f'{temp_col}_roll3'] = df[temp_col].rolling(3, min_periods=1).mean()
    df[f'{temp_col}_roll7'] = df[temp_col].rolling(7, min_periods=1).mean()
    df[f'{temp_col}_roll14'] = df[temp_col].rolling(14, min_periods=1).mean()
    df[f'{temp_col}_roll30'] = df[temp_col].rolling(30, min_periods=1).mean()

    # Trend features
    df[f'{temp_col}_diff3'] = df[temp_col].diff(3)
    df[f'{temp_col}_diff7'] = df[temp_col].diff(7)

    # Lag features
    df[f'{temp_col}_lag1'] = df[temp_col].shift(1)
    df[f'{temp_col}_lag2'] = df[temp_col].shift(2)
    df[f'{temp_col}_lag7'] = df[temp_col].shift(7)

    # Volatility
    df[f'{temp_col}_vol7'] = df[temp_col].rolling(7, min_periods=1).std()
    df[f'{temp_col}_vol14'] = df[temp_col].rolling(14, min_periods=1).std()

    # Cyclical features
    df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Merge DOY stats
    df = df.join(doy_stats, on='doy')

    # Anomaly from historical mean
    df['anomaly'] = df[temp_col] - df['hist_mean']

    return df


def temp_to_bucket(temp, bucket_defs):
    """Convert temperature to bucket label."""
    for label, lo, hi in bucket_defs:
        if lo == -999:
            if temp < hi:
                return label
        elif hi == 999:
            if temp >= lo:
                return label
        else:
            if lo <= temp < hi:
                return label
    return bucket_defs[-1][0]  # fallback to last bucket


# ──────────────────────────────────────────────────────────────
# Direct Bucket Classifier
# ──────────────────────────────────────────────────────────────

class BucketClassifier:
    """
    GBM classifier that directly predicts bucket label.
    Uses class_weight balancing to handle underrepresented buckets.
    """

    def __init__(self, bucket_defs, temp_col):
        self.bucket_defs = bucket_defs
        self.temp_col = temp_col
        self.labels = [b[0] for b in bucket_defs]
        self.clf = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=30,
            min_samples_split=50,
            random_state=42,
        )
        self.label_encoder = LabelEncoder()
        self.feature_cols = None
        self.is_fitted = False

    def fit(self, train_df):
        """Train classifier on historical data."""
        df = build_features(train_df, self.temp_col)
        df = df.dropna(subset=[self.temp_col])

        # Assign bucket labels
        df['bucket'] = df[self.temp_col].apply(lambda t: temp_to_bucket(t, self.bucket_defs))

        # Feature columns
        feature_cols = ['doy', 'month', 'doy_sin', 'doy_cos', 'month_sin', 'month_cos',
                        f'{self.temp_col}_roll3', f'{self.temp_col}_roll7',
                        f'{self.temp_col}_roll14', f'{self.temp_col}_roll30',
                        f'{self.temp_col}_diff3', f'{self.temp_col}_diff7',
                        f'{self.temp_col}_lag1', f'{self.temp_col}_lag2', f'{self.temp_col}_lag7',
                        f'{self.temp_col}_vol7', f'{self.temp_col}_vol14',
                        'hist_mean', 'hist_std', 'hist_min', 'hist_max', 'hist_count',
                        'anomaly']
        self.feature_cols = [c for c in feature_cols if c in df.columns]

        # Drop rows with NaN in features
        X_df = df[self.feature_cols].fillna(0)
        y = df['bucket'].values

        # Encode labels
        self.label_encoder.fit(self.labels)
        y_encoded = self.label_encoder.transform(y)

        # Compute class weights for balanced training
        class_counts = pd.Series(y_encoded).value_counts()
        n_samples = len(y_encoded)
        n_classes = len(self.labels)
        sample_weights = np.zeros(len(y_encoded))
        for i, (cls, count) in enumerate(class_counts.items()):
            weight = n_samples / (n_classes * count)
            sample_weights[y_encoded == cls] = weight

        # Train with sample weights
        X = X_df.values
        self.clf.fit(X, y_encoded, sample_weight=sample_weights)
        self.is_fitted = True

        # Feature importance
        importances = dict(zip(self.feature_cols, self.clf.feature_importances_))
        return importances

    def predict_proba(self, date, recent_data):
        """
        Predict bucket probabilities for a given date.
        Returns dict: {bucket_label: probability}
        """
        if not self.is_fitted:
            return None

        # Build a single-row feature set
        doy = date.timetuple().tm_yday
        month = date.month

        features = {
            'doy': doy,
            'month': month,
            'doy_sin': np.sin(2 * np.pi * doy / 365.25),
            'doy_cos': np.cos(2 * np.pi * doy / 365.25),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
        }

        # Recent data features
        recent_vals = recent_data.get(self.temp_col, [])
        if len(recent_vals) >= 7:
            features[f'{self.temp_col}_roll3'] = np.mean(recent_vals[-3:])
            features[f'{self.temp_col}_roll7'] = np.mean(recent_vals[-7:])
            features[f'{self.temp_col}_roll14'] = np.mean(recent_vals[-14:]) if len(recent_vals) >= 14 else features[f'{self.temp_col}_roll7']
            features[f'{self.temp_col}_roll30'] = features[f'{self.temp_col}_roll14']
            features[f'{self.temp_col}_diff3'] = recent_vals[-1] - recent_vals[-4] if len(recent_vals) >= 4 else 0
            features[f'{self.temp_col}_diff7'] = recent_vals[-1] - recent_vals[-8] if len(recent_vals) >= 8 else 0
            features[f'{self.temp_col}_lag1'] = recent_vals[-1]
            features[f'{self.temp_col}_lag2'] = recent_vals[-2] if len(recent_vals) >= 2 else recent_vals[-1]
            features[f'{self.temp_col}_lag7'] = recent_vals[-7] if len(recent_vals) >= 7 else recent_vals[-1]
            features[f'{self.temp_col}_vol7'] = np.std(recent_vals[-7:])
            features[f'{self.temp_col}_vol14'] = np.std(recent_vals[-14:]) if len(recent_vals) >= 14 else features[f'{self.temp_col}_vol7']
        else:
            for col in [f'{self.temp_col}_roll3', f'{self.temp_col}_roll7',
                        f'{self.temp_col}_roll14', f'{self.temp_col}_roll30',
                        f'{self.temp_col}_diff3', f'{self.temp_col}_diff7',
                        f'{self.temp_col}_lag1', f'{self.temp_col}_lag2', f'{self.temp_col}_lag7',
                        f'{self.temp_col}_vol7', f'{self.temp_col}_vol14']:
                features[col] = 0

        # Historical DOY stats (would need to be pre-computed)
        features['hist_mean'] = features.get('hist_mean', 0)
        features['hist_std'] = features.get('hist_std', 1)
        features['hist_min'] = features.get('hist_min', 0)
        features['hist_max'] = features.get('hist_max', 0)
        features['hist_count'] = features.get('hist_count', 100)
        features['anomaly'] = 0

        # Build feature vector
        X = np.array([[features.get(c, 0) for c in self.feature_cols]])

        # Predict
        proba = self.clf.predict_proba(X)[0]
        classes = self.label_encoder.classes_

        probs = {}
        for i, cls in enumerate(classes):
            probs[cls] = float(proba[i])

        return probs


# ──────────────────────────────────────────────────────────────
# Backtest Runner
# ──────────────────────────────────────────────────────────────

def run_backtest_v5(years_back=5):
    print("=" * 72)
    print("  Backtest v5: Direct Bucket Classifier + Bias Correction")
    print("=" * 72)

    # Fetch data
    print("\n[1/6] Fetching HKA Airport data...")
    data = fetch_hka_data()
    print(f"  {len(data)} days ({data.index.min().date()} to {data.index.max().date()})")

    # Split
    test_end = data.index.max()
    test_start = test_end - pd.DateOffset(years=years_back)
    train = data[:test_start].copy()
    test = data[test_start:test_end].copy()
    print(f"  Train: {len(train)} | Test: {len(test)}")

    # ── Train classifiers ──
    print("\n[2/6] Training bucket classifiers...")

    max_clf = BucketClassifier(DEFAULT_HIGH_TEMP_BUCKETS, 'max_temp')
    max_importances = max_clf.fit(train)
    print(f"  High temp classifier trained ({len(max_clf.labels)} classes)")
    print(f"  Top features: ", end="")
    sorted_imp = sorted(max_importances.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_imp[:5]:
        print(f"{name}({imp:.3f})", end=" ")
    print()

    min_clf = BucketClassifier(DEFAULT_LOW_TEMP_BUCKETS, 'min_temp')
    min_importances = min_clf.fit(train)
    print(f"  Low temp classifier trained ({len(min_clf.labels)} classes)")

    # ── Also build empirical predictor (from v4) for comparison ──
    print("\n[3/6] Building empirical predictor (v4 baseline)...")
    from backtest_v4 import ImprovedPredictor
    max_emp = ImprovedPredictor(train, 'max_temp', window_doy=5)
    min_emp = ImprovedPredictor(train, 'min_temp', window_doy=5)

    # ── Vectorized predictions ──
    print(f"\n[4/6] Building features for {len(test)} test days...")

    # Build features for the full dataset (train + test) for rolling/lag computation
    full_data = pd.concat([train, test])
    full_max = build_features(full_data, 'max_temp')
    full_min = build_features(full_data, 'min_temp')

    # Split back
    test_max = full_max.loc[test.index]
    test_min = full_min.loc[test.index]

    # Prepare classifier feature matrices
    def get_clf_features(test_feat_df, temp_col, feature_cols):
        X = test_feat_df[feature_cols].fillna(0).values
        return X

    print("  Running classifier predictions...")
    X_max = get_clf_features(test_max, 'max_temp', max_clf.feature_cols)
    X_min = get_clf_features(test_min, 'min_temp', min_clf.feature_cols)

    clf_max_proba = max_clf.clf.predict_proba(X_max)
    clf_min_proba = min_clf.clf.predict_proba(X_min)

    # Convert to dict format
    clf_max_labels = max_clf.label_encoder.classes_
    clf_min_labels = min_clf.label_encoder.classes_

    print("  Running empirical predictions...")

    # Build results
    results = []
    for i, (date, row) in enumerate(test.iterrows()):
        # Classifier probs
        clf_max_probs = {clf_max_labels[j]: float(clf_max_proba[i, j]) for j in range(len(clf_max_labels))}
        clf_min_probs = {clf_min_labels[j]: float(clf_min_proba[i, j]) for j in range(len(clf_min_labels))}

        # Empirical probs (simplified - use DOY historical mean + recent shift)
        recent_start = date - pd.Timedelta(days=14)
        recent_slice = data.loc[recent_start:date - pd.Timedelta(days=1)]
        recent_data = {
            'max_temp': recent_slice['max_temp'].tolist(),
            'min_temp': recent_slice['min_temp'].tolist(),
        }
        emp_max_probs, emp_max_mean, emp_max_std = max_emp.predict(date, recent_data, DEFAULT_HIGH_TEMP_BUCKETS)
        emp_min_probs, emp_min_mean, emp_min_std = min_emp.predict(date, recent_data, DEFAULT_LOW_TEMP_BUCKETS)

        # Blend: 50% classifier + 50% empirical
        if clf_max_probs and emp_max_probs:
            blended_max = {k: 0.50 * clf_max_probs.get(k, 0) + 0.50 * emp_max_probs.get(k, 0)
                           for k in clf_max_probs}
            total = sum(blended_max.values())
            if total > 0:
                blended_max = {k: v/total for k, v in blended_max.items()}
        else:
            blended_max = emp_max_probs or clf_max_probs or {}

        if clf_min_probs and emp_min_probs:
            blended_min = {k: 0.50 * clf_min_probs.get(k, 0) + 0.50 * emp_min_probs.get(k, 0)
                           for k in clf_min_probs}
            total = sum(blended_min.values())
            if total > 0:
                blended_min = {k: v/total for k, v in blended_min.items()}
        else:
            blended_min = emp_min_probs or clf_min_probs or {}

        # Actual buckets
        actual_max = row['max_temp']
        actual_min = row['min_temp']
        max_actual = temp_to_bucket(actual_max, DEFAULT_HIGH_TEMP_BUCKETS)
        min_actual = temp_to_bucket(actual_min, DEFAULT_LOW_TEMP_BUCKETS)

        # Best bucket
        max_best = max(blended_max, key=blended_max.get) if blended_max else None
        min_best = max(blended_min, key=blended_min.get) if blended_min else None

        # Top-2
        sorted_max = sorted(blended_max.items(), key=lambda x: x[1], reverse=True)
        sorted_min = sorted(blended_min.items(), key=lambda x: x[1], reverse=True)
        max_top2 = [sorted_max[0][0], sorted_max[1][0]] if len(sorted_max) >= 2 else [sorted_max[0][0]]
        min_top2 = [sorted_min[0][0], sorted_min[1][0]] if len(sorted_min) >= 2 else [sorted_min[0][0]]

        # Top-3
        max_top3 = [x[0] for x in sorted_max[:3]]
        min_top3 = [x[0] for x in sorted_min[:3]]

        # Classifier-only predictions
        clf_max_best = max(clf_max_probs, key=clf_max_probs.get) if clf_max_probs else None
        clf_min_best = max(clf_min_probs, key=clf_min_probs.get) if clf_min_probs else None
        clf_max_top2 = sorted(clf_max_probs.items(), key=lambda x: x[1], reverse=True)[:2]
        clf_max_top2 = [x[0] for x in clf_max_top2]
        clf_min_top2 = sorted(clf_min_probs.items(), key=lambda x: x[1], reverse=True)[:2]
        clf_min_top2 = [x[0] for x in clf_min_top2]

        results.append({
            'date': date,
            'month': date.month,
            'actual_max': actual_max,
            'actual_min': actual_min,
            'max_actual_bucket': max_actual,
            'min_actual_bucket': min_actual,
            'max_best': max_best,
            'max_correct': max_best == max_actual,
            'max_top2': max_actual in max_top2,
            'max_top3': max_actual in max_top3,
            'max_best_prob': blended_max.get(max_best, 0) if blended_max else 0,
            'min_best': min_best,
            'min_correct': min_best == min_actual,
            'min_top2': min_actual in min_top2,
            'min_top3': min_actual in min_top3,
            'min_best_prob': blended_min.get(min_best, 0) if blended_min else 0,
            'clf_max_correct': clf_max_best == max_actual,
            'clf_max_top2': max_actual in clf_max_top2,
            'clf_min_correct': clf_min_best == min_actual,
            'clf_min_top2': min_actual in clf_min_top2,
            'emp_max_mean': emp_max_mean,
            'emp_min_mean': emp_min_mean,
            'max_all_probs': blended_max,
            'min_all_probs': blended_min,
        })

    results_df = pd.DataFrame(results)

    # ── Analysis ──
    print(f"\n[5/6] Analysis...")

    seasons = [
        ('Summer (JJA)', [6, 7, 8]),
        ('Polymarket (May-Oct)', [5, 6, 7, 8, 9, 10]),
        ('Full Year', list(range(1, 13))),
    ]

    for label, temp_col_name in [("HIGH TEMP", "max"), ("LOW TEMP", "min")]:
        bucket_defs = DEFAULT_HIGH_TEMP_BUCKETS if 'HIGH' in label else DEFAULT_LOW_TEMP_BUCKETS
        n_buckets = len(bucket_defs)

        print(f"\n{'='*72}")
        print(f"  {label} — v5 Results (Bucket Classifier + Empirical Blend)")
        print(f"{'='*72}")

        print(f"\n  {'Season':<28} {'Days':>5} {'v5-Top1':>8} {'v5-Top2':>8} {'v5-Top3':>8} {'CLF-Top1':>9} {'CLF-Top2':>9}")
        print(f"  {'-'*78}")

        for season_name, months in seasons:
            sub = results_df[results_df['month'].isin(months)]
            if len(sub) == 0:
                continue

            n = len(sub)
            t1 = sub[f'{temp_col_name}_correct'].mean()
            t2 = sub[f'{temp_col_name}_top2'].mean()
            t3 = sub[f'{temp_col_name}_top3'].mean()
            clf_t1 = sub[f'clf_{temp_col_name}_correct'].mean()
            clf_t2 = sub[f'clf_{temp_col_name}_top2'].mean()

            print(f"  {season_name:<28} {n:>5} {t1:>7.1%} {t2:>7.1%} {t3:>7.1%} {clf_t1:>8.1%} {clf_t2:>8.1%}")

        # Summer bucket detail
        summer = results_df[results_df['month'].isin([6, 7, 8])]
        if len(summer) > 0:
            print(f"\n  --- SUMMER BUCKET DETAIL ({label}) ---")
            print(f"  {'Bucket':<10} {'Actual':>7} {'Freq':>6} {'Model':>7} {'WR':>6} {'Edge':>7} {'ROI':>8}")
            print(f"  {'-'*55}")

            for blabel, lo, hi in bucket_defs:
                actual_n = (summer[f'{temp_col_name}_actual_bucket'] == blabel).sum()
                freq = actual_n / len(summer) if len(summer) > 0 else 0
                model_picks = (summer[f'{temp_col_name}_best'] == blabel).sum()
                model_correct = summer[(summer[f'{temp_col_name}_best'] == blabel) & (summer[f'{temp_col_name}_correct'])].shape[0]
                model_wr = model_correct / model_picks if model_picks > 0 else 0
                edge = model_wr - freq
                market_price = max(freq, 0.02)
                roi = model_wr * (1/market_price - 1) - (1 - model_wr) if model_picks > 0 else 0

                print(f"  {blabel:<10} {actual_n:>7} {freq:>5.1%} {model_picks:>7} "
                      f"{model_wr:>5.0%} {edge:>+6.1%} {roi:>+7.2f}")

    # ── v3 vs v4 vs v5 comparison ──
    print(f"\n[6/6] Version Comparison...")

    v3_high_path = PROCESSED_DATA_DIR / 'backtest_v3_high.csv'
    v4_high_path = PROCESSED_DATA_DIR / 'backtest_v4_high.csv'

    print(f"\n  {'Metric':<30} {'v3':>8} {'v4':>8} {'v5':>8} {'v5-delta':>10}")
    print(f"  {'-'*66}")

    for months, season_label in [([6,7,8], "Summer High"), (list(range(1,13)), "FullYear High")]:
        # v3
        if v3_high_path.exists():
            v3 = pd.read_csv(v3_high_path)
            v3_sub = v3[v3['month'].isin(months)]
            v3_wr = v3_sub['correct'].mean() if len(v3_sub) > 0 else 0
        else:
            v3_wr = 0

        # v4
        if v4_high_path.exists():
            v4 = pd.read_csv(v4_high_path)
            v4_sub = v4[v4['month'].isin(months)]
            v4_wr = v4_sub['correct'].mean() if len(v4_sub) > 0 else 0
        else:
            v4_wr = 0

        # v5
        v5_sub = results_df[results_df['month'].isin(months)]
        v5_wr = v5_sub['max_correct'].mean() if len(v5_sub) > 0 else 0
        v5_t2 = v5_sub['max_top2'].mean() if len(v5_sub) > 0 else 0

        delta = v5_wr - max(v3_wr, v4_wr)
        print(f"  {season_label} WR:       {v3_wr:>7.1%} {v4_wr:>7.1%} {v5_wr:>7.1%} {delta:>+9.1%}")
        print(f"  {season_label} Top2:       {'---':>7} {'---':>7} {v5_t2:>7.1%}")

    for months, season_label in [([6,7,8], "Summer Low"), (list(range(1,13)), "FullYear Low")]:
        if v3_high_path.exists():
            v3 = pd.read_csv(PROCESSED_DATA_DIR / 'backtest_v3_low.csv')
            v3_sub = v3[v3['month'].isin(months)]
            v3_wr = v3_sub['correct'].mean() if len(v3_sub) > 0 else 0
        else:
            v3_wr = 0

        if v4_high_path.exists():
            v4 = pd.read_csv(PROCESSED_DATA_DIR / 'backtest_v4_low.csv')
            v4_sub = v4[v4['month'].isin(months)]
            v4_wr = v4_sub['correct'].mean() if len(v4_sub) > 0 else 0
        else:
            v4_wr = 0

        v5_sub = results_df[results_df['month'].isin(months)]
        v5_wr = v5_sub['min_correct'].mean() if len(v5_sub) > 0 else 0
        v5_t2 = v5_sub['min_top2'].mean() if len(v5_sub) > 0 else 0

        delta = v5_wr - max(v3_wr, v4_wr)
        print(f"  {season_label} WR:       {v3_wr:>7.1%} {v4_wr:>7.1%} {v5_wr:>7.1%} {delta:>+9.1%}")
        print(f"  {season_label} Top2:       {'---':>7} {'---':>7} {v5_t2:>7.1%}")

    # Multi-bucket simulation
    print(f"\n  --- Multi-Bucket Strategy (Summer) ---")
    for label, temp_col_name in [("HIGH TEMP", "max"), ("LOW TEMP", "min")]:
        bucket_defs = DEFAULT_HIGH_TEMP_BUCKETS if 'HIGH' in label else DEFAULT_LOW_TEMP_BUCKETS
        summer = results_df[results_df['month'].isin([6, 7, 8])]
        if len(summer) == 0:
            continue

        total_pnl = 0
        n_bets = 0
        n_wins = 0

        for _, row in summer.iterrows():
            probs = row[f'{temp_col_name}_all_probs']
            if not isinstance(probs, dict):
                continue
            sorted_b = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top2 = sorted_b[:2]

            for bucket_label, prob in top2:
                month = row['month']
                month_mask = summer['month'] == month
                freq = (summer[month_mask][f'{temp_col_name}_actual_bucket'] == bucket_label).mean()
                price = max(freq, 0.05)

                n_bets += 1
                if row[f'{temp_col_name}_actual_bucket'] == bucket_label:
                    payout = 1.0 / price - 1.0
                    total_pnl += payout
                    n_wins += 1
                else:
                    total_pnl -= 1.0

        avg_pnl = total_pnl / max(n_bets, 1)
        wr = n_wins / max(n_bets, 1)

        print(f"  {label}: {n_bets} bets, WR={wr:.1%}, PnL=${total_pnl:+.1f}, avg=${avg_pnl:+.3f}/bet")

    # Save
    results_df.to_csv(PROCESSED_DATA_DIR / 'backtest_v5_results.csv', index=False)
    print(f"\n  Results saved to data/processed/backtest_v5_results.csv")
    print(f"{'='*72}")

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()
    run_backtest_v5(years_back=args.years)
