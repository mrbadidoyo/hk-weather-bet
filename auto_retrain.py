"""
Auto-Retrain Pipeline — Weekly model retraining with performance comparison.
Retrains the bucket classifier with new data and deploys only if improved.
"""
import io
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from config import (
    DEFAULT_HIGH_TEMP_BUCKETS,
    DEFAULT_LOW_TEMP_BUCKETS,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
)
from data_collector import HKODataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRAIN_LOG = PROCESSED_DATA_DIR / "retrain_history.jsonl"


# ── Data Loading ──────────────────────────────────────────────────

def fetch_hka_data():
    """Fetch HKO Airport (HKA) historical temperature data."""
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
        df['date'] = pd.to_datetime(
            df[[yc, mc, dc]].rename(columns={yc: 'Year', mc: 'Month', dc: 'Day'}),
            errors='coerce'
        )
        df['value'] = pd.to_numeric(df[vc], errors='coerce')
        return df.dropna(subset=['date', 'value']).sort_values('date').set_index('date')[['value']]

    combined = parse(fetch(urls['max'])).join(
        parse(fetch(urls['min'])),
        lsuffix='_max', rsuffix='_min', how='inner'
    )
    combined.columns = ['max_temp', 'min_temp']
    return combined.dropna()


# ── Feature Engineering ───────────────────────────────────────────

def build_features(df, temp_col):
    """Build features for the bucket classifier."""
    df = df.copy()
    df['doy'] = df.index.dayofyear
    df['month'] = df.index.month

    # Same-day historical stats
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

    # Cyclical
    df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Merge DOY stats
    df = df.join(doy_stats, on='doy')
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
    return bucket_defs[-1][0]


# ── Classifier ────────────────────────────────────────────────────

class BucketClassifier:
    """GBM classifier that directly predicts bucket label."""

    def __init__(self, bucket_defs, temp_col):
        self.bucket_defs = bucket_defs
        self.temp_col = temp_col
        self.labels = [b[0] for b in bucket_defs]
        self.clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=30, min_samples_split=50,
            random_state=42,
        )
        self.label_encoder = LabelEncoder()
        self.feature_cols = None

    def fit(self, train_df):
        """Train classifier."""
        df = build_features(train_df, self.temp_col)
        df = df.dropna(subset=[self.temp_col])
        df['bucket'] = df[self.temp_col].apply(lambda t: temp_to_bucket(t, self.bucket_defs))

        feature_cols = ['doy', 'month', 'doy_sin', 'doy_cos', 'month_sin', 'month_cos',
                        f'{self.temp_col}_roll3', f'{self.temp_col}_roll7',
                        f'{self.temp_col}_roll14', f'{self.temp_col}_roll30',
                        f'{self.temp_col}_diff3', f'{self.temp_col}_diff7',
                        f'{self.temp_col}_lag1', f'{self.temp_col}_lag2', f'{self.temp_col}_lag7',
                        f'{self.temp_col}_vol7', f'{self.temp_col}_vol14',
                        'hist_mean', 'hist_std', 'hist_min', 'hist_max', 'hist_count',
                        'anomaly']
        self.feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[self.feature_cols].fillna(0).values
        y = df['bucket'].values

        self.label_encoder.fit(self.labels)
        y_enc = self.label_encoder.transform(y)

        self.clf.fit(X, y_enc)
        return self

    def predict(self, df):
        """Predict bucket probabilities."""
        df = build_features(df, self.temp_col)
        X = df[self.feature_cols].fillna(0).values
        proba = self.clf.predict_proba(X)

        # Map back to bucket labels
        label_map = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        result = {}
        for i in range(proba.shape[1]):
            result[label_map[i]] = proba[:, i]
        return result

    def score(self, df):
        """Calculate win rate on data."""
        df = build_features(df, self.temp_col)
        df = df.dropna(subset=[self.temp_col])
        df['bucket'] = df[self.temp_col].apply(lambda t: temp_to_bucket(t, self.bucket_defs))

        feature_cols = [c for c in self.feature_cols if c in df.columns]
        X = df[feature_cols].fillna(0).values
        y_true = df['bucket'].values

        self.label_encoder.fit(self.labels)
        y_true_enc = self.label_encoder.transform(y_true)
        y_pred_enc = self.clf.predict(X)

        return np.mean(y_true_enc == y_pred_enc)


# ── Retrain Pipeline ──────────────────────────────────────────────

def retrain_model(temp_type="high", train_years=5, deploy_if_better=True):
    """
    Retrain the bucket classifier with latest data.
    
    Args:
        temp_type: 'high' or 'low'
        train_years: How many years of data to use
        deploy_if_better: Only save model if it beats the old one
    
    Returns:
        dict with training results and comparison
    """
    bucket_defs = DEFAULT_HIGH_TEMP_BUCKETS if temp_type == "high" else DEFAULT_LOW_TEMP_BUCKETS
    col = "max_temp" if temp_type == "high" else "min_temp"
    model_path = MODELS_DIR / f"retrain_{temp_type}_classifier.json"

    logger.info(f"Fetching HKA data for {temp_type} temperature retrain...")
    data = fetch_hka_data()

    # Use last N years
    cutoff = data.index.max() - pd.DateOffset(years=train_years)
    train_data = data[data.index >= cutoff]
    logger.info(f"Training on {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")

    # Train-test split: last 90 days as validation
    val_cutoff = train_data.index.max() - pd.Timedelta(days=90)
    train_set = train_data[train_data.index <= val_cutoff]
    val_set = train_data[train_data.index > val_cutoff]

    logger.info(f"Train: {len(train_set)} days, Validation: {len(val_set)} days")

    # Train new model
    classifier = BucketClassifier(bucket_defs, col)
    classifier.fit(train_set)

    # Evaluate
    train_score = classifier.score(train_set)
    val_score = classifier.score(val_set)

    logger.info(f"New model — Train: {train_score:.1%}, Validation: {val_score:.1%}")

    # Compare with old model if exists
    old_score = None
    if model_path.exists():
        old_meta = json.loads(model_path.read_text(encoding="utf-8"))
        old_score = old_meta.get("val_score", 0)
        logger.info(f"Old model validation: {old_score:.1%}")

    # Deploy decision
    improved = old_score is None or val_score > old_score
    deployed = False

    if deploy_if_better and improved:
        # Save model metadata (actual sklearn model saved as joblib)
        import joblib
        model_file = MODELS_DIR / f"retrain_{temp_type}_classifier.joblib"
        joblib.dump(classifier, model_file)

        meta = {
            "timestamp": datetime.now().isoformat(),
            "temp_type": temp_type,
            "train_days": len(train_set),
            "val_days": len(val_set),
            "train_score": round(train_score, 4),
            "val_score": round(val_score, 4),
            "old_val_score": old_score,
            "train_years": train_years,
            "data_range": [str(train_data.index[0].date()), str(train_data.index[-1].date())],
            "deployed": True,
        }
        model_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        deployed = True
        logger.info(f"Model deployed! ({old_score or 0:.1%} -> {val_score:.1%})")
    else:
        logger.info(f"Model NOT deployed (old={old_score:.1%}, new={val_score:.1%})")

    # Log retrain history
    result = {
        "timestamp": datetime.now().isoformat(),
        "temp_type": temp_type,
        "train_score": round(train_score, 4),
        "val_score": round(val_score, 4),
        "old_val_score": old_score,
        "improved": improved,
        "deployed": deployed,
        "n_train": len(train_set),
        "n_val": len(val_set),
    }

    with open(RETRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    return result


def run_full_retrain(train_years=5, deploy_if_better=True):
    """Retrain both high and low temperature models."""
    results = {}

    logger.info("=" * 50)
    logger.info("  Auto-Retrain Pipeline")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    for temp_type in ["high", "low"]:
        logger.info(f"\n--- Retraining {temp_type.upper()} model ---")
        results[temp_type] = retrain_model(temp_type, train_years, deploy_if_better)

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("  Retrain Summary")
    logger.info("=" * 50)
    for tt, r in results.items():
        status = "DEPLOYED" if r["deployed"] else "KEPT OLD"
        old = f"{r['old_val_score']:.1%}" if r["old_val_score"] else "N/A"
        logger.info(f"  {tt.upper()}: {old} -> {r['val_score']:.1%} [{status}]")

    return results


def get_retrain_history():
    """Get retrain history."""
    if not RETRAIN_LOG.exists():
        return []

    entries = []
    for line in RETRAIN_LOG.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def get_model_status():
    """Get current model status for dashboard."""
    status = {}
    for temp_type in ["high", "low"]:
        model_path = MODELS_DIR / f"retrain_{temp_type}_classifier.json"
        if model_path.exists():
            meta = json.loads(model_path.read_text(encoding="utf-8"))
            status[temp_type] = {
                "exists": True,
                "val_score": meta.get("val_score"),
                "trained_at": meta.get("timestamp"),
                "train_days": meta.get("train_days"),
                "data_range": meta.get("data_range"),
            }
        else:
            status[temp_type] = {"exists": False}
    return status
