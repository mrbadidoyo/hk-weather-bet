"""
Test script - Train ML models with synthetic data and validate full pipeline
"""
import logging
import numpy as np
import pandas as pd

from features import build_feature_matrix
from model import HKWeatherEnsemble
from polymarket_strategy import (
    analyze_market, format_analysis, parse_buckets,
    compute_bucket_probabilities, DEFAULT_HIGH_TEMP_BUCKETS, DEFAULT_LOW_TEMP_BUCKETS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_hk_weather():
    """Generate realistic synthetic HK weather data (5 years)"""
    dates = pd.date_range("2021-01-01", "2025-12-31")
    n = len(dates)
    doy = dates.dayofyear
    seasonal = 5.0 * np.sin(2 * np.pi * (doy - 80) / 365.25)

    # Correlated noise between HKO and VHHH
    noise_max = np.random.normal(0, 1.5, n)
    noise_min = np.random.normal(0, 1.2, n)

    # HKO HQ is in urban Kowloon, VHHH is at the airport (slightly different)
    hko_max = 28 + seasonal + noise_max
    hko_min = 23 + seasonal * 0.7 + noise_min
    wu_max = hko_max + np.random.normal(0.3, 0.6, n)  # VHHH slightly warmer
    wu_min = hko_min + np.random.normal(0.2, 0.5, n)

    df = pd.DataFrame({
        "hko_max_temp": hko_max,
        "hko_min_temp": hko_min,
        "hko_mean_temp": (hko_max + hko_min) / 2 + np.random.normal(0, 0.3, n),
        "hko_rh": 78 + 8 * np.sin(2 * np.pi * (doy - 120) / 365.25) + np.random.normal(0, 8, n),
        "hko_wind_speed": 12 + 3 * np.sin(2 * np.pi * (doy - 30) / 365.25) + np.random.normal(0, 3, n),
        "hko_pressure": 1013 - 5 * np.cos(2 * np.pi * doy / 365.25) + np.random.normal(0, 3, n),
        "wu_max_temp_c": wu_max,
        "wu_min_temp_c": wu_min,
        "wu_avg_temp_c": (wu_max + wu_min) / 2,
    }, index=dates)

    # Clip to realistic HK ranges
    df["hko_rh"] = df["hko_rh"].clip(30, 100)
    df["hko_wind_speed"] = df["hko_wind_speed"].clip(0, 50)

    return df


def main():
    print("=" * 70)
    print("  HK Weather Prediction - Full Pipeline Test")
    print("=" * 70)

    # Step 1: Generate synthetic data
    print("\n[1/4] Generating synthetic HK weather data...")
    raw_df = generate_synthetic_hk_weather()
    print(f"  Generated {len(raw_df)} days of data ({raw_df.index.min().date()} to {raw_df.index.max().date()})")

    # Step 2: Build features
    print("\n[2/4] Building feature matrix...")
    features = build_feature_matrix(raw_df)
    print(f"  Feature matrix: {features.shape}")

    # Step 3: Train models
    print("\n[3/4] Training ensemble models...")
    ensemble = HKWeatherEnsemble()
    metrics = ensemble.fit(features, "wu_max_temp_c", "wu_min_temp_c")

    print(f"\n  Results:")
    print(f"  High Temp - MAE: {metrics['max_temp']['mae']:.2f}C, RMSE: {metrics['max_temp']['rmse']:.2f}C, R2: {metrics['max_temp']['r2']:.4f}")
    print(f"  Low Temp  - MAE: {metrics['min_temp']['mae']:.2f}C, RMSE: {metrics['min_temp']['rmse']:.2f}C, R2: {metrics['min_temp']['r2']:.4f}")

    # Step 4: Simulate Polymarket betting analysis
    print("\n[4/4] Simulating Polymarket analysis...")

    # Simulated market prices (what Polymarket might show)
    high_prices = {"<30": 0.05, "30-31": 0.10, "31-32": 0.18, "32-33": 0.25, "33-34": 0.22, "34-35": 0.12, "35+": 0.08}
    low_prices = {"<25": 0.06, "25-26": 0.12, "26-27": 0.22, "27-28": 0.28, "28-29": 0.18, "29-30": 0.10, "30+": 0.04}

    # Get prediction for a recent date
    recent = features.tail(1)
    preds = ensemble.predict(recent)

    max_dist = preds["max_temp"]
    min_dist = preds["min_temp"]

    print(f"\n  Predicted High: {max_dist['mean'].iloc[0]:.1f}C (std: {max_dist['std'].iloc[0]:.1f})")
    print(f"  Predicted Low:  {min_dist['mean'].iloc[0]:.1f}C (std: {min_dist['std'].iloc[0]:.1f})")

    # High temp analysis
    high_analysis = analyze_market(
        date="2025-12-31",
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
    print(format_analysis(high_analysis))

    # Low temp analysis
    low_analysis = analyze_market(
        date="2025-12-31",
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
    print(format_analysis(low_analysis))

    # Save models
    ensemble.save()
    print("\n  Models saved to models/ directory")

    print("\n" + "=" * 70)
    print("  Full pipeline test PASSED!")
    print("=" * 70)


def test_pipeline():
    """Alias for main() — for programmatic import."""
    main()

if __name__ == "__main__":
    main()
