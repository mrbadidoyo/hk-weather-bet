"""
Feature Engineering - Transform raw weather data into predictive features
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from config import ROLLING_WINDOWS, LAG_DAYS

logger = logging.getLogger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features: day of year, month, season, etc."""
    df = df.copy()
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)

    df["day_of_year"] = idx.dayofyear
    df["month"] = idx.month
    df["day_of_week"] = idx.dayofweek
    df["quarter"] = idx.quarter
    df["year"] = idx.year
    df["week_of_year"] = idx.isocalendar().week.astype(int)

    # Cyclical encoding for day_of_year and month
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Season indicator (HK seasons)
    df["season"] = pd.cut(
        df["month"],
        bins=[0, 2, 5, 8, 11, 12],
        labels=["winter", "spring", "summer", "autumn", "winter"],
        ordered=False,
    )
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_typhoon_season"] = df["month"].isin([5, 6, 7, 8, 9, 10]).astype(int)

    # Is weekend
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


def add_rolling_features(df: pd.DataFrame, temp_cols: list[str]) -> pd.DataFrame:
    """Add rolling window statistics for temperature columns"""
    df = df.copy()
    new_cols = {}
    for col in temp_cols:
        if col not in df.columns:
            continue
        for window in ROLLING_WINDOWS:
            roller = df[col].rolling(window, min_periods=1)
            new_cols[f"{col}_roll_mean_{window}"] = roller.mean()
            new_cols[f"{col}_roll_std_{window}"] = roller.std()
            new_cols[f"{col}_roll_min_{window}"] = roller.min()
            new_cols[f"{col}_roll_max_{window}"] = roller.max()
            new_cols[f"{col}_roll_median_{window}"] = roller.median()

        new_cols[f"{col}_ewm_7"] = df[col].ewm(span=7, min_periods=1).mean()
        new_cols[f"{col}_ewm_14"] = df[col].ewm(span=14, min_periods=1).mean()
        new_cols[f"{col}_ewm_30"] = df[col].ewm(span=30, min_periods=1).mean()

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def add_lag_features(df: pd.DataFrame, temp_cols: list[str]) -> pd.DataFrame:
    """Add lag features for temperature columns"""
    df = df.copy()
    new_cols = {}
    for col in temp_cols:
        if col not in df.columns:
            continue
        for lag in LAG_DAYS:
            new_cols[f"{col}_lag_{lag}"] = df[col].shift(lag)
        new_cols[f"{col}_diff_1"] = df[col].diff(1)
        new_cols[f"{col}_diff_7"] = df[col].diff(7)
        new_cols[f"{col}_pct_change_1"] = df[col].pct_change(1)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def add_same_day_historical_features(df: pd.DataFrame, temp_cols: list[str]) -> pd.DataFrame:
    """
    Add features based on same-day historical statistics.
    For each day, compute the historical mean/std/min/max for that day-of-year.
    """
    df = df.copy()
    doy = df.index.dayofyear if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index).dayofyear

    for col in temp_cols:
        if col not in df.columns:
            continue
        # Group by day-of-year and compute historical statistics
        doy_stats = df.groupby(doy)[col].agg(["mean", "std", "min", "max", "median"])
        doy_stats.columns = [f"{col}_doy_mean", f"{col}_doy_std", f"{col}_doy_min", f"{col}_doy_max", f"{col}_doy_median"]
        doy_stats.index.name = "day_of_year_feat"

        # Merge back
        df_temp = df.copy()
        df_temp["day_of_year_feat"] = doy
        df_temp = df_temp.merge(doy_stats, on="day_of_year_feat", how="left")
        for stat_col in [f"{col}_doy_mean", f"{col}_doy_std", f"{col}_doy_min", f"{col}_doy_max", f"{col}_doy_median"]:
            df[stat_col] = df_temp[stat_col].values

        # Anomaly: how far current value is from historical mean
        df[f"{col}_anomaly"] = df[col] - df[f"{col}_doy_mean"]
        df[f"{col}_anomaly_z"] = df[f"{col}_anomaly"] / (df[f"{col}_doy_std"] + 0.01)

    return df


def add_hko_forecast_features(df: pd.DataFrame, forecast_data: dict | None = None) -> pd.DataFrame:
    """
    Parse HKO 9-day forecast and add forecast features.
    The 9-day forecast provides daily max/min temperature predictions.
    """
    df = df.copy()

    if forecast_data is None:
        return df

    try:
        # Parse 9-day forecast
        weather_forecast = forecast_data.get("weatherForecast", [])
        if not weather_forecast:
            return df

        for day_forecast in weather_forecast:
            forecast_date = day_forecast.get("forecastDate", "")
            max_temp = day_forecast.get("forecastMaxtemp", {}).get("value")
            min_temp = day_forecast.get("forecastMintemp", {}).get("value")
            weather = day_forecast.get("forecastWeather", "")

            if max_temp is not None:
                df.loc[df.index == forecast_date, "hko_forecast_max"] = float(max_temp)
            if min_temp is not None:
                df.loc[df.index == forecast_date, "hko_forecast_min"] = float(min_temp)

        # Forward fill forecast features for days within the forecast window
        if "hko_forecast_max" in df.columns:
            df["hko_forecast_max"] = df["hko_forecast_max"].ffill(limit=1)
            df["hko_forecast_min"] = df["hko_forecast_min"].ffill(limit=1)

    except Exception as e:
        logger.warning(f"Failed to parse HKO forecast: {e}")

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between temperature and other variables"""
    df = df.copy()

    # Temperature range (if we have both max and min from HKO)
    max_cols = [c for c in df.columns if "max" in c.lower() and "temp" in c.lower()]
    min_cols = [c for c in df.columns if "min" in c.lower() and "temp" in c.lower()]

    if max_cols and min_cols:
        max_col = max_cols[0]
        min_col = min_cols[0]
        df["temp_range"] = df[max_col] - df[min_col]
        df["temp_mean"] = (df[max_col] + df[min_col]) / 2

    # Humidity-temperature interaction
    rh_cols = [c for c in df.columns if "rh" in c.lower() or "humidity" in c.lower()]
    if rh_cols and max_cols:
        df["heat_index_approx"] = (
            -8.78469475556
            + 1.61139411 * df[max_cols[0]]
            + 2.33854883889 * df[rh_cols[0]]
            - 0.14611605 * df[max_cols[0]] * df[rh_cols[0]]
            - 0.012308094 * df[max_cols[0]] ** 2
            - 0.0164248277778 * df[rh_cols[0]] ** 2
            + 0.002211732 * df[max_cols[0]] ** 2 * df[rh_cols[0]]
            + 0.00072546 * df[max_cols[0]] * df[rh_cols[0]] ** 2
            - 0.000003582 * df[max_cols[0]] ** 2 * df[rh_cols[0]] ** 2
        )

    return df


def build_feature_matrix(raw_df: pd.DataFrame, target_col_max: str = None, target_col_min: str = None) -> pd.DataFrame:
    """
    Build the complete feature matrix from raw data.
    Returns a DataFrame with all features and target columns.
    """
    logger.info(f"Building feature matrix from {len(raw_df)} rows...")
    df = raw_df.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Identify temperature columns for rolling/lag features
    temp_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["temp", "max", "min", "mean"])]

    # 1. Temporal features
    logger.info("  Adding temporal features...")
    df = add_temporal_features(df)

    # 2. Rolling features
    logger.info("  Adding rolling features...")
    df = add_rolling_features(df, temp_cols)

    # 3. Lag features
    logger.info("  Adding lag features...")
    df = add_lag_features(df, temp_cols)

    # 4. Same-day historical features
    logger.info("  Adding same-day historical features...")
    df = add_same_day_historical_features(df, temp_cols)

    # 5. Interaction features
    logger.info("  Adding interaction features...")
    df = add_interaction_features(df)

    # Identify target columns
    if target_col_max is None:
        # Auto-detect WU max temp (Polymarket resolution source)
        candidates = [c for c in df.columns if "wu" in c.lower() and "max" in c.lower()]
        if candidates:
            target_col_max = candidates[0]
        else:
            candidates = [c for c in df.columns if "max" in c.lower() and "temp" in c.lower()]
            target_col_max = candidates[0] if candidates else None

    if target_col_min is None:
        candidates = [c for c in df.columns if "wu" in c.lower() and "min" in c.lower()]
        if candidates:
            target_col_min = candidates[0]
        else:
            candidates = [c for c in df.columns if "min" in c.lower() and "temp" in c.lower()]
            target_col_min = candidates[0] if candidates else None

    # Store target info
    df.attrs["target_col_max"] = target_col_max
    df.attrs["target_col_min"] = target_col_min

    logger.info(f"  Feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
    logger.info(f"  Target (max): {target_col_max}")
    logger.info(f"  Target (min): {target_col_min}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test with sample data
    dates = pd.date_range("2023-01-01", "2024-12-31")
    sample = pd.DataFrame({
        "hko_max_temp": np.random.normal(28, 5, len(dates)),
        "hko_min_temp": np.random.normal(23, 4, len(dates)),
        "hko_mean_temp": np.random.normal(25.5, 4.5, len(dates)),
        "hko_rh": np.random.normal(75, 10, len(dates)),
        "wu_max_temp_c": np.random.normal(28, 5, len(dates)),
        "wu_min_temp_c": np.random.normal(23, 4, len(dates)),
    }, index=dates)

    features = build_feature_matrix(sample)
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Columns: {features.columns.tolist()[:20]}...")
