"""
Data Collector - Fetches historical weather data from HKO and Weather Underground
"""
import io
import re
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

from config import (
    HKO_API_BASE, DATAGOVHK_DATASETS, RAW_DATA_DIR,
    WUNDERGROUND_URL_PATTERN, RESOLUTION_STATION,
)

logger = logging.getLogger(__name__)


class HKODataCollector:
    """Collects data from Hong Kong Observatory APIs and DATA.GOV.HK"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def fetch_hko_api(self, data_type: str, lang: str = "en") -> dict:
        """Fetch data from HKO Open Data API"""
        params = {"dataType": data_type, "lang": lang}
        resp = self.session.get(HKO_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_current_weather(self) -> dict:
        """Fetch current weather report (hourly readings)"""
        return self.fetch_hko_api("rhrread")

    def fetch_9day_forecast(self) -> dict:
        """Fetch 9-day weather forecast"""
        return self.fetch_hko_api("fnd")

    def fetch_local_forecast(self) -> dict:
        """Fetch local weather forecast (today/tomorrow)"""
        return self.fetch_hko_api("flw")

    def fetch_historical_csv(self, dataset_key: str) -> pd.DataFrame:
        """Fetch a historical dataset CSV from DATA.GOV.HK"""
        if dataset_key not in DATAGOVHK_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATAGOVHK_DATASETS.keys())}")
        url = DATAGOVHK_DATASETS[dataset_key]["url"]
        logger.info(f"Fetching {dataset_key} from {url}")
        resp = self.session.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        return df

    def fetch_all_historical(self) -> dict[str, pd.DataFrame]:
        """Fetch all available historical datasets"""
        datasets = {}
        for key in DATAGOVHK_DATASETS:
            try:
                datasets[key] = self.fetch_historical_csv(key)
                logger.info(f"  Loaded {key}: {len(datasets[key])} rows")
                time.sleep(0.5)  # rate limiting
            except Exception as e:
                logger.warning(f"  Failed to fetch {key}: {e}")
        return datasets

    def save_historical_data(self, datasets: dict[str, pd.DataFrame]):
        """Save datasets to local CSV files"""
        for key, df in datasets.items():
            path = RAW_DATA_DIR / f"hko_{key}.csv"
            df.to_csv(path, index=False)
            logger.info(f"  Saved {key} -> {path} ({len(df)} rows)")


class WundergroundCollector:
    """
    Collects VHHH (HK Airport) daily temperature data from Weather Underground.
    This is the RESOLUTION SOURCE for Polymarket HK weather markets.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })

    def fetch_daily_history_page(self, date: datetime) -> dict | None:
        """
        Fetch the WU daily history page for VHHH on a given date.
        Returns parsed temperature data (max, min, avg).
        """
        date_str = date.strftime("%Y-%m-%d")
        url = WUNDERGROUND_URL_PATTERN.format(date=date_str)
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return self._parse_history_page(resp.text, date)
        except Exception as e:
            logger.warning(f"Failed to fetch WU data for {date_str}: {e}")
            return None

    def _parse_history_page(self, html: str, date: datetime) -> dict:
        """Parse WU history page for daily temperature summary"""
        soup = BeautifulSoup(html, "lxml")
        result = {"date": date.strftime("%Y-%m-%d"), "max_temp_c": None, "min_temp_c": None, "avg_temp_c": None}

        # WU history pages have a summary table with Temperature row
        # Try to find the summary table
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 4:
                    header = cells[0].get_text(strip=True).lower()
                    if "temperature" in header:
                        try:
                            values = [c.get_text(strip=True) for c in cells[1:]]
                            # Values are typically Max, Avg, Min in °F
                            nums = [float(re.sub(r'[^\d.-]', '', v)) for v in values if re.search(r'[\d.-]', v)]
                            if len(nums) >= 3:
                                # Convert F to C
                                result["max_temp_c"] = round((nums[0] - 32) * 5 / 9, 1)
                                result["avg_temp_c"] = round((nums[1] - 32) * 5 / 9, 1)
                                result["min_temp_c"] = round((nums[2] - 32) * 5 / 9, 1)
                            elif len(nums) >= 2:
                                result["max_temp_c"] = round((nums[0] - 32) * 5 / 9, 1)
                                result["min_temp_c"] = round((nums[1] - 32) * 5 / 9, 1)
                        except (ValueError, IndexError):
                            pass

        # Also try the JSON-LD or embedded script data
        scripts = soup.find_all("script", type="application/json")
        for script in scripts:
            try:
                import json
                data = json.loads(script.string)
                # Navigate the JSON to find temperature data
                if isinstance(data, dict):
                    self._extract_from_json(data, result)
            except (json.JSONDecodeError, TypeError):
                continue

        return result

    def _extract_from_json(self, data: dict, result: dict):
        """Recursively extract temperature data from WU JSON"""
        for key, value in data.items():
            if isinstance(value, dict):
                if "max" in value and "min" in value and "temperature" in str(key).lower():
                    try:
                        if result["max_temp_c"] is None:
                            result["max_temp_c"] = round(float(value["max"]), 1)
                            result["min_temp_c"] = round(float(value["min"]), 1)
                            if "avg" in value:
                                result["avg_temp_c"] = round(float(value["avg"]), 1)
                    except (ValueError, TypeError):
                        pass
                self._extract_from_json(value, result)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._extract_from_json(item, result)

    def fetch_date_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch VHHH daily temperatures for a date range"""
        records = []
        current = start
        total_days = (end - start).days
        logger.info(f"Fetching WU VHHH data from {start.date()} to {end.date()} ({total_days} days)")

        while current <= end:
            data = self.fetch_daily_history_page(current)
            if data and data.get("max_temp_c") is not None:
                records.append(data)
            current += timedelta(days=1)
            # Rate limiting - be respectful
            time.sleep(1.5)
            if len(records) % 30 == 0:
                logger.info(f"  Fetched {len(records)}/{total_days} days...")

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def save_data(self, df: pd.DataFrame, filename: str = "wunderground_vhhh.csv"):
        """Save WU data to local CSV"""
        path = RAW_DATA_DIR / filename
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["date"])
            df = pd.concat([existing, df]).drop_duplicates(subset=["date"]).sort_values("date")
        df.to_csv(path, index=False)
        logger.info(f"Saved WU data -> {path} ({len(df)} rows)")
        return path


def build_combined_dataset(
    hko_datasets: dict[str, pd.DataFrame],
    wu_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine HKO historical data with Wunderground VHHH data into a single
    analysis-ready DataFrame.
    """
    # Parse HKO temperature datasets
    # These typically have columns: Year, Month, Day, Value (or similar)
    combined = {}

    for key, df in hko_datasets.items():
        cols = df.columns.tolist()
        logger.info(f"  {key} columns: {cols}")

        # Standardize date column
        if "Year" in cols and "Month" in cols and "Day" in cols:
            df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
        elif "Date" in cols:
            df["date"] = pd.to_datetime(df["Date"])

        if "date" not in df.columns:
            continue

        # Extract the value column
        value_col = None
        for col in cols:
            if col not in ["Year", "Month", "Day", "Date", "date"]:
                value_col = col
                break
        if value_col is None:
            continue

        df = df[["date", value_col]].copy()
        df = df.rename(columns={value_col: f"hko_{key}"})
        df = df.set_index("date")
        combined[key] = df

    if not combined:
        logger.error("No HKO datasets could be parsed")
        return pd.DataFrame()

    # Merge all HKO datasets
    result = combined[list(combined.keys())[0]]
    for key in list(combined.keys())[1:]:
        result = result.join(combined[key], how="outer")

    # Add WU data
    if wu_df is not None and not wu_df.empty:
        wu_indexed = wu_df.set_index("date")
        for col in wu_indexed.columns:
            result[f"wu_{col}"] = wu_indexed[col]

    result = result.sort_index()
    return result


def collect_all_data():
    """Main data collection pipeline"""
    logger.info("=" * 60)
    logger.info("Starting HK Weather Data Collection")
    logger.info("=" * 60)

    # 1. Collect HKO historical data
    logger.info("\n[1/3] Fetching HKO historical data...")
    hko = HKODataCollector()
    hko_datasets = hko.fetch_all_historical()
    hko.save_historical_data(hko_datasets)

    # 2. Fetch current HKO forecast
    logger.info("\n[2/3] Fetching HKO current forecast...")
    try:
        forecast = hko.fetch_9day_forecast()
        forecast_path = RAW_DATA_DIR / "hko_9day_forecast.json"
        import json
        with open(forecast_path, "w") as f:
            json.dump(forecast, f, indent=2)
        logger.info(f"  Saved 9-day forecast -> {forecast_path}")
    except Exception as e:
        logger.warning(f"  Failed to fetch forecast: {e}")

    # 3. Collect Wunderground VHHH data
    logger.info("\n[3/3] Fetching Weather Underground VHHH data...")
    wu = WundergroundCollector()

    # Fetch last 2 years of VHHH data for training
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    # Check if we have existing data
    wu_path = RAW_DATA_DIR / "wunderground_vhhh.csv"
    if wu_path.exists():
        existing = pd.read_csv(wu_path, parse_dates=["date"])
        last_date = existing["date"].max()
        if last_date > start_date:
            start_date = last_date + timedelta(days=1)
            logger.info(f"  Resuming from {start_date.date()}")

    wu_df = wu.fetch_date_range(start_date, end_date)
    wu.save_data(wu_df)

    # 4. Build combined dataset
    logger.info("\n[4/4] Building combined dataset...")
    combined = build_combined_dataset(hko_datasets, wu_df)
    combined_path = RAW_DATA_DIR / "combined_dataset.csv"
    combined.to_csv(combined_path)
    logger.info(f"  Combined dataset: {len(combined)} rows, {len(combined.columns)} columns")

    logger.info("\n" + "=" * 60)
    logger.info("Data collection complete!")
    logger.info("=" * 60)

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    collect_all_data()
