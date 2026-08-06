"""
Configuration for HK Weather Prediction System
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# HKO API Configuration
HKO_API_BASE = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
HKO_DATA_TYPES = {
    "fnd": "9-day weather forecast",
    "rhrread": "Current weather report",
    "flw": "Local weather forecast",
    "CLMTEMP": "Daily mean temperature (historical)",
    "CLMMAXT": "Daily maximum temperature (historical)",
    "CLMMINT": "Daily minimum temperature (historical)",
}

# HKO Station Codes
HKO_STATIONS = {
    "HKO": "Hong Kong Observatory HQ",
    "HKA": "HK International Airport",
    "KP": "King's Park",
    "TKL": "Ta Kwu Ling",
    "SHA": "Sha Tin",
    "TMS": "Tai Mo Shan",
    "TC": "Tate's Cairn",
    "CCH": "Cheung Chau",
    "WGL": "Waglan Island",
}

# Polymarket Resolution Source
# HK markets resolve on HKO "Absolute Daily Max (deg. C)" from the Daily Extract
# URL: https://www.weather.gov.hk/en/cis/climat.htm
RESOLUTION_STATION = "HKO_Absolute_Daily_Max"
RESOLUTION_SOURCE_URL = "https://www.weather.gov.hk/en/cis/climat.htm"

# Wunderground historical data URL pattern
WUNDERGROUND_URL_PATTERN = (
    "https://www.wunderground.com/history/daily/hk/hong-kong/VHHH/date/{date}"
)

# HKO Historical CSV URLs (DATA.GOV.HK via HKO Open Data API)
# These are monthly updated CSV files with full historical records
# HKA = HK International Airport (matches Polymarket VHHH resolution source)
# HKO = HKO Headquarters (urban heat island, ~0.5-1C warmer than airport)
# URL pattern: opendata.php?dataType=<TYPE>&rformat=csv&station=<STATION>
HKO_HISTORICAL_CSVS = {
    "daily_max_temp_hka": "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKA",
    "daily_min_temp_hka": "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMINT&rformat=csv&station=HKA",
    "daily_mean_temp_hka": "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMTEMP&rformat=csv&station=HKA",
    # HQ station (kept for reference/comparison)
    "daily_max_temp_hko": "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKO",
    "daily_min_temp_hko": "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMINT&rformat=csv&station=HKO",
}

# Open-Meteo API Configuration (free, no API key)
# Coordinates for VHHH (HK International Airport)
OPEN_METEO_LAT = 22.31
OPEN_METEO_LON = 113.93
OPEN_METEO_TIMEZONE = "Asia/Hong_Kong"

OPEN_METEO_APIS = {
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "ensemble": "https://ensemble-api.open-meteo.com/v1/ensemble",
}

# NWP ensemble models for probability estimation
NWP_MODELS = ["ecmwf_ifs025", "gfs_seamless"]  # 50 + 30 members = 80 total

# DATA.GOV.HK Dataset IDs for historical CSV downloads
# Using HKA (Airport) station to match Polymarket VHHH resolution
DATAGOVHK_DATASETS = {
    "daily_max_temp": {
        "url": "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMAXT&rformat=csv&station=HKA",
        "description": "Daily maximum temperature at HK Airport (VHHH match)",
    },
    "daily_min_temp": {
        "url": "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMMINT&rformat=csv&station=HKA",
        "description": "Daily minimum temperature at HK Airport (VHHH match)",
    },
    "daily_mean_temp": {
        "url": "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMTEMP&rformat=csv&station=HKA",
        "description": "Daily mean temperature at HK Airport",
    },
    "daily_rainfall": {
        "url": "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMRF&rformat=csv&station=HKA",
        "description": "Daily total rainfall at HK Airport",
    },
    "daily_rh": {
        "url": "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMRH&rformat=csv&station=HKA",
        "description": "Daily mean relative humidity at HK Airport",
    },
    "daily_wind_speed": {
        "url": "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMWS&rformat=csv&station=HKA",
        "description": "Daily mean wind speed at HK Airport",
    },
    "daily_pressure": {
        "url": "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMP&rformat=csv&station=HKA",
        "description": "Daily mean pressure at HK Airport",
    },
}

# Climatological normals (1991-2020)
NORMALS_URL = "https://www.hko.gov.hk/en/cis/normal/1991_2020/normals.htm"

# Model Configuration
MODEL_PARAMS = {
    "lightgbm": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    },
    "xgboost": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    },
}

# Feature engineering
ROLLING_WINDOWS = [3, 7, 14, 30]  # days for rolling averages
LAG_DAYS = [1, 2, 3, 7]  # lag features

# Polymarket bucket definitions (ACTUAL market structure as of Aug 2026)
# Source: https://polymarket.com/event/highest-temperature-in-hong-kong-on-august-6-2026
# Resolution: HKO "Absolute Daily Max (deg. C)" from Daily Extract
#
# IMPORTANT: Bucket ranges shift per day based on expected temperatures.
# These are the DEFAULT/most common buckets. The scraper dynamically
# discovers actual buckets from the live market.

DEFAULT_HIGH_TEMP_BUCKETS = [
    ("27°C or below", -999, 28),   # < 28°C
    ("28°C", 28, 29),
    ("29°C", 29, 30),
    ("30°C", 30, 31),
    ("31°C", 31, 32),
    ("32°C", 32, 33),
    ("33°C", 33, 34),
    ("34°C", 34, 35),
    ("35°C", 35, 36),
    ("36°C", 36, 37),
    ("37°C or higher", 37, 999),   # >= 37°C
]

DEFAULT_LOW_TEMP_BUCKETS = [
    ("21°C or below", -999, 22),   # < 22°C
    ("22°C", 22, 23),
    ("23°C", 23, 24),
    ("24°C", 24, 25),
    ("25°C", 25, 26),
    ("26°C", 26, 27),
    ("27°C", 27, 28),
    ("28°C", 28, 29),
    ("29°C", 29, 30),
    ("30°C", 30, 31),
    ("31°C or higher", 31, 999),   # >= 31°C
]

# Extended buckets for very hot days (discovered from Aug 7-8 markets)
EXTENDED_HIGH_TEMP_BUCKETS = [
    ("29°C or below", -999, 30),
    ("30°C", 30, 31),
    ("31°C", 31, 32),
    ("32°C", 32, 33),
    ("33°C", 33, 34),
    ("34°C", 34, 35),
    ("35°C", 35, 36),
    ("36°C", 36, 37),
    ("37°C", 37, 38),
    ("38°C", 38, 39),
    ("39°C or higher", 39, 999),
]

# Training/test split year
TRAIN_END_YEAR = 2023
VAL_END_YEAR = 2024
