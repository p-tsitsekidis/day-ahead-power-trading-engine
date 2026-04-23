"""Reusable function for the Day-Ahead Trading Engine."""

import os
import logging
import pandas as pd

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================
# MARKET CONFIGURATIONS
# ============================================================
MARKET_CONFIGS = {
    "AT": {
        "prefix": "AT_",
        "timezone": "Europe/Vienna",
        "price_col": "AT_price_day_ahead",
        "load_forecast_col": "AT_load_forecast_entsoe_transparency",
        "load_actual_col": "AT_load_actual_entsoe_transparency",
        "wind_cols": ["AT_wind_onshore_generation_actual"],
        "solar_col": "AT_solar_generation_actual"
    },
    "DK_1": {
        "prefix": "DK_1_",
        "timezone": "Europe/Copenhagen",
        "price_col": "DK_1_price_day_ahead",
        "load_forecast_col": "DK_1_load_forecast_entsoe_transparency",
        "load_actual_col": "DK_1_load_actual_entsoe_transparency",
        "wind_cols": [
            "DK_1_wind_offshore_generation_actual", 
            "DK_1_wind_onshore_generation_actual"
            ],
        "solar_col": "DK_1_solar_generation_actual"
    },
    "DK_2": {
        "prefix": "DK_2_",
        "timezone": "Europe/Copenhagen",
        "price_col": "DK_2_price_day_ahead",
        "load_forecast_col": "DK_2_load_forecast_entsoe_transparency",
        "load_actual_col": "DK_2_load_actual_entsoe_transparency",
        "wind_cols": [
            "DK_2_wind_offshore_generation_actual",
            "DK_2_wind_onshore_generation_actual"
        ],
        "solar_col": "DK_2_solar_generation_actual"
    }
}

# ============================================================
# FUNCTION 1: Load Raw Data
# ============================================================
def load_raw_data(
        url: str = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv",
        local_path: str = "data/power_data_raw.csv"
        ) -> pd.DataFrame:
    """
    Download and cache the OPSD raw CSV. Returns a pandas DataFrame.
    
    If the file already exists locally, it reads from disk instead of downloading.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        logger.info(f"Reading cached CSV from {local_path}")
        df = pd.read_csv(local_path)
    else:
        logger.info(f"Downloading power data from {url}")
        df = pd.read_csv(url)
        df.to_csv(local_path, index=False)
        logger.info(f"Saved to {local_path}")

    return df
