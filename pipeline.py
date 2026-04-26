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
        "wind_cols": ["AT_wind_onshore_generation_actual"],
        "solar_col": "AT_solar_generation_actual"
    },
    "DK_1": {
        "prefix": "DK_1_",
        "timezone": "Europe/Copenhagen",
        "price_col": "DK_1_price_day_ahead",
        "load_forecast_col": "DK_1_load_forecast_entsoe_transparency",
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

# ============================================================
# FUNCTION 2: Select Market Columns & Set Index
# ============================================================
def select_market_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Filter the raw OPSD DataFrame to a single market's columns,
    convert the timestamp to the market's local timezone, and set it as index.
    """
    prefix = config["prefix"]
    timezone = config["timezone"]

    market_cols = df.columns[df.columns.str.startswith(prefix)]
    selected_cols = ["cet_cest_timestamp"] + list(market_cols)
    df_selected = df[selected_cols].copy()

    df_selected["cet_cest_timestamp"] = pd.to_datetime(df_selected["cet_cest_timestamp"], utc=True).dt.tz_convert(timezone)
    df_selected = df_selected.set_index("cet_cest_timestamp")

    return df_selected

# ============================================================
# FUNCTION 3: Enforce Hourly Frequency
# ============================================================
def enforce_hourly_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex the DataFrame to a complete hourly range, filling any missing timestamps with NaN. 
    This ensures downstream time-based operations (lags, rolling windows) work on a complete grid.
    """
    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h",
        tz=df.index.tz
    )
    return df.reindex(full_idx)

# ============================================================
# FUNCTION 4: Slice to Valid Range
# ============================================================
def slice_to_valid_range(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Find the common time range where all the important columns have data,
    and slice the DataFrame to that range. Different columns start and end
    at different times in the OPSD dataset, so we slice to the intersection
    to avoid leading/trailing NaN periods.
    """
    important_cols = [
        config["load_forecast_col"],
        config["price_col"],
        config["solar_col"]
        ] + config["wind_cols"]
    
    first_valid = df[important_cols].apply(lambda col: col.first_valid_index())
    last_valid = df[important_cols].apply(lambda col: col.last_valid_index())

    global_start = first_valid.max()
    global_end = last_valid.min()

    logger.info(f"Slicing {config['prefix']} data to range: {global_start} to {global_end}")

    return df.loc[global_start:global_end].copy()