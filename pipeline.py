"""Reusable function for the Day-Ahead Trading Engine."""

import os
import logging
import pandas as pd
import numpy as np

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

    # Drop load_actual (we use forecast only to simulate D-1 auction conditions)
    actual_load_col = f"{prefix}load_actual_entsoe_transparency"
    if actual_load_col in df_selected.columns:
        df_selected = df_selected.drop(columns=[actual_load_col])

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

# ============================================================
# FUNCTION 5: Engineer Features
# ============================================================
def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Build the full feature set for day-ahead price forecasting.

    Feature groups:
        1. 48h lags of generation actuals (simulating data availability at 12:00 CET D-1)
        2. 24h lag of price (prior day's prices are always known)
        3. 168h (weekly) lags of price and load
        4. Load ramp (first-order difference)
        5. 24h rolling means of load, wind, and lagged price
        6. Cyclical encoding of hour and weekday (sin/cos)

    Drops the raw generation actuals after lagging to ensure the model only sees
    information available at the auction time.
    """

    df = df.copy()
    price_col = config["price_col"]
    load_col = config["load_forecast_col"]
    solar_col = config["solar_col"]
    wind_cols = config["wind_cols"]

    # 1. Lag generation actuals by 48h (not available at D-1 12:00 auction time)
    lagged_wind_cols = []
    for wind_col in wind_cols:
        lagged_name = f"{wind_col}_lag48h"
        df[lagged_name] = df[wind_col].shift(48)
        lagged_wind_cols.append(lagged_name)
    
    df[f"{solar_col}_lag48h"] = df[solar_col].shift(48)

    # Drop raw actuals
    df = df.drop(columns=wind_cols + [solar_col])

    # 2. Lag price by 24h
    df[f"{price_col}_lag24h"] = df[price_col].shift(24)

    # 3. Weekly (168h) lags
    df[f"{price_col}_lag168h"] = df[price_col].shift(168)
    df[f"{load_col}_lag168h"] = df[load_col].shift(168)

    # 4. load ramp (first difference of forecast load)
    df["load_ramp"] = df[load_col].diff()

    # 5. 24h rolling means
    df[f"{load_col}_rolling24h"] = df[load_col].rolling(window=24).mean()
    df[f"{price_col}_lag24h_rolling24h"] = df[f"{price_col}_lag24h"].rolling(window=24).mean()
    for lagged_name in lagged_wind_cols:
        df[f"{lagged_name}_rolling24h"] = df[lagged_name].rolling(window=24).mean()
    
    # Trim the first 168 rows
    df = df.iloc[168:].copy()

    # 6. Cyclical encoding of time
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)

    return df

# Helper for split_and_fill
def _fill_seasonal(df: pd.DataFrame, period: int = 24) -> pd.DataFrame:
    """
    Fill NaN values with the value from the same hour 'period' hours ago,
    preserving daily seasonality. Falls back to forward-fill, then back-fill,
    for any remaining NaNs at the edges.
    """
    df_filled = df.copy()
    for col in df_filled.columns:
        df_filled[col] = df_filled[col].fillna(df_filled[col].shift(period))
        df_filled[col] = df_filled[col].ffill()
        df_filled[col] = df_filled[col].bfill()
    return df_filled

# ============================================================
# FUNCTION 6: Train/Test Split with Seasonal NaN Fill
# ============================================================
def split_and_fill(
        df: pd.DataFrame,
        config: dict,
        test_weeks: int = 2
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split the engineered DataFrame into train and test sets, 
    then fill NaN values within each set independently using 
    a seasonal pattern (same hour 24h prior, then ffill, then bfill)

    The fill happens AFTER the split to prevent leakage: if we filled
    before splitting, the test set could inherit seasonal patterns
    derived from the train data, which would bias evaluation.

    Returns:
        X_train, y_train, X_test, y_test
    """
    price_col = config["price_col"]
    test_hours = 24 * 7 * test_weeks

    split_idx = len(df) - test_hours
    split_date = df.index[split_idx]
    logger.info(f"Train ends / Test starts at: {split_date}")
    logger.info(f"Test duration: {test_hours} hours ({test_weeks} weeks)")

    train_set = df.iloc[:split_idx].copy()
    test_set = df.iloc[split_idx:].copy()

    # Fill NaNs using seasonal pattern (same hour 24h ago), then ffill/bfill
    train_set = _fill_seasonal(train_set)
    test_set = _fill_seasonal(test_set)

    # Separate target from features
    y_train = train_set[price_col].copy()
    y_test = test_set[price_col].copy()
    X_train = train_set.drop(columns=[price_col])
    X_test = test_set.drop(columns=[price_col])

    return X_train, y_train, X_test, y_test

# ============================================================
# FUNCTION 7: Train XGBoost
# ============================================================
def train_xgboost(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_estimators: int = 3000,
        learning_rate: float = 0.01,
        max_depth: int = 8,
        subsample: float = 0.7,
        colsample_bytree: float = 0.7,
        early_stopping_rounds: int = 100,
        random_state: int = 42,
        verbose: int = 100,
):
    """
    Train an XGBoost regressor on the engineered features and return the fitted model.

    Early stopping watches the test set's loss (last eval_set entry) to pick the
    optimal number of trees. The model exposes feature importance after fitting.

    Returns:
        The fitted XGBRegressor.
    """
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=-1,
        random_state=random_state,
    )

    logger.info("Training XGBoost...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=verbose,
    )

    return model

def generate_daily_signals(
        y_pred: pd.Series, 
        y_actual: pd.Series, 
        top_n: int=2
        ) -> pd.DataFrame:
    """
    Generate daily buy/sell signals from forecasted prices.

    For each day, the 'top_n' cheapest predicted hours are marked as BUY (1),
    the 'top_n' most expensive predicted hours are marked as SELL (-1),
    and all other hours are HOLD (0).

    The Day-Ahead market lets you see all 24 hourly predictions at once,
    so ranking within a day is the natural unit of decision.

    Returns a DataFrame indexed by timestamp with columns: actual, pred, signal.
    """
    df_sim = pd.DataFrame({
        "actual": y_actual,
        "pred": y_pred
    }, index=y_actual.index)

    df_sim["date"] = df_sim.index.date

    def _signals_for_day(group: pd.DataFrame) -> pd.Series:
        ranks = group["pred"].rank(method="first")
        signals = pd.Series(0, index=group.index)
        signals[ranks <= top_n] = 1
        signals[ranks > (len(group) - top_n)] = -1
        
        return signals
    
    df_sim["signal"] = (
        df_sim.groupby("date")
        .apply(_signals_for_day)
        .reset_index(0, drop=True)
    )

    return df_sim

def run_battery_backtest(
        signals_df: pd.DataFrame,
        capacity_mwh: float = 1.0,
        max_rate_mw: float = 1.0,
        efficiency: float = 0.90
        ) -> pd.DataFrame:
    """
    Simulate a battery acting on the buy/sell signals.

    The battery state evolves sequentially through time. On a BUY signal it charges
    up to capacity at the actual market price. On a SELL signal it discharges what
    it has, with the round-trip efficiency penalty applied to discharge revenue.

    No directional bets, the strategy only captures intraday spreads.

    Returns a DataFrame indexed by timestamp with columns: pnl, soc, action.
    """
    current_soc = 0.0
    cash = 0.0
    history = []

    logger.info("Running battery backtest...")

    for t in range(len(signals_df)):
        row = signals_df.iloc[t]
        price = row["actual"]
        sig = row["signal"]
        action = "HOLD"

        if sig == 1 and current_soc < capacity_mwh:
            energy_to_buy = min(max_rate_mw, capacity_mwh - current_soc)
            cash -= energy_to_buy * price
            current_soc += energy_to_buy
            action = "CHARGE"
        
        elif sig == -1 and current_soc > 0:
            energy_to_sell = min(max_rate_mw, current_soc)
            cash += energy_to_sell * price * efficiency
            current_soc -= energy_to_sell
            action = "DISCHARGE"

        history.append({"pnl": cash, "soc": current_soc, "action": action})

    results = pd.DataFrame(history, index=signals_df.index)
    logger.info(f"Final trading PnL: €{cash:.2f}")

    return results