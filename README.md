# Day Ahead Power Trading Engine

## Project Overview
This project focuses on predicting **Day-Ahead Power Prices (Europe/Vienna)** to drive a **Market-Neutral Battery Arbitrage Strategy.**

Moving beyond standard univariate time-series analysis, this project iterates through Linear Regression, rolling ARIMA, and finally XGBoost to capture the non-linear dynamics of the energy merit order curve. The final predictive model powers a backtested simulation of a 1MW Battery Storage asset, optimizing charge/discharge cycles based on dynamic predicted intraday spreads.

## Key Results
- **Final Model:** XGBoost Regressor (Gradient Boosting)
- **Performance:** RMSE ~16.62 (Small improvment over ARIMA)
- **Trading Strategy:** Dynamic Percentile Arbitrage (Market Neutral)
- **Outcome:** Positive PnL in backtesting (see notebook for equity curve).

## Tech Stack
- **Core:** Python, Pandas, NumPy
- **Modeling:** Scikit-Learn (Linear OLS), Statsmodels/PMDARIMA (ARIMA), XGBoost
- **Visualization:** Matplotlib/Seaborn

---

## 🛑 The "War Stories": Challenges & Solutions
*Quantitative modeling is rarely a straight line. Here are the specific structural challenges I encountered and how I solved them:*

### 1. The "Look-Ahead" Trap (Data Leakage)
* **Diagnosis:** Early models achieved suspiciously low RMSE scores (<5.0). Upon inspection, I realized the model was accessing "Actual" Wind/Load generation, which is not available at the 12:00 CET auction time.
* **Fix:** I enforced a strict **Information Horizon**, replacing all actuals with **Forecasts** (ENTSO-E) or **48h Lags** to simulate realistic Day-Ahead data availability.

### 2. The Linearity Wall (ARIMA Limitations)
* **Hypothesis:** A standard ARIMA(5,1,4) would capture the autoregressive nature of prices.
* **Failure:** The model failed to predict extreme volatility (spikes). It smoothed out the "Merit Order Effect" where low wind + high load causes exponential price jumps.
* **Solution:** Pivoted to **XGBoost** to capture non-linear interactions between exogenous drivers (Wind Forecasts, Load Forecasts) and Price.

### 3. Feature Engineering: The "Midnight Gap"
* **Problem:** Raw numerical inputs for Hours (0-23) confused linear models, as "Hour 23" and "Hour 0" appeared numerically distant despite being temporally adjacent.
* **Solution:** Implemented **Cyclical Encoding** (Sine/Cosine transformation) to map time onto a 2D circle, preserving the continuity of the daily cycle.

---

## Strategy Logic: Battery Arbitrage
Instead of taking directional bets (which are risky during high-volatility regimes like the 2022 Energy Crisis), I implemented a **Spread Capture Strategy**:
1. **Forecast** the next 24 hourly prices at 12:00 D-1.
2. **Identify** the dynamic percentiles (Cheapest 25% hours vs. Most Expensive 25%).
3. **Simulate** a 1MWh battery:
   - **Charge** during the bottom 2 hours.
   - **Discharge** during the top 2 hours.
   
*Full backtest results and equity curves are available in the notebook.*

## Future Improvements & Critical Reflections
*I recently reviewed this project with a Senior Risk Manager who highlighted several statistical limitations in my Linear Regression baseline. If I were to continue this project, I would focus on fixing these fundamental issues:*

### 1. Checking Model Assumptions (Diagnostics)
I learned that Linear Regression (OLS) makes strict assumptions that time-series data often violates. I need to explicitly check for:
* **Serial Correlation:** The current model likely assumes that yesterday's error doesn't affect today's, which isn't true for power prices. I need to check the residuals to see if patterns still exist.
* **Heteroskedasticity:** The "variance" of the errors isn't constant (prices are much more volatile in winter or crisis months). I need to verify this and potentially transform the data to fix it.

### 2. Better Feature Selection
* **Bivariate Analysis:** Instead of feeding all data into the model at once, I should plot each feature (like Wind Forecast) against the Price individually to understand the relationship better.
* **Stationarity:** I need to make sure my input variables (Xs) are "stationary" (constant mean/variance). Using raw time-series data in a linear model can lead to misleading results, so I should probably use the *change* in price/load rather than the absolute value.

### 3. Moving Beyond OLS
* Because my target (Price) and inputs (Load/Wind) are all time-series with their own trends, a simple Linear Regression is likely "mis-specified."
* I would explore **Generalized Least Squares (GLS)** or more robust time-series specific models to handle these correlations correctly.
