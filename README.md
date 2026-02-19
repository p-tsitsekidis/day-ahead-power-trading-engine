# Day-Ahead Power Trading Engine

## Project Overview
A quantitative forecasting and backtesting engine designed to predict **Day-Ahead Power Prices (Europe/Vienna)** and drive a **Market-Neutral Battery Arbitrage Strategy**.

Moving beyond standard univariate time-series analysis, this project iterates through Linear Regression, rolling ARIMA, and finally XGBoost to capture the highly non-linear dynamics of the energy merit order curve. The final predictive model powers a custom backtested simulation of a 1MWh Battery Storage asset, optimizing charge/discharge cycles based on dynamic intraday spreads.


## Key Results & Tech Stack
* **Final Model:** XGBoost Regressor (Gradient Boosting) over an ARIMA(5,1,4) baseline.
* **Performance:** RMSE ~16.62 
* **Trading Strategy:** Dynamic Percentile Spread Capture (Market Neutral)
* **Outcome:** Positive PnL in backtesting across high-volatility regimes (see Jupyter notebook for equity curves).
* **Tech Stack:** Python, Pandas, NumPy, Scikit-Learn, PMDARIMA, XGBoost.

---

## The Execution: Battery Arbitrage Strategy
Instead of taking directional bets—which carry severe risk during high-volatility regimes like the 2022 Energy Crisis—this engine implements a strict **Spread Capture Strategy**:

1. **Forecast:** Generate the next 24 hourly prices at 12:00 CET (D-1 auction).
2. **Identify:** Calculate dynamic percentiles (identifying the cheapest 25% of hours vs. the most expensive 25% of hours for that specific day).
3. **Simulate Execution:** Automate a 1MWh battery to charge during the bottom 2 hours and discharge during the top 2 hours, enforcing a strict 90% round-trip efficiency penalty.

---

## Quantitative Research Notes: Overcoming Structural Challenges

Quantitative modeling is rarely linear. Below are the specific market micro-structure and statistical challenges encountered, and how they were resolved:

### 1. The "Look-Ahead" Trap (Data Leakage)
* **Diagnosis:** Early baseline models achieved suspiciously low RMSE scores. Upon auditing the feature pipeline, I realized the model was accessing "Actual" Wind and Load generation data—which is not available at the 12:00 CET day-ahead auction time.
* **Resolution:** Enforced a strict **Information Horizon**, replacing all actuals with ENTSO-E **Forecasts** or **48h Lags** to simulate realistic Day-Ahead data availability.

### 2. The Linearity Wall
* **Diagnosis:** A standard ARIMA(5,1,4) model failed to predict extreme volatility. It smoothed out the "Merit Order Effect," where low wind combined with high load causes exponential price spikes.
* **Resolution:** Transitioned to **XGBoost** to capture the non-linear interactions between exogenous drivers (Wind/Load forecasts) and wholesale prices.

### 3. Feature Engineering: The "Midnight Gap"
* **Diagnosis:** Raw numerical inputs for hours ($0$ to $23$) confused linear algorithms. "Hour 23" and "Hour 0" appeared numerically distant despite being temporally adjacent.
* **Resolution:** Implemented **Cyclical Encoding** (Sine/Cosine transformation) to map time onto a 2D circle, preserving the continuous nature of the daily cycle.

---

## Critical Reflections & Future Iterations

Following a review with a Senior Risk Manager, future iterations of this engine will focus on addressing fundamental statistical assumptions present in the baseline models:

1. **Model Diagnostics (Heteroskedasticity & Serial Correlation):** Energy prices exhibit non-constant variance (higher volatility in winter). Future models will explicitly test residuals for autocorrelation and apply necessary variance-stabilizing transformations.
2. **Stationarity:** Feeding raw time-series data into linear models can yield spurious correlations. Future feature engineering will focus on stationarity (e.g., using the $\Delta$ change in load rather than absolute load).
3. **Beyond OLS:** Exploring Generalized Least Squares (GLS) or advanced state-space models to better handle the correlated errors inherent in power market time-series.
