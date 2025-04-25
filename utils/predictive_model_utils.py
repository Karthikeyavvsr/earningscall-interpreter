# utils/predictive_model_utils.py
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import streamlit as st # For caching

# --- Helper Functions ---

@st.cache_data(ttl=3600, show_spinner=False) # Cache data for 1 hour
def get_historical_data(ticker, period="5y", interval="1d"):
    """Fetches historical stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
             return hist # Return empty DataFrame if no data

        # Ensure index is datetime type and handle potential timezone issues
        if not isinstance(hist.index, pd.DatetimeIndex):
             hist.index = pd.to_datetime(hist.index)
        if hist.index.tz:
             hist.index = hist.index.tz_localize(None) # Remove timezone for simplicity

        # --- CHANGE HERE: Return the full DataFrame, drop rows where 'Close' is NaN ---
        return hist.dropna(subset=['Close'])
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        # Return an empty DataFrame on error
        return pd.DataFrame()

# --- Prediction Models (Placeholders/Simple Examples) ---

def predict_intraday(ticker):
    """
    Placeholder for Intraday prediction (e.g., next few hours).
    Requires high-frequency data (e.g., 1m or 5m intervals), which has limitations with free yfinance.
    """
    # Fetch recent intraday data (limited availability with yfinance free tier)
    # Ensure this returns a DataFrame now
    hist_data = get_historical_data(ticker, period="7d", interval="60m") # Example: Hourly data for 7 days

    # Check if DataFrame is not empty and has 'Close' column
    if hist_data.empty or 'Close' not in hist_data.columns or len(hist_data) < 10:
        return None, 0.0, "Not enough recent intraday data available."

    data = hist_data['Close'] # Use the 'Close' series for modeling

    # Simple Example: Predict next hour based on linear trend of last 5 hours
    try:
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        model = LinearRegression()
        model.fit(X[-5:], y[-5:]) # Fit on last 5 data points
        next_point_index = X[-1][0] + 1
        prediction = model.predict([[next_point_index]])[0]
        # Very basic confidence (could be improved with model evaluation)
        confidence = 0.3 # Low confidence for simple model
        trend = "Up" if prediction > y[-1] else "Down"
        explanation = f"Simple linear trend projection based on the last 5 hours. Prediction: {prediction:.2f}. Trend: {trend}."
        return prediction, confidence, explanation
    except Exception as e:
        return None, 0.0, f"Intraday prediction failed: {e}"


def predict_short_term(ticker, days=7):
    """
    Predicts stock price for the next few days (e.g., 1-7 days) using ARIMA.
    """
    hist_data = get_historical_data(ticker, period="1y", interval="1d") # Use 1 year of daily data

    if hist_data.empty or 'Close' not in hist_data.columns or len(hist_data) < 30:
        return None, 0.0, "Not enough historical daily data for short-term prediction."

    data = hist_data['Close'] # Use the 'Close' series

    try:
        # Simple ARIMA model (parameters p,d,q might need tuning)
        model = ARIMA(data, order=(5,1,0)) # Example order, adjust as needed
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=days)
        prediction = forecast.iloc[-1] # Prediction for the end of the period
        # Basic confidence based on forecast variance (example only)
        confidence = max(0.1, 1 - np.mean(model_fit.get_forecast(steps=days).var_pred_mean[-days:]) / data.var()) if data.var() != 0 else 0.1 # Avoid division by zero
        trend = "Up" if prediction > data.iloc[-1] else "Down"
        explanation = f"ARIMA(5,1,0) model prediction for {days} days out. Predicted Price: {prediction:.2f}. Trend: {trend}."
        return prediction, confidence, explanation
    except Exception as e:
        # Common issue: Non-stationarity or model fitting errors
        return None, 0.0, f"Short-term prediction failed (ARIMA): {e}. Data might require transformation (differencing)."


def predict_long_term(ticker, days=30):
    """
    Predicts stock price further out (e.g., 30+ days) using a simple trend.
    Note: Long-term prediction is highly speculative.
    """
    hist_data = get_historical_data(ticker, period="5y", interval="1d") # Use 5 years of daily data

    if hist_data.empty or 'Close' not in hist_data.columns or len(hist_data) < 60: # Need more data for long term
        return None, 0.0, "Not enough historical data for long-term prediction."

    data = hist_data['Close'] # Use the 'Close' series

    try:
        # Very Simple Example: Linear Regression over the whole period
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        model = LinearRegression()
        model.fit(X, y)
        future_indices = np.arange(len(data), len(data) + days).reshape(-1, 1)
        predictions = model.predict(future_indices)
        prediction = predictions[-1] # Final prediction
        confidence = 0.2 # Very low confidence for simple long-term model
        trend = "Up" if prediction > y[-1] else "Down"
        explanation = f"Simple linear trend projection over 5 years for {days} days out. Predicted Price: {prediction:.2f}. Trend: {trend}. (Note: Highly speculative)."
        return prediction, confidence, explanation
    except Exception as e:
        return None, 0.0, f"Long-term prediction failed: {e}"

# Optional: Add SHAP/Feature Importance visualization placeholders if needed
# def plot_feature_importance(model, feature_names):
#     # Placeholder for SHAP or basic feature importance plots
#     pass