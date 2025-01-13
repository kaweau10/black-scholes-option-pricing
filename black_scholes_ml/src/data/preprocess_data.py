from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data.fetch_data import fetch_stock_data, fetch_options_data

def process_and_save_data(ticker, start_date, end_date, expiration, root_dir="black_scholes_ml"):
    """
    Fetch, preprocess, and save data for a specific stock and expiration.
    """
    try:
        # Fetch stock data
        fetch_stock_data(ticker, start_date, end_date, root_dir)

        # Fetch options data and get the actual expiration date
        actual_expiration = fetch_options_data(ticker, expiration, root_dir)

        # Load and preprocess data
        stock_data, calls_data, puts_data = load_data(root_dir, ticker, actual_expiration)
        calls_data = preprocess_data_with_outliers_and_features(stock_data, calls_data, actual_expiration)

        # Append processed data to the cumulative file
        save_processed_data(calls_data, root_dir=root_dir)
        print(f"Processing and saving for {ticker} complete.")
    except Exception as e:
        print(f"Error processing data for {ticker}, expiration {expiration}: {e}")

def load_data(root_dir="black_scholes_ml", ticker="AAPL", expiration="2025-01-10"):
    """
    Load stock and options data from CSV files, correctly parsing the header row.
    """
    stock_file = os.path.join(root_dir, "data", "raw", f"{ticker}_stock_data.csv")
    calls_file = os.path.join(root_dir, "data", "raw", f"{ticker}_calls_{expiration}.csv")
    puts_file = os.path.join(root_dir, "data", "raw", f"{ticker}_puts_{expiration}.csv")

    stock_data = pd.read_csv(stock_file, skiprows=2)
    stock_data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    stock_data.set_index("Date", inplace=True)

    calls_data = pd.read_csv(calls_file)
    puts_data = pd.read_csv(puts_file)

    return stock_data, calls_data, puts_data

def preprocess_data(stock_data, options_data, expiration):
    expiration_date = pd.to_datetime(expiration).tz_localize(None)
    options_data["lastTradeDate"] = pd.to_datetime(options_data["lastTradeDate"]).dt.tz_localize(None)

    options_data["moneyness"] = stock_data["Close"].iloc[-1] / options_data["strike"]
    options_data["time_to_maturity"] = (expiration_date - options_data["lastTradeDate"]).dt.days / 365

    options_data = options_data.dropna()
    options_data = options_data[options_data["time_to_maturity"] > 0]

    return options_data

def remove_outliers(data, features, lower_percentile=0.01, upper_percentile=0.99):
    """
    Winsorize the outliers in the specified features by capping them at the given percentiles.
    """
    for feature in features:
        lower_bound = data[feature].quantile(lower_percentile)
        upper_bound = data[feature].quantile(upper_percentile)
        data[feature] = np.clip(data[feature], lower_bound, upper_bound)
    return data

def preprocess_data_with_outliers_and_features(stock_data, options_data, expiration):
    # Preprocess basic features
    options_data = preprocess_data(stock_data, options_data, expiration)

    # Engineer new features
    options_data = engineer_features(options_data)

    # Remove outliers
    outlier_features = ["lastPrice", "moneyness", "iv_proxy"]
    options_data = remove_outliers(options_data, outlier_features)

    # Reorder columns to ensure important ones are saved
    required_columns = [
        "contractSymbol", "lastTradeDate", "strike", "lastPrice", "bid", "ask",
        "volume", "openInterest", "impliedVolatility", "moneyness",
        "time_to_maturity", "iv_proxy", "historical_volatility", "log_iv_proxy",
        "moneyness_bin", "moneyness_time", "moneyness_squared"
    ]

    # Keep only the required columns, dropping unnecessary ones
    options_data = options_data[required_columns]

    return options_data

def engineer_features(data):
    """
    Add new features to enhance model inputs.
    """
    # Calculate implied volatility proxy
    data["iv_proxy"] = data["lastPrice"] / (
        (data["time_to_maturity"] + 1e-5) * (data["moneyness"] + 1e-5)
    )
    data["iv_proxy"] = data["iv_proxy"].fillna(0)

    # Historical volatility proxy
    data["historical_volatility"] = data["lastPrice"].rolling(window=20).std()

    # Interaction terms
    data["moneyness_time"] = data["moneyness"] * data["time_to_maturity"]
    data["moneyness_squared"] = data["moneyness"] ** 2

    # Binning moneyness into quantiles
    data["moneyness_bin"] = pd.qcut(data["moneyness"], q=5, labels=False)

    # Log transformation for highly skewed features
    data["log_iv_proxy"] = np.log1p(data["iv_proxy"])

    return data

def save_processed_data(data, filename="combined_processed_data.csv", root_dir="black_scholes_ml"):
    processed_path = os.path.join(root_dir, "data", "processed")
    os.makedirs(processed_path, exist_ok=True)
    file_path = os.path.join(processed_path, filename)

    if os.path.exists(file_path):
        data.to_csv(file_path, mode="a", header=False, index=False)
        print(f"Appended data to {file_path}")
    else:
        data.to_csv(file_path, index=False)
        print(f"Created new cumulative file at {file_path}")
