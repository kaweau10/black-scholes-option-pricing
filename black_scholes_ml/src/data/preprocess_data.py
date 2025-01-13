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
        calls_data = preprocess_data(stock_data, calls_data, actual_expiration)
        calls_data = transform_features(calls_data)
        calls_data = add_iv_proxy(calls_data)

        # Append processed data to the cumulative file
        save_processed_data(calls_data, root_dir=root_dir)
        print(f"Processing and saving for {ticker} complete.")
    except Exception as e:
        print(f"Error processing data for {ticker}, expiration {expiration}: {e}")


def load_data(root_dir="black_scholes_ml", ticker="AAPL", expiration="2025-01-10"):
    """
    Load stock and options data from CSV files, correctly parsing the header row.
    """
    # Paths to data files
    stock_file = os.path.join(root_dir, "data", "raw", f"{ticker}_stock_data.csv")
    calls_file = os.path.join(root_dir, "data", "raw", f"{ticker}_calls_{expiration}.csv")
    puts_file = os.path.join(root_dir, "data", "raw", f"{ticker}_puts_{expiration}.csv")

    # Load stock data with correct column headers
    stock_data = pd.read_csv(stock_file, skiprows=2)  # Skip metadata
    stock_data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]  # Set correct column names
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])  # Parse dates
    stock_data.set_index("Date", inplace=True)  # Set date as index

    # Load options data
    calls_data = pd.read_csv(calls_file)
    puts_data = pd.read_csv(puts_file)

    return stock_data, calls_data, puts_data

def preprocess_data(stock_data, options_data, expiration):
    """
    Align stock and options data and calculate derived features.
    """
    # Ensure expiration is a timezone-naive datetime object
    expiration_date = pd.to_datetime(expiration).tz_localize(None)

    # Ensure all dates in the options data are timezone-naive
    options_data["lastTradeDate"] = pd.to_datetime(options_data["lastTradeDate"]).dt.tz_localize(None)

    # Calculate moneyness (S/K) and time to maturity (T)
    options_data["moneyness"] = stock_data["Close"].iloc[-1] / options_data["strike"]
    options_data["time_to_maturity"] = (expiration_date - options_data["lastTradeDate"]).dt.days / 365

    # Drop rows with invalid or missing data
    options_data = options_data.dropna()
    options_data = options_data[options_data["time_to_maturity"] > 0]

    return options_data

def transform_features(data):
    """
    Apply transformations to reduce skewness in features while retaining original columns.
    """
    data["log_moneyness"] = np.log(data["moneyness"])
    data["boxcox_lastPrice"], _ = boxcox(data["lastPrice"] + 1e-5)
    data["log_time_to_maturity"] = np.cbrt(data["time_to_maturity"])

    return data

def add_iv_proxy(data):
    """
    Add a simple proxy for implied volatility and apply transformations.
    """
    # Avoid division by zero or negative values
    data["iv_proxy"] = data["lastPrice"] / (
        (data["log_time_to_maturity"] + 1e-5) * (data["moneyness"] + 1e-5)
    )
    data["iv_proxy"] = data["iv_proxy"].fillna(0)  # Handle any NaN values

    # Box-Cox Transformation
    data["boxcox_iv_proxy"], _ = boxcox(data["iv_proxy"] + 1)

    return data

def normalize_features(data, feature_cols):
    """
    Normalize features using Min-Max Scaling.
    """
    scaler = MinMaxScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    return data

def remove_outliers(data, feature, percentile=0.99):
    """
    Remove outliers above a certain percentile.
    """
    threshold = data[feature].quantile(percentile)
    data = data[data[feature] <= threshold]
    return data

def plot_distributions(data):
    """
    Plot distributions of key features.
    """
    if "log_moneyness" in data:
        sns.histplot(data["log_moneyness"], kde=True, bins=30)
        plt.title("Distribution of Log Moneyness (S/K)")
        plt.xlabel("Log Moneyness")
        plt.ylabel("Frequency")
        plt.savefig("black_scholes_ml/reports/visualizations/log_moneyness_distribution.png")
        plt.show()
        plt.close()

    if "log_time_to_maturity" in data:
        sns.histplot(data["log_time_to_maturity"], kde=True, bins=30)
        plt.title("Distribution of Log Time to Maturity")
        plt.xlabel("Log Transformation of Time to Maturity")
        plt.ylabel("Frequency")
        plt.savefig("black_scholes_ml/reports/visualizations/log_time_to_maturity_distribution.png")
        plt.show()
        plt.close()

    if "boxcox_lastPrice" in data:
        sns.histplot(data["boxcox_lastPrice"], kde=True, bins=30)
        plt.title("Distribution of Box-Cox Option Prices")
        plt.xlabel("Box-Cox Option Price")
        plt.ylabel("Frequency")
        plt.savefig("black_scholes_ml/reports/visualizations/boxcox_lastPrice_distribution.png")
        plt.show()
        plt.close()

    if "boxcox_iv_proxy" in data:
        sns.histplot(data["boxcox_iv_proxy"], kde=True, bins=30)
        plt.title("Distribution of Implied Volatility")
        plt.xlabel("Implied Volatility")
        plt.ylabel("Frequency")
        plt.savefig("black_scholes_ml/reports/visualizations/boxcox_iv_proxy_distribution.png")
        plt.show()
        plt.close()

def scatter_plots(options_data):
    """
    Plot scatter plots of features vs. option prices.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(options_data["moneyness"], options_data["lastPrice"], alpha=0.5)
    plt.xlabel("Moneyness (S/K)")
    plt.ylabel("Option Price")
    plt.title("Moneyness vs. Option Price")
    plt.grid(True)
    plt.savefig("black_scholes_ml/reports/visualizations/moneyness_scatter.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(options_data["time_to_maturity"], options_data["lastPrice"], alpha=0.5)
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Option Price")
    plt.title("Time to Maturity vs. Option Price")
    plt.grid(True)
    plt.savefig("black_scholes_ml/reports/visualizations/time_to_maturity_scatter.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(options_data["boxcox_iv_proxy"], options_data["lastPrice"], alpha=0.5)
    plt.xlabel("Implied Volatility")
    plt.ylabel("Option Price")
    plt.title("Implied Volatility vs. Option Price")
    plt.grid(True)
    plt.savefig("black_scholes_ml/reports/visualizations/boxcox_iv_proxy_scatter.png")
    plt.show()
    plt.close()

def save_processed_data(data, filename="combined_processed_data.csv", root_dir="black_scholes_ml"):
    """
    Append preprocessed data to a cumulative CSV file in the processed folder.
    Creates the file if it does not exist.
    """
    processed_path = os.path.join(root_dir, "data", "processed")
    os.makedirs(processed_path, exist_ok=True)
    file_path = os.path.join(processed_path, filename)

    # Append to existing file or create a new one
    if os.path.exists(file_path):
        data.to_csv(file_path, mode="a", header=False, index=False)
        print(f"Appended data to {file_path}")
    else:
        data.to_csv(file_path, index=False)
        print(f"Created new cumulative file at {file_path}")


def summarize_data(data):
    """
    Compute and display summary statistics and skewness for each feature.
    """
    print("Summary Statistics:\n", data.describe())
    print("\nSkewness:\n", data.skew())

