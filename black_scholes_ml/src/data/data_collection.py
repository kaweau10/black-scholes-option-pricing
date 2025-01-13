import os
import pandas as pd
from src.data.fetch_data import fetch_stock_data, fetch_options_data
from src.data.preprocess_data import process_and_save_data

def load_tickers(file_path):
    """Load tickers from a structured text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ticker file not found at {file_path}")
    tickers_data = pd.read_csv(file_path, sep="|")
    tickers = tickers_data["Symbol"].tolist()
    return tickers

def collect_data(tickers, start_date, end_date, expiration):
    """Fetch, preprocess, and save data for all tickers."""
    for ticker in tickers:
        print(f"Processing data for {ticker}...")
        try:
            process_and_save_data(ticker, start_date, end_date, expiration)
        except Exception as e:
            print(f"Error processing data for {ticker}: {e}")

def main():
    root_dir = "black_scholes_ml"

    # File paths
    tickers_file = os.path.join(root_dir, "data", "tickers.txt")

    # Load tickers
    print("Loading tickers...")
    tickers = load_tickers(tickers_file)

    # Define date range and expiration
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    expiration = "2024-12-20"

    # Collect and process data
    print("Starting data collection...")
    collect_data(tickers, start_date, end_date, expiration)
    print("Data collection complete.")

if __name__ == "__main__":
    main()
