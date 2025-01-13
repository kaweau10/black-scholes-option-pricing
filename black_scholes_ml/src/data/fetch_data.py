import os
import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date, root_dir="black_scholes_ml"):
    """
    Fetch historical stock data and save it as a CSV file in the raw data folder.
    """
    # Construct the path relative to the main script
    save_path = os.path.join(root_dir, "data", "raw")
    os.makedirs(save_path, exist_ok=True)

    print(f"Fetching stock data for {ticker} from {start_date} to {end_date}")
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    stock_file = os.path.join(save_path, f"{ticker}_stock_data.csv")
    stock_data.to_csv(stock_file)
    print(f"Saved stock data to {stock_file}")
    return stock_data

def fetch_options_data(ticker, expiration, root_dir="black_scholes_ml"):
    """
    Fetch options chain data and save calls and puts as CSV files in the raw data folder.  Check expiration date validity.
    """
    # Construct the path relative to the main script
    save_path = os.path.join(root_dir, "data", "raw")
    os.makedirs(save_path, exist_ok=True)

    print(f"Fetching options data for {ticker} at expiration {expiration}")
    stock = yf.Ticker(ticker)
    available_expirations = stock.options
    if expiration not in available_expirations:
        print(f"Expiration {expiration} is invalid. Using the first available expiration: {available_expirations[0]}")
        expiration = available_expirations[0]
    options_chain = stock.option_chain(expiration)

    calls_file = os.path.join(save_path, f"{ticker}_calls_{expiration}.csv")
    puts_file = os.path.join(save_path, f"{ticker}_puts_{expiration}.csv")
    options_chain.calls.to_csv(calls_file, index=False)
    options_chain.puts.to_csv(puts_file, index=False)
    print(f"Saved call options to {calls_file}")
    print(f"Saved put options to {puts_file}")
    return expiration
