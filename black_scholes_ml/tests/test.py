import yfinance as yf
data = yf.download("BKNG", start="2022-01-01", end="2023-01-01")
print(data.head())