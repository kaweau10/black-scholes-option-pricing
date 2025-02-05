import os
import pandas as pd
from src.data.fetch_data import fetch_stock_data, fetch_options_data
from src.data.preprocess_data import process_and_save_data


def main():
    path = os.path.join(os.getcwd())
    print(path)

if __name__ == "__main__":
    main()
