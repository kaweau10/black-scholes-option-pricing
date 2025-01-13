import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def load_processed_data(root_dir="black_scholes_ml", filename="combined_processed_data.csv"):
    """
    Load the processed data from the processed folder.
    """
    file_path = os.path.join(root_dir, "data", "processed", filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data file not found at {file_path}")
    return pd.read_csv(file_path)

def plot_distributions(data):
    """
    Generate distributions of key features to evaluate preprocessing.
    """
    features_to_plot = [
        "lastPrice", "moneyness", "time_to_maturity", "iv_proxy", 
        "historical_volatility", "log_iv_proxy", "moneyness_bin"
    ]
    for feature in features_to_plot:
        if feature in data.columns:
            sns.histplot(data[feature], kde=True, bins=30)
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.savefig(f"black_scholes_ml/reports/visualizations/{feature}_distribution.png")
            plt.show()
            plt.close()

def scatter_plots(data):
    """
    Create scatter plots of features against the target (option price).
    """
    scatter_features = ["moneyness", "time_to_maturity", "iv_proxy", "historical_volatility"]
    for feature in scatter_features:
        if feature in data.columns:
            plt.figure(figsize=(8, 6))
            plt.scatter(data[feature], data["lastPrice"], alpha=0.5)
            plt.xlabel(feature)
            plt.ylabel("Option Price")
            plt.title(f"{feature} vs. Option Price")
            plt.grid(True)
            plt.savefig(f"black_scholes_ml/reports/visualizations/{feature}_scatter.png")
            plt.show()
            plt.close()

def summary_statistics(data):
    """
    Print summary statistics and skewness of numeric features only.
    """
    numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
    print("Summary Statistics:\n", numeric_data.describe())
    print("\nSkewness:\n", numeric_data.skew())
    print("\nCorrelations:\n", numeric_data.corr()["lastPrice"].sort_values(ascending=False))

def main():
    # Load processed data
    root_dir = "black_scholes_ml"
    data = load_processed_data(root_dir)

    # Generate plots
    print("Generating distributions...")
    plot_distributions(data)

    print("Generating scatter plots...")
    scatter_plots(data)

    # Print summary statistics
    print("Computing summary statistics...")
    summary_statistics(data)

if __name__ == "__main__":
    main()
