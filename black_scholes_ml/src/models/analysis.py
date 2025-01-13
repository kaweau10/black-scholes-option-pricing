import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
def load_combined_data():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    file_path = os.path.join(root_dir,"data", "processed", "combined_processed_data.csv")
    return pd.read_csv(file_path)

# Plot scatter plots for selected features
def plot_scatter(data, features, target, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for feature in features:
        plt.figure(figsize=(8, 6))
        plt.scatter(data[feature], data[target], alpha=0.5)
        plt.title(f"{feature} vs. {target}")
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.grid(True)

        # Save the plot
        output_path = os.path.join(output_dir, f"{feature}_vs_{target}.png")
        plt.savefig(output_path)
        plt.show()
        print(f"Saved scatter plot to {output_path}")

def calculate_correlations(data, features, target):
    print("Correlation Analysis:")
    
    # Pearson correlation
    print("\nPearson Correlation:")
    for feature in features:
        correlation = data[[feature, target]].corr().iloc[0, 1]
        print(f"{feature} vs. {target}: {correlation:.4f}")

    # Spearman correlation
    print("\nSpearman Correlation:")
    for feature in features:
        correlation = data[[feature, target]].corr(method='spearman').iloc[0, 1]
        print(f"{feature} vs. {target}: {correlation:.4f}")

if __name__ == "__main__":
    # Load data
    data = load_combined_data()

    # Define features and target
    features = ['log_moneyness', 'log_time_to_maturity', 'boxcox_iv_proxy']
    target = 'boxcox_lastPrice'

    # Directory to save plots
    output_dir = os.path.join(os.getcwd(), "black_scholes_ml", "reports", "scatter_plots")

    # Generate scatter plots
    plot_scatter(data, features, target, output_dir)

    # Calculate correlations
    calculate_correlations(data, features, target)
