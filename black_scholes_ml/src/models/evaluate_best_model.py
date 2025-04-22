import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# === Load Data Function ===
def load_combined_data():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    file_path = os.path.join(root_dir, "data", "processed", "combined_processed_data.csv")
    return pd.read_csv(file_path)

# === Preprocessing Function ===
def preprocess_data(data, features, target):
    data_cleaned = data.dropna(subset=[target])
    X = data_cleaned[features]
    y = data_cleaned[target]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y

def main():
    # Define features and target
    features = [
        "moneyness", "time_to_maturity", "iv_proxy", "historical_volatility",
        "log_iv_proxy", "moneyness_bin", "moneyness_time", "moneyness_squared"
    ]
    target = "lastPrice"

    # Load and preprocess data
    data = load_combined_data()
    X, y = preprocess_data(data, features, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the best tuned model
    model_path = os.path.join("models", "neural_network", "best_tuned_model.h5")
    model = tf.keras.models.load_model(model_path)

    # Predict and evaluate
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Final Evaluation of Tuned Model:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Save evaluation results
    report_path = os.path.join("reports", "model_evaluations", "neural_network_tuned_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"Mean Squared Error: {mse:.4f}\n")
        f.write(f"R^2 Score: {r2:.4f}\n")

    # === Visualization Directory ===
    vis_dir = os.path.join("reports", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # --- Plot 1: Predicted vs. Actual ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel("Actual Option Price")
    plt.ylabel("Predicted Option Price")
    plt.title("Predicted vs. Actual Option Prices")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "predicted_vs_actual.png"))
    plt.show()

    # --- Plot 2: Residual Plot ---
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, residuals, alpha=0.4)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual Option Price")
    plt.ylabel("Prediction Error (Residual)")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "residual_plot.png"))
    plt.show()

if __name__ == "__main__":
    main()
