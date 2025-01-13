import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import joblib

MODEL_DIR = os.path.join(os.getcwd(), "black_scholes_ml", "models")

def load_combined_data():
    """Load the combined processed data."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    file_path = os.path.join(root_dir, "data", "processed", "combined_processed_data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The combined data file does not exist at {file_path}")
    return pd.read_csv(file_path)

def preprocess_and_split_data(data, features, target):
    """Prepare data for training."""
    X = data[features]
    y = data[target]
    return X, y

def train_random_forest(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model_with_cv(model, X, y):
    """Evaluate the model using cross-validation."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=cv)
    return mse_scores, r2_scores

def save_model(model, model_name="random_forest"):
    """Save the trained model."""
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    return model_path

def main():
    # Load and preprocess data
    data = load_combined_data()

    # Define features and target
    features = [
        "moneyness", "time_to_maturity", "iv_proxy", "historical_volatility",
        "log_iv_proxy", "moneyness_bin", "moneyness_time", "moneyness_squared"
    ]
    target = "lastPrice"

    if not all(feature in data.columns for feature in features):
        raise ValueError("One or more features are missing in the data. Check preprocessing.")

    # Prepare data
    X, y = preprocess_and_split_data(data, features, target)

    # Train the model
    print("Training the Random Forest model...")
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

    # Evaluate the model with cross-validation
    print("Evaluating the model with cross-validation...")
    mse_scores, r2_scores = evaluate_model_with_cv(model, X, y)
    print(f"Cross-Validation Mean Squared Error: {mse_scores.mean()} \u00b1 {mse_scores.std()}")
    print(f"Cross-Validation R^2 Score: {r2_scores.mean()} \u00b1 {r2_scores.std()}")

    # Train final model on full data
    model.fit(X, y)

    # Save the model
    model_path = save_model(model)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()
