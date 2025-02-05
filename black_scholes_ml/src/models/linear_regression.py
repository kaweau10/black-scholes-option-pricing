import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.impute import SimpleImputer
import joblib

MODEL_DIR = os.path.join(os.getcwd(), "models")
EVALUATION_DIR = os.path.join(os.getcwd(), "reports", "model_evaluations")

def load_combined_data():
    """Load the combined processed data."""
    # Calculate the root directory relative to main.py
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Construct the path to the combined data file
    file_path = os.path.join(root_dir, "data", "processed", "combined_processed_data.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The combined data file does not exist at {file_path}")
    
    print("Loading data from file...")
    return pd.read_csv(file_path)

def preprocess_data(data, features, target):
    """Prepare data for linear regression."""
    # Select features and target
    X = data[features]
    y = data[target]
    return X, y

def evaluate_model_with_cv(model, X, y):
    """Evaluate the model using cross-validation."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=cv)
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    file_path = os.path.join(EVALUATION_DIR, "linear_regression_report.txt")
    with open(file_path, "w") as file:
        file.write(f"Cross-Validation Mean Squared Error: {mse_scores.mean()} \u00b1 {mse_scores.std()}\n")
        file.write(f"Cross-Validation R^2 Score: {r2_scores.mean()} \u00b1 {r2_scores.std()}\n")
    print(f"Saved values to {file_path}")
    return mse_scores, r2_scores

def save_model(model, model_name="linear_regression"):
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

    # Preprocess data.  Handle missing values in 'X' and drop rows where 'y' is NaN
    X, y = preprocess_data(data, features, target)
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X = X[y.notna()]
    y = y.dropna()

    # Train the model
    print("Training the Linear Regression model...")
    model = LinearRegression()

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
