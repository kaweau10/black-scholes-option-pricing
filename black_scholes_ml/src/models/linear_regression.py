import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

MODEL_DIR = os.path.join(os.getcwd(), "black_scholes_ml", "models")

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

def preprocess_and_split_data(data):
    """Prepare data for linear regression."""
    # Select features and target
    X = data[['log_moneyness', 'log_time_to_maturity', 'boxcox_iv_proxy']]  # Add relevant features here
    y = data['boxcox_lastPrice']  # Target variable

    # Train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_linear_regression(X_train, y_train):
    """Train the linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

def save_model(model, model_name="linear_regression", root_dir="black_scholes_ml"):
    """Save the trained model to a dedicated folder in the models directory."""
    model_dir = os.path.join(root_dir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    return model_path
