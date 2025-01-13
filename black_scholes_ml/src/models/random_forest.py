import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_random_forest(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

def save_model(model, model_name="random_forest"):
    """Save the trained model."""
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    return model_path
