import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

# Directories for saving models and reports
MODEL_DIR = os.path.join(os.getcwd(), "models", "neural_network")
EVALUATION_DIR = os.path.join(os.getcwd(), "reports", "model_evaluations", "temp")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

def load_combined_data():
    """Load the combined processed data."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    file_path = os.path.join(root_dir, "data", "processed", "combined_processed_data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The combined data file does not exist at {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(data, features, target):
    """Prepare and scale data for the neural network."""
    X = data[features]
    y = data[target]

    # Drop rows where target (y) is NaN **before filling X**
    data_cleaned = data.dropna(subset=[target])

    X_cleaned = data_cleaned[features]
    y_cleaned = data_cleaned[target]

    # Fill missing values in X with the mean
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_cleaned)
    joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))  # Save imputer

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))  # Save scaler

    return X_scaled, y_cleaned


def build_model(input_dim):
    """Define the Neural Network architecture."""
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)  # Regression output
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def main():
    # Load and preprocess data
    data = load_combined_data()
    features = [
        "moneyness", "time_to_maturity", "iv_proxy", "historical_volatility",
        "log_iv_proxy", "moneyness_bin", "moneyness_time", "moneyness_squared"
    ]
    target = "lastPrice"
    
    if not all(feature in data.columns for feature in features):
        raise ValueError("One or more features are missing in the data. Check preprocessing.")
    
    X, y = preprocess_data(data, features, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = build_model(input_dim=X.shape[1])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Training the Neural Network model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save model and evaluation
    model.save(os.path.join(MODEL_DIR, "model.h5"))
    with open(os.path.join(EVALUATION_DIR, "neural_network_report.txt"), "w") as file:
        file.write(f"Mean Squared Error: {mse}\n")
        file.write(f"R^2 Score: {r2}\n")
    
    print(f"Model saved at {MODEL_DIR}/model.h5")
    print(f"Evaluation saved at {EVALUATION_DIR}/neural_network_report.txt")

if __name__ == "__main__":
    main()
