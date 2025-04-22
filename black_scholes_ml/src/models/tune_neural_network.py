import os
import pandas as pd
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load data
def load_combined_data():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    file_path = os.path.join(root_dir, "data", "processed", "combined_processed_data.csv")
    return pd.read_csv(file_path)

# Preprocess data
def preprocess_data(data, features, target):
    data_cleaned = data.dropna(subset=[target])
    X = data_cleaned[features]
    y = data_cleaned[target]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y

# Build model with hyperparameters
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=256, step=32),
                    activation=hp.Choice('activation_1', ['relu', 'tanh']),
                    input_shape=(8,),
                    kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2', 1e-4, 1e-2, sampling='LOG'))))
    
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Dense(units=hp.Int('units_2', min_value=16, max_value=128, step=16),
                    activation=hp.Choice('activation_2', ['relu', 'tanh'])))
    
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]

    )
    return model

def main():
    # Load and preprocess
    data = load_combined_data()
    features = [
        "moneyness", "time_to_maturity", "iv_proxy", "historical_volatility",
        "log_iv_proxy", "moneyness_bin", "moneyness_time", "moneyness_squared"
    ]
    target = "lastPrice"

    X, y = preprocess_data(data, features, target)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tuner
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=15,
        executions_per_trial=1,
        directory="models/neural_network_tuning",
        project_name="black_scholes_tuning"
    )

    tuner.search(X_train, y_train,
                 epochs=100,
                 validation_data=(X_val, y_val),
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save("models/neural_network/best_tuned_model.h5")

    print("Best hyperparameters:", tuner.get_best_hyperparameters()[0].values)

if __name__ == "__main__":
    main()
