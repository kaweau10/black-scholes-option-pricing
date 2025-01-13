from src.models.random_forest import (
    load_combined_data,
    preprocess_and_split_data,
    train_random_forest,
    evaluate_model,
    save_model,
)

def main():
    # Load combined processed data
    print("Loading combined processed data...")
    data = load_combined_data()

    # Define features and target
    features = ['log_moneyness', 'log_time_to_maturity', 'boxcox_iv_proxy']
    target = 'boxcox_lastPrice'

    # Split data into training and testing sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = preprocess_and_split_data(data, features, target)

    # Train Random Forest model
    print("Training Random Forest model...")
    model = train_random_forest(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Random Forest Mean Squared Error: {mse}")
    print(f"Random Forest R2 Score: {r2}")

    # Save the model
    print("Saving the model...")
    model_path = save_model(model)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()
