#!/usr/bin/env python3

# nn_regress.py
# Neural Network Regressor


import time
from sklearn.neural_network     import MLPRegressor
from sklearn.model_selection    import train_test_split, RandomizedSearchCV, GridSearchCV

from functions import load_data, prepare_features, evaluate_model


def run_multi_layer_perceptron(X_train, y_train, X_test, y_test):
    for l in [50, 100]:
        model = MLPRegressor(
                max_iter=1500, 
                activation='relu', 
                solver='adam', 
                learning_rate_init=0.001,
                hidden_layer_sizes=l)
        model.fit(X_train, y_train)
        evaluate_model(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    start = time.time()

    df = load_data("regressors_data.csv")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    run_multi_layer_perceptron(X_train, y_train, X_test, y_test)

    end = time.time()
    print(f"\nRuntime: {end - start:.4f} seconds")
    print("\nEND.")