#!/usr/bin/env python3

# svm_regress.py
# Support Vector Machine Regressor

import time
from sklearn.svm import SVR
from functions import load_data, prepare_features, evaluate_model
from sklearn.model_selection import train_test_split


def run_support_vector_machine(X_train, y_train, X_test, y_test):
    for k in ['poly', 'linear']:
        model = SVR(
            kernel = k,
            C = 10,
            gamma = 'scale')
        model.fit(X_train, y_train)
        evaluate_model(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    start = time.time()

    df = load_data("regressors_data.csv")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    run_support_vector_machine(X_train, y_train, X_test, y_test)

    end = time.time()
    print(f"\nRuntime: {end - start:.4f} seconds")
    print("\nEND.")