#!/usr/bin/env python3

# missing_data.py
# simulates missing data using a Mask

import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from functions import load_data, prepare_features, evaluate_model, ampute_mcar_df

def run_knn_missing_data(X_train, y_train, X_test, y_test):
    for p in [0.2, 0.4, 0.6, 0.8]:
        X_missing_df = ampute_mcar_df(X_train, missing_rate=p)
        model = KNeighborsRegressor()
        evaluate_model(model.fit(X_missing_df, y_train), X_missing_df, y_train, X_test, y_test)


if __name__ == "__main__":
    start = time.time()

    df = load_data("regressors_data.csv")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    run_knn_missing_data(X_train, y_train, X_test, y_test)

    end = time.time()
    print(f"\nRuntime: {end - start:.4f} seconds")
    print("\nEND.")