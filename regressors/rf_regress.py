#!/usr/bin/env python3

# rf_regress.py
# Random Forest Regressor


import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from functions import load_data, prepare_features, evaluate_model


def run_random_forest(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [100, 1000, 10000],
        'max_features': [None, 'sqrt', 'log2'],
        'max_depth'   : [4, 6]
    }
    grid = GridSearchCV(
        RandomForestRegressor(),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        return_train_score=True,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print(">> Best RF params:", grid.best_params_)
    best = grid.best_estimator_
    evaluate_model(best, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    start = time.time()

    df = load_data("regressors_data.csv")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    run_random_forest(X_train, y_train, X_test, y_test)

    end = time.time()
    print(f"\nRuntime: {end - start:.4f} seconds")
    print("\nEND.")