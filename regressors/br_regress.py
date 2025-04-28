#!/usr/bin/env python3

# br_regress.py
# Bayesian Ridge Regressor

import time
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection    import train_test_split, RandomizedSearchCV, GridSearchCV

from functions import load_data, prepare_features, evaluate_model


def run_bayesian_ridge(X_train, y_train, X_test, y_test):
    param_grid = {
        'max_iter':  [100, 200, 300, 400, 500],
        'lambda_1':  [1, 1e-6, 1e-2, 1e-4],
        'alpha_1' :  [1, 1e-6, 1e-2, 1e-4]
    }
    grid = GridSearchCV(
        BayesianRidge(),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        return_train_score=True,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print("Best Bayesian Ridge params:", grid.best_params_)
    best = grid.best_estimator_
    evaluate_model(best, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    start = time.time()
    df = load_data("regressors_data.csv")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    run_bayesian_ridge(X_train, y_train, X_test, y_test)

    end = time.time()
    print(f"\nRuntime: {end - start:.4f} seconds")
    print("\nEND.")



