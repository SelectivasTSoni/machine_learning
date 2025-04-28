#!/usr/bin/env python3

# knn_regress.py
# k-nearest neighbors regressor


import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from functions import load_data, prepare_features, evaluate_model


def run_knn(X_train, y_train, X_test, y_test):
    knn_param={
        'n_neighbors':[5,10,15,20,25,30,70,140],
        'weights':['uniform','distance'],
        'metric': ['euclidean','manhattan']
        }
    grid=RandomizedSearchCV(
            KNeighborsRegressor(), 
            knn_param, 
            verbose=1, 
            cv=5, 
            n_jobs=-1, 
            scoring='neg_mean_squared_error', 
            return_train_score=True)
    grid.fit(X_train, y_train)
    print(">> Best KNN params:", grid.best_params_)
    best = grid.best_estimator_
    evaluate_model(best, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    start = time.time()

    df = load_data("regressors_data.csv")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    run_knn(X_train, y_train, X_test, y_test)

    end = time.time()
    print(f"\nRuntime: {end - start:.4f} seconds")
    print("\nEND.")