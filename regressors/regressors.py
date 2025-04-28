#!/usr/bin/env python3

# BE WARNED:
# Runtime: 1591.4072 seconds

# rf_regressor.py
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot

from sklearn.model_selection    import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.neighbors          import KNeighborsRegressor
from sklearn.ensemble           import RandomForestRegressor
from sklearn.tree               import DecisionTreeRegressor
from sklearn.linear_model       import BayesianRidge
from sklearn.neural_network     import MLPRegressor
from sklearn.svm                import SVR
from sklearn.metrics            import mean_absolute_error, mean_squared_error, r2_score

# This is timer system:
# start = time.time()
# ...code goes here...
# end = time.time()
# print(f"Runtime: {end - start:.4f} seconds")

start = time.time()

def load_data(path):
    df = pd.read_csv(path)
    print("Dataset sample:")
    print(df.sample(5))
    return df

def prepare_features(df):
    # assume col 0 is ID, cols 1–21 are features, col 22 is the target
    X = df.iloc[:, 1:22].astype(float)
    y = df.iloc[:, 22].astype(float)
    return X, y


# new helper function:
def evaluate_model(model, X_train, y_train, X_test, y_test):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    print(f"{model.__class__.__name__} → MAE: {mae:.3f}, MSE: {mse:.3f}, R²: {r2:.3f}")


##################################
# K-Nearest Neighbors (KNN) Model
##################################

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


##################################
# Random Forest Regressor
##################################

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


#################################
# Bayesian Ridge
#################################

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



##################################
# Decision Tree (DT) Model
##################################

def run_decision_tree(X_train, y_train, X_test, y_test):
    param_grid = {
        'random_state':[0, 1, 2, 3, 4, 5],
        'min_samples_leaf':[1, 6, 12, 18],
        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
    }
    grid = GridSearchCV(
        DecisionTreeRegressor(),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        return_train_score=True,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print(">> Best Decision Tree params:", grid.best_params_)
    best = grid.best_estimator_
    evaluate_model(best, X_train, y_train, X_test, y_test)


##################################
# Neural Network (NN) Model
##################################

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


##################################
# Support Vector Machine (SVM) Model
##################################

def run_support_vector_machine(X_train, y_train, X_test, y_test):
    for k in ['poly', 'linear']:
        model = SVR(
            kernel = k,
            C = 10,
            gamma = 'scale')
        model.fit(X_train, y_train)
        evaluate_model(model, X_train, y_train, X_test, y_test)


##################################
# Simulate Missing Data
##################################

# Generate missing data.
# pattern: ampute training data, convert to 0
# Removes data at random and marks them NaN
# def ampute_mcar(X_complete, missing_rate=0.1):
#     # mcar: Mask Completely At Random
#     M = np.random.binomial(1, missing_rate, size = X_complete.shape)
#     X_obs = X_complete.copy()
#     np.putmask(X_obs, M, np.nan)
#     print('Percentage of newly generated missing values: {}'.\
#       format(np.round(np.sum(np.isnan(X_obs))/X_obs.size,3)))

#     # warning if a full row is missing
#     for row in X_obs:
#         if np.all(np.isnan(row)):
#             warnings.warn('Some row(s) contains only nan values.')
#             break

#     # warning if a full col is missing
#     for col in X_obs.T:
#         if np.all(np.isnan(col)):
#             warnings.warn('Some col(s) contains only nan values.')
#             break

#     X_obs[ np.isnan(X_obs) ] = 0

#     return X_obs

def ampute_mcar_df(df, missing_rate=0.1):
    # create a boolean mask DataFrame. We had this using np function as shown above but it gave type error: TypeError: putmask: first argument must be an array. Since we are working in dataframes primarily, we use a mask function from Pandas, see df.mask(M) below.
    M = pd.DataFrame(
        np.random.binomial(1, missing_rate, size=df.shape).astype(bool),
        index=df.index,
        columns=df.columns
    )
    # 2) apply the mask, turning masked cells into NaN
    df_obs = df.mask(M)

    # report
    pct = (df_obs.isna().sum().sum() / df_obs.size).round(3)
    print(f"Percentage of newly generated missing values: {pct}")

    # 3) warnings
    if any(df_obs.isna().all(axis=1)): warnings.warn("Some row(s) only NaN")
    if any(df_obs.isna().all(axis=0)): warnings.warn("Some col(s) only NaN")

    # fill NaNs with zero
    df_filled = df_obs.fillna(0)
    return df_filled



###################################################
# RFR with missing values 0.20, 0.40, 0.60, 0.80
###################################################

def run_rfr_missing_data(X_train, y_train, X_test, y_test):
    for p in [0.2, 0.4, 0.6, 0.8]:
        X_missing_df = ampute_mcar_df(X_train, missing_rate=p)
        model = RandomForestRegressor(n_estimators=1000)
        evaluate_model(model.fit(X_missing_df, y_train), X_missing_df, y_train, X_test, y_test)


###################################################
# KNN with missing values 0.20. 0.40, 0.60, 0.80
###################################################

def run_knn_missing_data(X_train, y_train, X_test, y_test):
    for p in [0.2, 0.4, 0.6, 0.8]:
        X_missing_df = ampute_mcar_df(X_train, missing_rate=p)
        model = KNeighborsRegressor()
        evaluate_model(model.fit(X_missing_df, y_train), X_missing_df, y_train, X_test, y_test)


# Main guard and function calls
if __name__ == "__main__":
    df = load_data("regressors_data.csv")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Run each section in turn:
    run_knn(X_train, y_train, X_test, y_test)
    run_random_forest(X_train, y_train, X_test, y_test)
    run_bayesian_ridge(X_train, y_train, X_test, y_test)
    run_decision_tree(X_train, y_train, X_test, y_test)
    run_multi_layer_perceptron(X_train, y_train, X_test, y_test)
    run_support_vector_machine(X_train, y_train, X_test, y_test)
    run_rfr_missing_data(X_train, y_train, X_test, y_test)
    run_knn_missing_data(X_train, y_train, X_test, y_test)

    end = time.time()
    print(f"\nRuntime: {end - start:.4f} seconds")

    print("\nEND.")
    



