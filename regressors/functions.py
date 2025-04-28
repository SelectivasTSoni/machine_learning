# functions.py

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(path):
    df = pd.read_csv(path)
    print("Dataset sample:")
    print(df.sample(5))
    return df


def prepare_features(df):
    # Assume col 0 is ID, cols 1–21 are features, col 22 is the target
    X = df.iloc[:, 1:22].astype(float)
    y = df.iloc[:, 22].astype(float)
    return X, y

def evaluate_model(model, X_train, y_train, X_test, y_test):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    print(f"{model.__class__.__name__} >> MAE: {mae:.3f}, MSE: {mse:.3f}, r_2: {r2:.3f}")


def ampute_mcar_df(df, missing_rate=0.1):
    # Create a boolean mask DataFrame.
    M = pd.DataFrame(
        np.random.binomial(1, missing_rate, size=df.shape).astype(bool),
        index=df.index,
        columns=df.columns)

    # Apply the mask, turning masked cells into NaN
    df_obs = df.mask(M)

    # Report
    pct = (df_obs.isna().sum().sum() / df_obs.size).round(3)
    print(f"Percentage of newly generated missing values: {pct}")

    if any(df_obs.isna().all(axis=1)): warnings.warn("Some row(s) only NaN")
    if any(df_obs.isna().all(axis=0)): warnings.warn("Some col(s) only NaN")

    # Fill NaNs with zero
    df_filled = df_obs.fillna(0)

    return df_filled


