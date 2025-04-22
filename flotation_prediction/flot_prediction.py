import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from flot_visualization import plot_residuals

def evaluate_models_delta(df: pd.DataFrame,
                          features: list,
                          target: str = 'delta_silica') -> pd.DataFrame:
    """
    Train and evaluate regression models to predict the silica change.
    Computes MAE, RMSE, R², and MAPE.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing features and the delta target.
    features : list
        List of feature column names.
    target : str
        Name of the delta target column.

    Returns:
    --------
    pd.DataFrame
        Evaluation metrics for each model.
    """
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(
                n_estimators=100, random_state=42, objective='reg:squarederror'
            ))
        ])
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        results.append({
            'model': name,
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': mse ** 0.5,
            'R2': r2_score(y_test, preds),
            'MAPE (%)': mean_absolute_percentage_error(y_test, preds) * 100
        })

    return pd.DataFrame(results)

def evaluate_models(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    """
    Train and evaluate regression models (Linear, RF, XGBoost) to predict the target.
    Computes MAE, RMSE (via sqrt of MSE to avoid FutureWarning), R², and MAPE.
    """
    X = df[features]
    y = df[target]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )
    
    models = {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=1000, max_depth=3))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(
                n_estimators=1000,
                objective='reg:squarederror',
            ))
        ])
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        results.append({
            'model': name,
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': np.sqrt(mse),
            'R2': r2_score(y_test, preds),
            'MAPE (%)': mean_absolute_percentage_error(y_test, preds) * 100
        })
    
    return pd.DataFrame(results)

def evaluate_models_delta_residuals(df: pd.DataFrame,
                          features: list,
                          target: str = 'delta_silica') -> pd.DataFrame:
    """
    Train, evaluate, and plot residuals for regression models predicting silica change.
    """
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(
                n_estimators=100, random_state=42, objective='reg:squarederror'
            ))
        ])
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Plot residuals for this model
        plot_residuals(y_test, preds,
                       actual_name=target,
                       pred_name=f'pred_{name}')

        mse = mean_squared_error(y_test, preds)
        results.append({
            'model': name,
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': np.sqrt(mse),
            'R2': r2_score(y_test, preds),
            'MAPE (%)': mean_absolute_percentage_error(y_test, preds) * 100
        })

    return pd.DataFrame(results)


def evaluate_models_residuals(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    """
    Train, evaluate, and plot residuals for regression models predicting the target.
    """
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    models = {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=1000, max_depth=3))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(
                n_estimators=1000,
                objective='reg:squarederror',
            ))
        ])
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Plot residuals for this model
        plot_residuals(y_test, preds,
                       actual_name=target,
                       pred_name=f'pred_{name}')

        mse = mean_squared_error(y_test, preds)
        results.append({
            'model': name,
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': np.sqrt(mse),
            'R2': r2_score(y_test, preds),
            'MAPE (%)': mean_absolute_percentage_error(y_test, preds) * 100
        })

    return pd.DataFrame(results)

# Example usage:
# results_df = evaluate_models(df_clean, features, 'conc_silica')
# tools.display_dataframe_to_user('Model evaluation results with XGBoost and MAPE (updated RMSE)', results_df)

