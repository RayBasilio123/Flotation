import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from flot_visualization import plot_residuals


def evaluate_models_weights(df: pd.DataFrame,
                            features: list,
                            target: str,
                            show_residuals: bool = False) -> pd.DataFrame:
    """
    Train, evaluate (with sample weights), and optionally plot residuals for regression models.

    Uses squared weights = (y / mean(y))**2 to emphasize larger silica values.
    """
    X = df[features]
    y = df[target]
    weights = (y / y.mean())**2

    # split X, y, and weights together
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    models = {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=1000, max_depth=3, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(n_estimators=1000, objective='reg:squarederror', random_state=42))
        ])
    }

    results = []
    for name, pipeline in models.items():
        # determine final estimator step name
        final_step = list(pipeline.named_steps.keys())[-1]
        # fit with weights passed to the final estimator
        pipeline.fit(
            X_train, y_train,
            **{f"{final_step}__sample_weight": w_train}
        )
        preds = pipeline.predict(X_test)

        if show_residuals:
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



def evaluate_models(df: pd.DataFrame, features: list, target: str, show_residuals: bool = False) -> pd.DataFrame:
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
        if show_residuals:
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

def evaluate_models_quantile(df: pd.DataFrame,
                             features: list,
                             target: str,
                             show_residuals: bool = False,
                             quantile: float = 0.5) -> pd.DataFrame:
    """
    Train and evaluate regression models (Linear, RF, XGBoost, QuantileGBR)
    without sample weights, using quantile loss for the GradientBoostingRegressor.
    Computes MAE, RMSE, R², and MAPE.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing features and the target.
    features : list
        List of feature column names.
    target : str
        Name of the target column.
    plot_residuals : bool
        Whether to plot residual diagnostics for each model.
    quantile : float
        Quantile level for the quantile regressor (alpha).

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
            ('rf', RandomForestRegressor(n_estimators=1000, max_depth=3))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(n_estimators=1000, objective='reg:squarederror'))
        ]),
        f'QuantileGBR_{quantile}': Pipeline([
            ('scaler', StandardScaler()),
            ('gbr', GradientBoostingRegressor(
                loss='quantile',
                alpha=quantile,
                n_estimators=100
            ))
        ])
    }

    results = []
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        if show_residuals:
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

