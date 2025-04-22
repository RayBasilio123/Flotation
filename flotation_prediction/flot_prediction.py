import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
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
from sklearn.linear_model import TweedieRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import statsmodels.api as sm


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



def evaluate_tweedie_models(df: pd.DataFrame,
                            features: list,
                            target: str,
                            power: float = 1.5,
                            test_size: float = 0.2,
                            show_residuals: bool = False,
                            random_state=None) -> pd.DataFrame:
    """
    Train and evaluate:
      - GLM Gamma (log link)
      - Sklearn TweedieRegressor
      - XGBoost with Tweedie objective
      - LightGBM with Tweedie objective

    Computes MAE, RMSE, R2, and MAPE.
    """
    X = df[features]
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size
    )

    results = []

    # 1) Statsmodels GLM Gamma family, log link
    X_sm = sm.add_constant(X_train)
    glm = sm.GLM(y_train, X_sm, family=sm.families.Gamma(link=sm.families.links.log()))
    glm_res = glm.fit()
    preds_glm = glm_res.predict(sm.add_constant(X_test))

    if show_residuals:
            plot_residuals(y_test, preds_glm,
                           actual_name=target,
                           pred_name=f'pred_GLM_Gamma_Log')
            
    mse = mean_squared_error(y_test, preds_glm)
    results.append({
        'model': 'GLM_Gamma_Log',
        'MAE': mean_absolute_error(y_test, preds_glm),
        'RMSE': np.sqrt(mse),
        'R2': r2_score(y_test, preds_glm),
        'MAPE (%)': mean_absolute_percentage_error(y_test, preds_glm) * 100
    })

    # Common pipeline for sklearn Tweedie
    tw_pipe = make_pipeline(
        StandardScaler(),
        TweedieRegressor(power=power, link='log')
    )
    tw_pipe.fit(X_train, y_train)
    preds_tw = tw_pipe.predict(X_test)

    if show_residuals:
            plot_residuals(y_test, preds_tw,
                           actual_name=target,
                           pred_name=f'pred_Tweedie_Sklearn')

    mse = mean_squared_error(y_test, preds_tw)
    results.append({
        'model': f'Sklearn_Tweedie(p={power})',
        'MAE': mean_absolute_error(y_test, preds_tw),
        'RMSE': np.sqrt(mse),
        'R2': r2_score(y_test, preds_tw),
        'MAPE (%)': mean_absolute_percentage_error(y_test, preds_tw) * 100
    })

    # XGBoost Tweedie
    xgb = XGBRegressor(
        objective='reg:tweedie',
        tweedie_variance_power=power,
        n_estimators=500,
        random_state=random_state,
        # optional: learning_rate=0.1, max_depth=6
    )
    xgb.fit(X_train, y_train)
    preds_xgb = xgb.predict(X_test)

    if show_residuals:
            plot_residuals(y_test, preds_xgb,
                           actual_name=target,
                           pred_name=f'pred_XGBoost_Tweedie')

    mse = mean_squared_error(y_test, preds_xgb)
    results.append({
        'model': f'XGB_Tweedie(p={power})',
        'MAE': mean_absolute_error(y_test, preds_xgb),
        'RMSE': np.sqrt(mse),
        'R2': r2_score(y_test, preds_xgb),
        'MAPE (%)': mean_absolute_percentage_error(y_test, preds_xgb) * 100
    })

    # LightGBM Tweedie
    lgbm = LGBMRegressor(
        objective='tweedie',
        tweedie_variance_power=power,
        n_estimators=500,
        random_state=random_state,
        # optional: learning_rate=0.1, num_leaves=31
    )
    lgbm.fit(X_train, y_train)
    preds_lgb = lgbm.predict(X_test)

    if show_residuals:
            plot_residuals(y_test, preds_xgb,
                           actual_name=target,
                           pred_name=f'pred_LightGBM_Tweedie')

    mse = mean_squared_error(y_test, preds_lgb)
    results.append({
        'model': f'LGBM_Tweedie(p={power})',
        'MAE': mean_absolute_error(y_test, preds_lgb),
        'RMSE': np.sqrt(mse),
        'R2': r2_score(y_test, preds_lgb),
        'MAPE (%)': mean_absolute_percentage_error(y_test, preds_lgb) * 100
    })

    return pd.DataFrame(results)

# Usage example:
# results_tweedie = evaluate_tweedie_models(df_clean, features, 'conc_silica', power=1.5)
# display(results_tweedie)


# Example usage:
# results_df = evaluate_models(df_clean, features, 'conc_silica')
# tools.display_dataframe_to_user('Model evaluation results with XGBoost and MAPE (updated RMSE)', results_df)

