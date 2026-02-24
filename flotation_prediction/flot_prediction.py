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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# --- assume these two are already defined ---
# add_lag_silica_features(df, time_col='inicio', source_col='conc_silica', lags=[2,4,6])
# add_silica_quantile(df, source_col='conc_silica', new_col='conc_silica_quantile')

def two_stage_quantile_then_conc_percentiles(
    df: pd.DataFrame,
    base_features: list,
    source_col: str = 'conc_silica',
    test_size: float = 0.2,
    random_state: int = None,
    show_residuals: bool = False
):
    """
    Two‐stage pipeline using percentiles (1% bins) instead of deciles:
      1) Split train/test
      2) Compute continuous quantile on train
      3) Map test values into 100 percentile bins based on train cutoffs,
         assigning values < min → 0.0, > max → 1.0
      4) Stage 1: predict percentile via RF, report MAE, MAPE, extremes count
      5) Stage 2: predict conc_silica using base_features + pred_percentile, report MAE & MAPE
    """
    # 1) Prepare data and split
    df2 = df.dropna(subset=base_features + [source_col]).reset_index(drop=True)
    train_idx, test_idx = train_test_split(df2.index, test_size=test_size, random_state=random_state)
    train = df2.loc[train_idx].copy()
    test  = df2.loc[test_idx].copy()

    # 2) Compute continuous quantile on train
    train['conc_silica_quantile'] = train[source_col].rank(pct=True)

    # 3) Percentile cutpoints (0%,1%,2%,...,100%)
    cutoffs = train[source_col].quantile(np.linspace(0, 1, 101)).values
    labels  = np.linspace(0.01, 1.00, 100)  # percentiles 1%..100%
    test['conc_silica_quantile'] = pd.cut(
        test[source_col],
        bins=cutoffs,
        labels=labels,
        include_lowest=True
    ).astype(float)
    # assign extremes
    test.loc[test[source_col] < cutoffs[0], 'conc_silica_quantile'] = 0.0
    test.loc[test[source_col] > cutoffs[-1], 'conc_silica_quantile'] = 1.0

    # --- Stage 1: Percentile model ---
    quant_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=4, random_state=random_state))
    ])
    quant_pipe.fit(train[base_features], train['conc_silica_quantile'])
    yq_pred = quant_pipe.predict(test[base_features])

    # metrics
    q_mae = mean_absolute_error(test['conc_silica_quantile'], yq_pred)
    q_mape = mean_absolute_percentage_error(test['conc_silica_quantile'], yq_pred) * 100
    extreme_count = ((test['conc_silica_quantile'] == 0.0) | (test['conc_silica_quantile'] == 1.0)).sum()
    total_count = len(test)
    print(f"Stage 1 (Percentile) → MAE: {q_mae:.4f}, MAPE: {q_mape:.2f}%, extremes: {extreme_count}/{total_count}")

    # append predicted percentile
    train['pred_percentile'] = quant_pipe.predict(train[base_features])
    test['pred_percentile']  = yq_pred

    # --- Stage 2: Silica model ---
    final_feats = base_features + ['pred_percentile']
    final_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=4, random_state=random_state))
    ])
    final_pipe.fit(train[final_feats], train[source_col])
    yc_pred = final_pipe.predict(test[final_feats])

    c_mae = mean_absolute_error(test[source_col], yc_pred)
    c_mape = mean_absolute_percentage_error(test[source_col], yc_pred) * 100
    print(f"Stage 2 (Silica) → MAE: {c_mae:.4f}, MAPE: {c_mape:.2f}%")

    if show_residuals:
        plot_residuals(test[source_col], yc_pred,
                           actual_name=source_col,
                           pred_name=f'pred')

    return {
        'percentile_model': quant_pipe,
        'final_model':      final_pipe,
        'test_df':          test,
        'percentile_metrics': {'MAE': q_mae, 'MAPE': q_mape, 'extremes': extreme_count},
        'silica_metrics':     {'MAE': c_mae,  'MAPE': c_mape}
    }


def two_stage_quantile_then_conc(
    df: pd.DataFrame,
    base_features: list,
    source_col: str = 'conc_silica',
    test_size: float = 0.2,
    random_state: int = None
):
    """
    Two‐stage pipeline without lag features and without quantile leakage:
      1) Split train/test
      2) Compute true quantile on train only
      3) Map test values to train quantile bins (assign extremes to 0 or 1)
      4) Stage 1: fit quantile model, report MAE, MAPE, and count of extremes
      5) Stage 2: fit silica model using base + pred_quantile, report MAE & MAPE
    """
    # 1) Drop NA in required columns and split
    df2 = df.dropna(subset=base_features + [source_col]).reset_index(drop=True)
    train_idx, test_idx = train_test_split(df2.index, test_size=test_size, random_state=random_state, shuffle=True)
    train = df2.loc[train_idx].copy()
    test  = df2.loc[test_idx].copy()

    # 2) Compute train‐only continuous quantile (0–1)
    train['conc_silica_quantile'] = train[source_col].rank(pct=True)

    # 3) Derive decile cutpoints on train and map test into them
    cutoffs = train[source_col].quantile(np.linspace(0, 1, 11)).values
    labels  = np.linspace(0.1, 1.0, 10)
    test['conc_silica_quantile'] = pd.cut(
        test[source_col],
        bins=cutoffs,
        labels=labels,
        include_lowest=True
    ).astype(float)
    # Assign extremes
    test.loc[test[source_col] < cutoffs[0], 'conc_silica_quantile'] = 0.0
    test.loc[test[source_col] > cutoffs[-1], 'conc_silica_quantile'] = 1.0

    # --- Stage 1: Quantile model ---
    quant_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=random_state))
    ])
    quant_pipe.fit(train[base_features], train['conc_silica_quantile'])
    yq_pred = quant_pipe.predict(test[base_features])

    # Metrics
    q_mae  = mean_absolute_error(test['conc_silica_quantile'], yq_pred)
    q_mape = mean_absolute_percentage_error(test['conc_silica_quantile'], yq_pred) * 100
    # Count extremes
    extreme_mask  = (test['conc_silica_quantile'] == 0.0) | (test['conc_silica_quantile'] == 1.0)
    extreme_count = int(extreme_mask.sum())
    total_count   = len(test)
    print(
        f"Stage 1 (Quantile) → "
        f"MAE: {q_mae:.4f}, "
        f"MAPE: {q_mape:.2f}%, "
        f"extremes: {extreme_count}/{total_count}"
    )

    # 4) Append predicted quantile
    train['pred_quantile'] = quant_pipe.predict(train[base_features])
    test['pred_quantile']  = yq_pred

    # --- Stage 2: Silica model ---
    final_feats = base_features + ['pred_quantile']
    final_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=random_state))
    ])
    final_pipe.fit(train[final_feats], train[source_col])
    yc_pred = final_pipe.predict(test[final_feats])

    # Metrics
    c_mae  = mean_absolute_error(test[source_col], yc_pred)
    c_mape = mean_absolute_percentage_error(test[source_col], yc_pred) * 100
    print(f"Stage 2 (Silica) → MAE: {c_mae:.4f}, MAPE: {c_mape:.2f}%")

    return {
        'quantile_model': quant_pipe,
        'final_model':   final_pipe,
        'test_df':       test,
        'quantile_metrics': {'MAE': q_mae, 'MAPE': q_mape, 'extremes': extreme_count},
        'silica_metrics':   {'MAE': c_mae,  'MAPE': c_mape}
    }


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

