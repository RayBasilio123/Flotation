import pandas as pd
import numpy as np
import lightgbm as lgb
import re
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega o dataset a partir de um arquivo CSV.
    """
    return pd.read_csv(filepath)


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas não numéricas (exceto 'conc_silica' e 'inicio') e limpa nomes de colunas
    para evitar caracteres inválidos no LightGBM.
    """
    non_numeric = df.select_dtypes(exclude=[np.number, 'bool']).columns.tolist()
    keep_cols = {'conc_silica', 'inicio'}
    drop_cols = [c for c in non_numeric if c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df.columns = [re.sub(r'[^0-9A-Za-z_]', '_', col) for col in df.columns]
    return df


def split_features_target(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Separa features numéricas (descartando timestamps) e alvo, e faz split.

    Columns of type object (e.g., 'inicio') are automaticamente excluídas.
    """
    # Alvo
    y = df[target_col]
    # Seleciona apenas colunas numéricas, descartando o target
    X = df.select_dtypes(include=[np.number, 'bool']).drop(columns=[target_col])
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_threshold(y: pd.Series, quantile: float = 0.9) -> float:
    return y.quantile(quantile)


def compute_sample_weight(y: pd.Series, threshold: float, weight_high: float = 5.0, weight_low: float = 1.0) -> np.ndarray:
    return np.where(y > threshold, weight_high, weight_low)


def train_lightgbm(X_train, y_train, sample_weight=None, params=None, num_boost_round: int = 1000,
                   early_stopping_rounds: int = 50, X_val=None, y_val=None, val_weight=None):
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt'
        }
    train_set = lgb.Dataset(X_train, y_train, weight=sample_weight)
    valid_sets = [train_set]
    valid_names = ['train']
    callbacks = [lgb.log_evaluation(period=100)]
    if X_val is not None and y_val is not None:
        val_set = lgb.Dataset(X_val, y_val, weight=val_weight, reference=train_set)
        valid_sets.append(val_set)
        valid_names.append('valid')
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks
    )
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'mape': mean_absolute_percentage_error(y_test, preds),
        'r2': r2_score(y_test, preds)
    }


def plot_time_series_with_gaps(df: pd.DataFrame,
                               time_col: str = 'inicio',
                               value_col: str = 'conc_silica',
                               gap_threshold: float = None):
    df_plot = df.copy()
    df_plot[time_col] = pd.to_datetime(df_plot[time_col])
    df_plot = df_plot.sort_values(time_col).reset_index(drop=True)
    dt = df_plot[time_col].diff().dt.total_seconds()
    if gap_threshold is None:
        gap_threshold = dt.median() * 2
    segments = (dt > gap_threshold).cumsum()
    fig = go.Figure()
    for seg_id, segment_df in df_plot.assign(segment=segments).groupby('segment'):
        fig.add_trace(go.Scatter(
            x=segment_df[time_col],
            y=segment_df[value_col],
            mode='lines',
            showlegend=(seg_id == 0)
        ))
    fig.update_layout(
        title=f'{value_col} Over Time (gaps shown)',
        xaxis_title=time_col,
        yaxis_title=value_col
    )
    fig.show()


def plot_predicted_vs_observed(y_test: np.ndarray,
                               preds: np.ndarray,
                               ):
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            y=y_test,
            name='Observed',
        ))
    fig.add_trace(go.Scatter(
            y=preds,
            name='Predicted',
        ))
    fig.update_layout(
        title=f'Predicted vs Observed Silica % Over Time',
    )
    fig.show()

if __name__ == '__main__':
    df = load_data(r'C:\Users\rcpsi\OneDrive\Documents\langchain\Flotation\flotation_prediction\working_df_15_05_2025.csv')
    df = sanitize_dataframe(df)
    df = df.drop(columns=['Unnamed__0', 'inicio'])
    print(df.columns)
    X_train, X_test, y_train, y_test = split_features_target(df, 'conc_silica')
    threshold = get_threshold(y_train)
    weights = compute_sample_weight(y_train, threshold)
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, weights, test_size=0.2, random_state=42
    )
    params = {'objective':'quantile','metric':'rmse','verbosity':-1,'boosting_type':'gbdt', 'alpha':0.95}
    model = train_lightgbm(X_tr, y_tr, sample_weight=w_tr, params=params,
                           num_boost_round=1000, early_stopping_rounds=50,
                           X_val=X_val, y_val=y_val, val_weight=w_val)
    results = evaluate_model(model, X_test, y_test)
    print(f"Test RMSE: {results['rmse']:.4f}")
    print(f"Test MAPE: {results['mape']:.4f}")
    print(f"Test R²: {results['r2']:.4f}")
    preds = model.predict(X_test)
    plot_predicted_vs_observed(y_test, preds)
