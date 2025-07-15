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


def split_features_target(df: pd.DataFrame,
                          target_col: str,
                          test_size: float = 0.2,
                          random_state: int = 42):
    """
    Separa features e target, removendo colunas não numéricas.
    """
    # Alvo
    y = df[target_col]
    # Features: apenas numéricas
    X = df.select_dtypes(include=[np.number, 'bool']).drop(columns=[target_col])
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def pinball_loss(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 quantile: float) -> float:
    """
    Calcula a Quantile Loss (Pinball loss) para um quantil específico.
    """
    diff = y_true - y_pred
    return np.mean(np.maximum(quantile * diff, (quantile - 1) * diff))


def train_lightgbm_quantile(X_train,
                             y_train,
                             X_val=None,
                             y_val=None,
                             quantile: float = 0.9,
                             sample_weight=None,
                             val_weight=None,
                             num_boost_round: int = 1000,
                             early_stopping_rounds: int = 50) -> lgb.Booster:
    """
    Treina um modelo LightGBM usando o objetivo de quantile (pinball loss).
    """
    params = {
        'objective': 'quantile',
        'alpha': quantile,
        'metric': 'quantile',
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


def evaluate_model_quantile(model: lgb.Booster,
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             quantile: float = 0.9) -> dict:
    """
    Avalia o modelo de quantile retornando pinball loss e métricas complementares.
    """
    preds = model.predict(X_test)
    return {
        'pinball_loss': pinball_loss(y_test.values, preds, quantile),
        'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'mape': mean_absolute_percentage_error(y_test, preds),
        'r2': r2_score(y_test, preds)
    }


def plot_predicted_vs_observed(y_test: np.ndarray,
                               preds: np.ndarray):
    """
    Plota valores observados e preditos em índice de observação.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test, name='Observed'))
    fig.add_trace(go.Scatter(y=preds, name='Predicted'))
    fig.update_layout(title='Quantile Regression (Pinball) Predicted vs Observed')
    fig.show()


if __name__ == '__main__':
    # Carrega e prepara dados
    df = load_data(r'C:\Users\rcpsi\OneDrive\Documents\langchain\Flotation\flotation_prediction\working_df_15_05_2025.csv')
    df = sanitize_dataframe(df)

    # Split train/test
    X_train, X_test, y_train, y_test = split_features_target(df, 'conc_silica')
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Treina quantile model (p.ex. 90% quantile)
    q = 0.9
    model = train_lightgbm_quantile(
        X_tr, y_tr,
        X_val=X_val, y_val=y_val,
        quantile=q
    )

    # Avalia
    results = evaluate_model_quantile(model, X_test, y_test, quantile=q)
    print(f"Pinball Loss (Q{int(q*100)}): {results['pinball_loss']:.4f}")
    print(f"Test RMSE: {results['rmse']:.4f}")
    print(f"Test MAPE: {results['mape']:.4f}")
    print(f"Test R²: {results['r2']:.4f}")

    # Plota predições
    preds = model.predict(X_test)
    plot_predicted_vs_observed(y_test.values, preds)
