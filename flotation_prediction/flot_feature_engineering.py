import pandas as pd

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features: hour of day and day of week.
    """
    df = df.copy()
    df['hour'] = df['inicio'].dt.hour
    df['dayofweek'] = df['inicio'].dt.dayofweek
    return df

def add_ph_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pH difference feature between the two lines.
    """
    df = df.copy()
    df['ph_diff'] = df['ph_flotacao_linha01'] - df['ph_flotacao_linha02']
    return df

def add_dosage_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dosage-to-flow ratio feature.
    """
    df = df.copy()
    df['dosage_flow_ratio'] = (
        df['dosagem_amina_conc_magnetica'] / df['vazao_alimentacao_flotacao']
    )
    return df

def add_cell_level_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cell-level statistics: max, min, and range across sensors.
    """
    df = df.copy()
    level_cols = [
        'nivel_celula_li640101',
        'nivel_celula_li643101',
        'nivel_celula_li643201'
    ]
    df['cell_level_max'] = df[level_cols].max(axis=1)
    df['cell_level_min'] = df[level_cols].min(axis=1)
    df['cell_level_range'] = df['cell_level_max'] - df['cell_level_min']
    return df

def add_rolling_features(df: pd.DataFrame,
                         roll_cols: list = None,
                         windows: tuple = (3, 5)) -> pd.DataFrame:
    """
    Add lagged rolling mean features for specified columns and window sizes.
    """
    df = df.copy()
    if roll_cols is None:
        roll_cols = [
            'dosagem_amina_conc_magnetica',
            'vazao_alimentacao_flotacao',
            'densidade_alimentacao_flotacao',
            'param_dosagem_amido'
        ]
    for window in windows:
        for col in roll_cols:
            df[f'{col}_rollmean_{window}'] = (
                df[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .shift(1)
            )
    return df

def add_delta_silica_regular(df: pd.DataFrame,
                             source_col: str = 'conc_silica',
                             delta_col: str = 'delta_silica',
                             time_col: str = 'inicio',
                             expected_interval: pd.Timedelta = pd.Timedelta(hours=2)) -> pd.DataFrame:
    """
    Compute the change in silica concentration only when the time
    difference from the previous record equals the expected interval.
    Rows where this condition is not met (or where delta is undefined)
    are dropped.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned DataFrame sorted by time.
    source_col : str
        Name of the original silica concentration column.
    delta_col : str
        Name of the new delta column.
    time_col : str
        Name of the datetime column to check regularity.
    expected_interval : pd.Timedelta
        The expected time difference between consecutive samples.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with a new column delta_col and only rows where
        the previous timestamp is exactly expected_interval before.
    """
    df2 = df.copy().sort_values(time_col).reset_index(drop=True)
    # Compute time difference from previous record
    df2['time_diff'] = df2[time_col].diff()
    # Compute silica difference
    df2[delta_col] = df2[source_col].diff()
    # Keep rows only where time_diff == expected_interval
    df2 = df2[df2['time_diff'] == expected_interval]
    # Drop helper column
    return df2.drop(columns=['time_diff']).reset_index(drop=True)


# Usage example:
# df1 = add_time_features(df_clean)
# df2 = add_ph_features(df1)
# df3 = add_dosage_flow_features(df2)
# df4 = add_cell_level_stats(df3)
# df_feat = add_rolling_features(df4)

