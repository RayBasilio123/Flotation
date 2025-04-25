import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)


def add_silica_quantile(
    df: pd.DataFrame,
    source_col: str = 'conc_silica',
    new_col: str = 'conc_silica_quantile'
) -> pd.DataFrame:
    """
    Add a percentile-rank feature of conc_silica.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing source_col.
    source_col : str
        Name of the silica concentration column.
    new_col : str
        Name of the new quantile feature (0 to 1).

    Returns
    -------
    pd.DataFrame
        Copy of df with new_col added.
    """
    df2 = df.copy()
    df2[new_col] = df2[source_col].rank(method='average', pct=True)
    return df2


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract hour and weekday from the 'inicio' timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a datetime column 'inicio'.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'hour' and 'dayofweek' columns added.
    """
    df2 = df.copy()
    df2['hour'] = df2['inicio'].dt.hour
    df2['dayofweek'] = df2['inicio'].dt.dayofweek
    return df2


def add_ph_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute difference between two pH readings.

    Parameters
    ----------
    df : pd.DataFrame
        Input with 'ph_flotacao_linha01' and 'ph_flotacao_linha02'.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'ph_diff' added.
    """
    df2 = df.copy()
    df2['ph_diff'] = (
        df2['ph_flotacao_linha01'] - df2['ph_flotacao_linha02']
    )
    return df2


def add_dosage_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the ratio of amine dosage to feed flow.

    Parameters
    ----------
    df : pd.DataFrame
        Input with 'dosagem_amina_conc_magnetica' and 'vazao_alimentacao_flotacao'.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'dosage_flow_ratio' added.
    """
    df2 = df.copy()
    df2['dosage_flow_ratio'] = (
        df2['dosagem_amina_conc_magnetica']
        / df2['vazao_alimentacao_flotacao']
    )
    return df2


def add_cell_level_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive max, min, and range across cell-level sensors.

    Parameters
    ----------
    df : pd.DataFrame
        Input with level sensor columns.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'cell_level_max', 'cell_level_min', and 'cell_level_range'.
    """
    df2 = df.copy()
    level_cols = [
        'nivel_celula_li640101',
        'nivel_celula_li643101',
        'nivel_celula_li643201',
    ]
    df2['cell_level_max'] = df2[level_cols].max(axis=1)
    df2['cell_level_min'] = df2[level_cols].min(axis=1)
    df2['cell_level_range'] = (
        df2['cell_level_max'] - df2['cell_level_min']
    )
    return df2


def add_rolling_features(
    df: pd.DataFrame,
    roll_cols: list = None,
    windows: tuple = (3, 5)
) -> pd.DataFrame:
    """
    Add lagged rolling means for specified columns.

    Each rolling mean is shifted by 1 to avoid look-ahead.

    Parameters
    ----------
    df : pd.DataFrame
    roll_cols : list, optional
        Columns to roll. Defaults to key process vars.
    windows : tuple
        List of window sizes (rows) for rolling mean.

    Returns
    -------
    pd.DataFrame
        Copy of df with new rolling-mean columns.
    """
    df2 = df.copy()
    if roll_cols is None:
        roll_cols = [
            'dosagem_amina_conc_magnetica',
            'vazao_alimentacao_flotacao',
            'densidade_alimentacao_flotacao',
            'param_dosagem_amido',
        ]
    for window in windows:
        for col in roll_cols:
            col_name = f'{col}_rollmean_{window}'
            df2[col_name] = (
                df2[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .shift(1)
            )
    return df2


def add_delta_silica_regular(
    df: pd.DataFrame,
    source_col: str = 'conc_silica',
    delta_col: str = 'delta_silica',
    time_col: str = 'inicio',
    expected_interval: pd.Timedelta = pd.Timedelta(hours=2)
) -> pd.DataFrame:
    """
    Compute delta_silica only when timestamps are exactly 2h apart.

    Drops any row whose prior timestamp isn't exactly the interval.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data sorted by time_col.
    source_col : str
        Silica concentration column.
    delta_col : str
        Name for new delta column.
    time_col : str
        Datetime column to check interval.
    expected_interval : Timedelta
        Required gap between records.

    Returns
    -------
    pd.DataFrame
        Only rows where time_diff == expected_interval, with delta_col.
    """
    df2 = df.copy().sort_values(time_col).reset_index(drop=True)
    df2['time_diff'] = df2[time_col].diff()
    df2[delta_col] = df2[source_col].diff()
    df2 = df2[df2['time_diff'] == expected_interval]
    return df2.drop(columns=['time_diff']).reset_index(drop=True)


def add_lag_silica_features(
    df: pd.DataFrame,
    time_col: str = 'inicio',
    source_col: str = 'conc_silica',
    lags: list = [2, 4, 6]
) -> pd.DataFrame:
    """
    Merge in prior conc_silica values at t-2h, t-4h, and t-6h.

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    source_col : str
    lags : list of int
        Offsets (in hours) to pull from.

    Returns
    -------
    pd.DataFrame
        Copy of df with '<source_col>_lag_<h>h' columns.
    """
    df2 = df.copy()
    df2[time_col] = pd.to_datetime(df2[time_col])
    for lag in lags:
        tmp = df2[[time_col, source_col]].copy()
        tmp[time_col] += pd.Timedelta(hours=lag)
        tmp = tmp.rename(
            columns={source_col: f"{source_col}_lag_{lag}h"}
        )
        df2 = df2.merge(tmp, on=time_col, how='left')
    return df2


def add_lags(
        df: pd.DataFrame, 
        target_col: str = 'conc_silica', 
        shifts: list = [5,10,30,60]
) -> pd.DataFrame:
    """
    Add simple differences and shifts over fixed periods.

    Differences: 1, 2, 4 rows.    Lags: 5,10,30,60 rows.
    """
    df2 = df.copy()

    for shift in shifts:
        df2[F'lag_{shift}']   = df2[target_col].shift(shift)
    df2.dropna(inplace=True)
    
    return df2

def add_previous_changes(
    df: pd.DataFrame,
    target_col: str = 'conc_silica',
    periods: list[int] = [1, 2, 4]
) -> pd.DataFrame:
    """
    Add prior-period changes in the target without leaking the current value.

    For each period p in `periods`, computes:
      prev_diff_p = conc_silica[i-p] - conc_silica[i-p-1]

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted by time.
    target_col : str
        Name of the column to compute differences on.
    periods : list of int
        List of lag periods (in rows) over which to compute differences.

    Returns
    -------
    pd.DataFrame
        Copy of df with new columns:
        '{target_col}_prev_diff_{p}' for each p in periods.
        The first (max(periods)+1) rows will have NaNs for these features.
    """
    df2 = df.copy()
    for p in periods:
        col_name = f"{target_col}_prev_diff_{p}"
        # compute difference over p rows, then shift by 1 so current value isn't used
        df2[col_name] = df2[target_col].diff(periods=p).shift(1)
    return df2


# Example usage:
# df_feat = (
#     df_clean
#     .pipe(add_silica_quantile)
#     .pipe(add_time_features)
#     .pipe(add_ph_features)
#     .pipe(add_dosage_flow_features)
#     .pipe(add_cell_level_stats)
#     .pipe(add_rolling_features)
#     .pipe(add_delta_silica_regular)
#     .pipe(add_lag_silica_features)
#     .pipe(add_lags)
# )
