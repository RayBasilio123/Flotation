import pandas as pd

def parse_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'inicio' and 'fim' columns into datetimes and compute a 'duration_min' column.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw flotation data with string timestamps in 'inicio' and 'fim'.

    Returns:
    --------
    pd.DataFrame
        A copy of df with:
        - 'inicio' and 'fim' converted to datetime64[ns]
        - new 'duration_min' column (float: minutes)
    """
    df = df.copy()
    df['inicio'] = pd.to_datetime(df['inicio'], format='%Y-%m-%d %H:%M:%S.%f')
    df['fim']    = pd.to_datetime(df['fim'],    format='%Y-%m-%d %H:%M:%S.%f')
    df['duration_min'] = (df['fim'] - df['inicio']).dt.total_seconds() / 60.0
    return df

def remove_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all non-datetime, non-boolean columns to numeric (coercing errors to NaN)
    and then drop any row containing at least one NaN.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame already passed through parse_time.

    Returns:
    --------
    pd.DataFrame
        A copy of df with:
        - all columns except 'inicio', 'fim', 'operacao' converted to numeric
        - all rows with any NaN removed
    """
    df = df.copy()
    exclude = {'inicio', 'fim', 'operacao'}
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df

def drop_consecutive_duplicates_tolerance(df, column='conc_silica', tol=0.01) -> pd.DataFrame:
    """
    Drop consecutive rows where the change in `column` is below a tolerance,
    keeping only the first row of each block of near-constant values.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame (should already have been time‑parsed / cleaned).
    column : str
        The column name to test for near-constant consecutive values.
    tol : float
        Minimum absolute difference required to consider values as different.

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with near-constant duplicate blocks removed.
    """
    # Compute absolute difference from previous row
    delta = df[column].diff().abs()
    # Build mask: True for first row, or where difference >= tol
    mask = (delta >= tol)
    mask.iloc[0] = True  # always keep the very first row
    # Filter and reset index
    return df.loc[mask].reset_index(drop=True)

def filter_by_date_range(df: pd.DataFrame,
                         start_date,
                         end_date,
                         time_col: str = 'inicio') -> pd.DataFrame:
    """
    Return only the rows where `time_col` lies between `start_date` and `end_date` (inclusive).

    Parameters:
    -----------
    df : pd.DataFrame
        Your DataFrame with a datetime column.
    start_date : str or datetime-like
        Lower bound for filtering (e.g. '2024-02-01 00:00:00').
    end_date : str or datetime-like
        Upper bound for filtering.
    time_col : str
        Name of the datetime column to filter on (default: 'inicio').

    Returns:
    --------
    pd.DataFrame
        A new DataFrame containing only rows in the specified interval.
    """
    # Ensure the time column is datetime
    df_filtered = df.copy()
    df_filtered[time_col] = pd.to_datetime(df_filtered[time_col])

    # Parse the start and end boundaries
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)

    # Apply mask
    mask = (df_filtered[time_col] >= start) & (df_filtered[time_col] <= end)
    return df_filtered.loc[mask].reset_index(drop=True)

def filter_by_ph(df: pd.DataFrame,
                 ph_cols: list = ['ph_flotacao_linha01', 'ph_flotacao_linha02'],
                 threshold: float = 6.0) -> pd.DataFrame:
    """
    Remove any row where either of the specified pH columns is below the threshold.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    ph_cols : list of str
        Column names for pH measurements to filter on.
    threshold : float
        Minimum acceptable pH value.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame containing only rows where all specified pH columns
        are >= threshold.
    """
    df_filtered = df.copy()
    # Build a mask that is True only if all pH columns meet or exceed the threshold
    mask = pd.Series(True, index=df_filtered.index)
    for col in ph_cols:
        mask &= df_filtered[col] >= threshold
    # Apply mask and reset index
    return df_filtered.loc[mask].reset_index(drop=True)

def filter_by_silica(df: pd.DataFrame,
                 silica_cols: list = ['conc_silica'],
                 threshold: float = 12) -> pd.DataFrame:
    """
    Remove any row where either of the specified pH columns is below the threshold.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    ph_cols : list of str
        Column names for pH measurements to filter on.
    threshold : float
        Minimum acceptable pH value.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame containing only rows where all specified pH columns
        are >= threshold.
    """
    df_filtered = df.copy()
    # Build a mask that is True only if all pH columns meet or exceed the threshold
    mask = pd.Series(True, index=df_filtered.index)
    for col in silica_cols:
        mask &= df_filtered[col] <= threshold
    # Apply mask and reset index
    return df_filtered.loc[mask].reset_index(drop=True)

def filter_by_flow(df: pd.DataFrame,
                   flow_col: str = 'vazao_alimentacao_flotacao',
                   threshold: float = 470.0) -> pd.DataFrame:
    """
    Remove any row where the specified flow column is below the threshold.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    flow_col : str
        Name of the flow-rate column to filter on.
    threshold : float
        Minimum acceptable flow rate (rows with flow < threshold are removed).

    Returns:
    --------
    pd.DataFrame
        A new DataFrame containing only rows where flow_col >= threshold.
    """
    df_filtered = df.copy()
    mask = df_filtered[flow_col] >= threshold
    return df_filtered.loc[mask].reset_index(drop=True)

# Usage example:
# df_flow_ok = filter_by_flow(df_clean, 'vazao_alimentacao_flotacao', threshold=470)


# Usage example:
# df_ph_ok = filter_by_ph(df_clean, ['ph_flotacao_linha01', 'ph_flotacao_linha02'], threshold=6)


