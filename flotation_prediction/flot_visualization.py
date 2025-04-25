import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display

def plot_time_series(df, time_col, value_col):
    """
    Interactive time series plot using Plotly.
    """
    fig = px.line(df, x=time_col, y=value_col, title=f"{value_col} Over Time")
    fig.show()

def plot_dot_time_series(df, time_col, value_col):
    """
    Interactive time series plot using Plotly.
    """
    fig = px.plot(df, x=time_col, y=value_col, title=f"{value_col} Over Time")
    fig.show()

def plot_histogram(df, column, nbins=30):
    """
    Interactive histogram plot using Plotly.
    """
    fig = px.histogram(df, x=column, nbins=nbins, title=f"Distribution of {column}")
    fig.show()

def plot_scatter(df, x_col, y_col,xticks=None, yticks=None):
    """
    Interactive scatter plot using Plotly.
    """
    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
    if xticks:
        fig.update_xaxes(nticks=xticks)

    if yticks:
        fig.update_yaxes(nticks=yticks)

    fig.show()

def plot_pairwise(df, columns):
    """
    Interactive pairwise scatter matrix using Plotly.
    """
    fig = px.scatter_matrix(df, dimensions=columns, title="Pairwise Scatter Matrix")
    fig.show()

def plot_time_series_with_gaps(df: pd.DataFrame,
                               time_col: str = 'inicio',
                               value_col: str = 'conc_silica',
                               gap_threshold: float = None):
    """
    Plot a time series while breaking lines where data is missing (gaps in timestamps).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime and value columns.
    time_col : str
        Name of the datetime column.
    value_col : str
        Name of the value column to plot.
    gap_threshold : float, optional
        Threshold (in seconds) for detecting a gap. Rows where the time
        difference from the previous row exceeds this will start a new segment.
        If None, uses 2 * median interval.
    """
    # Prepare data
    df_plot = df.copy()
    df_plot[time_col] = pd.to_datetime(df_plot[time_col])
    df_plot = df_plot.sort_values(time_col).reset_index(drop=True)

    # Compute time deltas (seconds)
    dt = df_plot[time_col].diff().dt.total_seconds()

    # Determine threshold
    if gap_threshold is None:
        gap_threshold = dt.median() * 2

    # Assign segment IDs
    segments = (dt > gap_threshold).cumsum()

    # Build figure
    fig = go.Figure()
    for seg_id, segment_df in df_plot.assign(segment=segments).groupby('segment'):
        fig.add_trace(go.Scatter(
            x=segment_df[time_col],
            y=segment_df[value_col],
            mode='lines',
            name=f'Segment {seg_id}',
            showlegend=False  # hide legend entries
        ))

    fig.update_layout(
        title=f"{value_col} Over Time (gaps shown)",
        xaxis_title=time_col,
        yaxis_title=value_col
    )
    fig.show()

def plot_scatter_all(df: pd.DataFrame, target: str = 'conc_silica'):
    """
    Generate interactive scatter plots of the target variable against all other numeric features.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned DataFrame.
    target : str
        Name of the target variable column.
    """
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    # Exclude the target itself
    numeric_cols = [col for col in numeric_cols if col != target]
    # Optionally exclude boolean columns if they were coerced to numeric (e.g., operacao)
    for col in ['operacao']:
        if col in numeric_cols:
            numeric_cols.remove(col)

    # Loop and plot
    for col in numeric_cols:
        fig = px.scatter(
            df,
            x=col,
            y=target,
            title=f"{target} vs {col}",
            labels={col: col, target: target}
        )
        # Increase default nticks for axis detail
        fig.update_xaxes(nticks=20)
        fig.update_yaxes(nticks=20)
        display(fig)

def plot_residuals(y_true, y_pred, 
                   actual_name: str = 'actual', 
                   pred_name: str = 'predicted',
                   nbins: int = 30):
    """
    Generate residual diagnostics plots:
      1) Scatter of residuals vs. predicted values
      2) Scatter of residuals vs. actual values
      3) Scatter of actual vs. predicted values
      4) Histogram of residual distribution

    Parameters:
    -----------
    y_true : array-like or pd.Series
        True target values.
    y_pred : array-like or pd.Series
        Model predictions.
    actual_name : str
        Label for the true values (default 'actual').
    pred_name : str
        Label for the predicted values (default 'predicted').
    nbins : int
        Number of bins for the histogram (default 30).
    """
    # Build DataFrame
    df = pd.DataFrame({
        actual_name: y_true,
        pred_name: y_pred
    })
    df['residual'] = df[actual_name] - df[pred_name]

    # 1) Residuals vs. Predicted
    fig1 = px.scatter(
        df, x=pred_name, y='residual',
        title="Residuals vs. Predicted Values",
        labels={pred_name: pred_name, 'residual': 'Residual'}
    )
    fig1.add_hline(y=0, line_dash="dash", line_color="red")
    fig1.show()

    # 2) Residuals vs. Actual
    fig2 = px.scatter(
        df, x=actual_name, y='residual',
        title="Residuals vs. Actual Values",
        labels={actual_name: actual_name, 'residual': 'Residual'}
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    fig2.show()

    # 3) Actual vs. Predicted
    fig3 = px.scatter(
        df, x=actual_name, y=pred_name,
        title="Actual vs. Predicted Values",
        labels={actual_name: actual_name, pred_name: pred_name}
    )
    # Add identity line
    min_val = min(df[actual_name].min(), df[pred_name].min())
    max_val = max(df[actual_name].max(), df[pred_name].max())
    fig3.add_shape(
        type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
        line=dict(color='red', dash='dash')
    )
    fig3.show()

    # 4) Residual distribution
    fig4 = px.histogram(
        df, x='residual', nbins=nbins,
        title="Residuals Distribution",
        labels={'residual': 'Residual'}
    )
    fig4.show()