import pandas as pd

TIME_SERIES_INTERVALS = {
    'S': 'Second',
    'L': 'Second',
    'ms': 'Millisecond',
    'us': 'Microsecond',
    'U': 'Microsecond',
    'ns': 'Nanosecond',
    'T': 'Minute',
    'min': 'Minute',
    'H': 'Hour',
    'h': 'Hour',
    'D': 'Day',
    'B': 'Business Day',
    'W': 'Week',
    'W-SUN': 'Week (Sunday)',
    'W-MON': 'Week (Monday)',
    'W-TUE': 'Week (Tuesday)',
    'W-WED': 'Week (Wednesday)',
    'W-THU': 'Week (Thursday)',
    'W-FRI': 'Week (Friday)',
    'W-SAT': 'Week (Saturday)',
    'M': 'Month',
    'MS': 'Month Start',
    'Q': 'Quarter',
    'QS': 'Quarter Start',
    'Y': 'Year',
    'YS': 'Year Start',
}


def generate_numerical_range(df: pd.DataFrame) -> list[str]:
    """
    Generate numerical range based on the given dataframe.
    """
    numerical_range = []
    for idx, col in enumerate(df.columns):
        if (
            df[col].dtype == "int64"
            or df[col].dtype == "float64"
            or df[col].dtype == "datetime64[ns]"
        ):
            range = {"min": df[col].min(), "max": df[col].max()}
            numerical_range.append({col: range})
    return numerical_range


def generate_time_series_intervals(df: pd.DataFrame) -> list[str]:
    """
    Generate time series intervals based on the given dataframe.
    e.g., hourly, monthly, yearly, etc.
    """
    time_series_intervals = []
    for idx, col in enumerate(df.columns):
        if df[col].dtype == "datetime64[ns]":
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
            freq = pd.infer_freq(df.index)
            freq = (
                TIME_SERIES_INTERVALS[freq] if freq in TIME_SERIES_INTERVALS else freq
            )
            time_series_intervals.append({col: freq})

    return time_series_intervals
