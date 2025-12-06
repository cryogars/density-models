
import numpy as np
import pandas as pd


def preprocess_set_to_nan(df):
    """
    Set invalid values to NaN based on specified conditions.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Dataframe with invalid values set to NaN
    """
    # Temperature relationship
    mask_temp = ~(df['TMIN'] < df['TAVG']) | ~(df['TAVG'] < df['TMAX'])
    df.loc[mask_temp, ['TMIN', 'TAVG', 'TMAX']] = np.nan
    
    # Individual temperature ranges
    df.loc[df['TMIN'].lt(-40) | df['TMIN'].gt(40), 'TMIN'] = np.nan
    df.loc[df['TMAX'].lt(-40) | df['TMAX'].gt(40), 'TMAX'] = np.nan
    df.loc[df['TAVG'].lt(-40) | df['TAVG'].gt(40), 'TAVG'] = np.nan
    
    # Precipitation
    df.loc[df['PRECIPITATION'].lt(0) | df['PRECIPITATION'].gt(df['threshold']), 'PRECIPITATION'] = np.nan
    
    return df



def calculate_lagged_vars(df, col_of_interest, window=7):
    """
    Calculate the rolling mean of a column of interest for a given window size.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    col_of_interest (str): Name of the column of interest
    window (int): Window size for the rolling mean
    
    Returns:
    pd.DataFrame: Dataframe with added lagged variable column
    """
    df[f"{col_of_interest}_lag_{window}d"] = (
        df.groupby("Station Name")[col_of_interest]
        .transform(lambda x: x.shift().rolling(window=f"{window}D" , min_periods=1).mean())
    )
    # Set the first lag_days for each station to NaN
    df[f'{col_of_interest}_lag_{window}d'] = df.groupby("Station Name")[f'{col_of_interest}_lag_{window}d'].transform(
        lambda x: x.mask(x.index < x.index.min() + pd.Timedelta(days=window), np.nan)
    )
    return df

def add_multiple_lags(df, cols, max_lag):
    for col in cols:
        for window in range(1, max_lag + 1):
            df = calculate_lagged_vars(df, col_of_interest=col, window=window)
    return df


def calculate_pptwt(group, df_of_interest: pd.DataFrame) -> float:
    winter_starts = pd.Timestamp(year=int(group.winter_start.reset_index(drop=True).iloc[0]), month=12, day=1, tz='UTC')
    winter_ends = pd.Timestamp(year=int(group.winter_end.reset_index(drop=True).iloc[0]), month=3, day=1, tz='UTC')
    winter_data = df_of_interest[
        (df_of_interest['Date'] >= winter_starts) &
        (df_of_interest['Date'] < winter_ends) &
        (df_of_interest['Station_Name'] == group.name[0])
    ]
    if winter_data.empty or winter_data['Date'].dt.month.nunique() < 3 or winter_data.dropna(subset=['PRECIPITATION']).empty:
        return np.nan
    return winter_data['PRECIPITATION'].sum()