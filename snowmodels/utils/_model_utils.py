
"""This module contains modeling utils"""

import datetime
from typing import Dict, Union 
from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
import pandas as pd

## Machine Learning Libraries
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

from ._conversions import ConvertData, OutOfBoundsError

set_config(transform_output="pandas")

# Set the seed for reproducibility
SEED = 10

# Define the structured return type
SplitResult = namedtuple('SplitResult', [
    'X_train', 'X_val', 'X_test', 'X_temp',
    'y_train', 'y_val', 'y_test', 'y_temp',
    'train_df', 'val_df', 'test_df', 'temp_df'
])


class DataSplitter(ABC):

    """
    Abstract base class for different data splitting strategies
    """

    def __init__(self, seed: int = SEED):
        self.seed = seed

    @abstractmethod
    def split(self, station_metadata: pd.DataFrame, df: pd.DataFrame) -> SplitResult:
        """
        Split the data according to the strategy
        """

    def _prepare_output(self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            test_df: pd.DataFrame,
            temp_df: pd.DataFrame
    ) -> SplitResult:

        """Helper method to prepare the output as a named tuple"""

        # Split features and target
        x_train, y_train = train_df.drop('Snow_Density', axis=1), train_df.Snow_Density
        x_val, y_val = val_df.drop('Snow_Density', axis=1), val_df.Snow_Density
        x_test, y_test = test_df.drop('Snow_Density', axis=1), test_df.Snow_Density
        x_temp, y_temp = temp_df.drop('Snow_Density', axis=1), temp_df.Snow_Density


        return SplitResult(
        X_train=x_train,
        X_val=x_val,
        X_test=x_test,
        X_temp=x_temp,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_temp=y_temp,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        temp_df=temp_df
        )

    def get_split_info(self, splits: SplitResult) -> Dict[str, any]:
        """Get information about the splits"""
        return {
            'strategy': self.__class__.__name__,
            'train_samples': len(splits.train_df),
            'val_samples': len(splits.val_df),
            'test_samples': len(splits.test_df),
            'train_stations': splits.train_df['Station_Name'].nunique(),
            'val_stations': splits.val_df['Station_Name'].nunique(),
            'test_stations': splits.test_df['Station_Name'].nunique(),
            'total_stations_used': pd.concat([
                splits.train_df, splits.val_df, splits.test_df
            ])['Station_Name'].nunique()
        }

class SpatialSplitter(DataSplitter):
    """Strategy 1: Full spatial split (stations completely separated)"""

    def split(self, station_metadata: pd.DataFrame, df: pd.DataFrame) -> SplitResult:
        strata = station_metadata.Snow_Class

        # Split stations first
        temp_stations, test_stations = train_test_split(
            station_metadata, test_size=0.2,
            stratify=strata, random_state=300
        )

        strata2 = temp_stations.Snow_Class
        train_stations, val_stations = train_test_split(
            temp_stations, test_size=1/8,
            stratify=strata2, random_state=300
        )

        # Get data for each station set
        train_df = df.query("Station_Name in @train_stations.Station_Name")
        val_df = df.query("Station_Name in @val_stations.Station_Name")
        test_df = df.query("Station_Name in @test_stations.Station_Name")
        temp_df = df.query("Station_Name in @temp_stations.Station_Name")

        return self._prepare_output(train_df, val_df, test_df, temp_df)


class HybridSplitter(DataSplitter):
    """Strategy 2: Time Series train/val + spatial test"""

    def split(self, station_metadata: pd.DataFrame, df: pd.DataFrame) -> SplitResult:
        strata = station_metadata.Snow_Class

        # First, spatial split for test set (20% of stations)
        train_val_stations, test_stations = train_test_split(
            station_metadata, test_size=150,
            stratify=strata, random_state=self.seed
        )

        # Get data from train/val stations and test stations
        train_val_data = df.query("Station_Name in @train_val_stations.Station_Name")
        test_df = df.query("Station_Name in @test_stations.Station_Name")

        cutoff_year = 2021

        train_df = train_val_data[train_val_data["Date"].dt.year <= cutoff_year]
        val_df = train_val_data[train_val_data["Date"].dt.year > cutoff_year]

        temp_df = train_val_data

        return self._prepare_output(train_df, val_df, test_df, temp_df)


class SplitterFactory:
    """Factory class to create splitters"""

    @staticmethod
    def create_splitter(strategy: str, seed: int = SEED) -> DataSplitter:
        """Create Splitter"""
        splitters = {
            'spatial': SpatialSplitter,
            'hybrid': HybridSplitter
        }

        if strategy not in splitters:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(splitters.keys())}")

        return splitters[strategy](seed=seed)

    @staticmethod
    def get_all_strategies() -> list:
        """A function that returns all splitting strategy"""
        return ['spatial', 'hybrid']

# let's create a function that will help us evaluate the results of the model

def evaluate_model(
    y_true: Union[pd.Series, list],
    y_pred: Union[pd.Series, list],
    model_name: str
) -> pd.DataFrame:

    """
    A function that evaluates the performance of a model using the RMSE, MBE and R2 metrics.

    Parameters:
    -----------
    y_true : pandas Series or list
        A pandas Series or list containing the true values of the target variable.

    y_pred : pandas Series or list
        A pandas Series or list containing the predicted values of the target variable.
    
    model_name : str
        A string representing the name of the model.

    Returns:
    --------
    A pandas DataFrame containing the RMSE, MBE and R2 metrics for the model.
    """

    rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
    mbe  = np.mean(y_pred- y_true)
    rsq  = r2_score(y_true=y_true, y_pred=y_pred)

    score_df = pd.DataFrame({
        model_name: [rmse, mbe, rsq]
    }, index = ['RMSE', 'MBE', 'RSQ'])

    return score_df


def validate_DOY(x: int | float | str | pd.Timestamp | datetime.datetime, origin: int = None) -> int:

    """
    Validates or converts an input to a day of the year (DOY).
    Accepts integer, float, string, datetime.datetime, or pd.Timestamp inputs.
    If the input is an integer, float, or a string of integer it must be between 1 and 366. 
    If the input is a string, it must be convertible to a valid date.
    """

    try:
        float_x = float(x)
    except:
        pass

    else:
        if float_x.is_integer():
            doy = int(float_x)
            if doy >= 1 and doy <= 366:
                return doy
            else:
                raise OutOfBoundsError(f"DOY must be between 1 and 366. Got {doy}.")
        else:
            raise ValueError(f"DOY must be a whole number. Got {x}.")

    if isinstance(x, (str, pd.Timestamp, datetime.datetime)):
        try:
            timestamp = pd.Timestamp(x) if isinstance(x, str) else x

            if origin < 1 or origin > 12:
                raise OutOfBoundsError(f"Origin must be between 1 and 12. Got {origin}.")

            converter=ConvertData()
            return converter.date_to_DOY(date=timestamp, origin=origin, algorithm='default')
        except ValueError as e:
            raise ValueError(f"Could not convert {x} to a valid DOY. {e}") from e
    else:
        raise TypeError(f"Input type is not supported. Expected types are int, float, str, datetime.datetime, or pd.Timestamp, got {type(x).__name__}.")


def compare_multiple_models(preds_df: pd.DataFrame, y_true: str) -> pd.DataFrame:

    """
    A function that compares the performance of multiple models using the RMSE, MBE and R2 metrics.

    Parameters:
    -----------

    preds_df : pandas DataFrame
        A pandas DataFrame containing the predictions of multiple models.
    
    y_true : str
        A string representing the name of the target variable.

    Returns:
    --------

    A pandas DataFrame containing the RMSE, MBE and R2 metrics for each model.
    """

    other_preds=preds_df.drop(y_true, axis=1)

    evaluations=[]

    for col in other_preds.columns:
        eval_=evaluate_model(y_true=preds_df[y_true], y_pred=other_preds[col], model_name=col)
        evaluations.append(eval_)

    return pd.concat(evaluations, axis=1)
