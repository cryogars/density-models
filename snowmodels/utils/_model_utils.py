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

# Define the structured return type
SplitResult = namedtuple('SplitResult', [
    'X_train', 'X_val', 'X_test', 'X_temp',
    'y_train', 'y_val', 'y_test', 'y_temp',
    'train_df', 'val_df', 'test_df', 'temp_df'
])


# Set the seed for reproducibility
SEED = 10

class DataSplitter(ABC):

    """
    Abstract base class for different data splitting strategies
    """

    def __init__(self, seed: int = SEED):

        self.see = seed

    @abstractmethod
    def split(self, station_metadata: pd.DataFrame, df: pd.DataFrame) -> SplitResult:
        """
        Split the data according to the strategy
        """
        pass

    def _prepare_output(self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            test_df: pd.DataFrame,
            temp_df: pd.DataFrame
    ) -> SplitResult:
        
         """Helper method to prepare the output as a named tuple"""

         # Split features and target
         X_train, y_train = train_df.drop('Snow_Density', axis=1), train_df.Snow_Density
         X_val, y_val = val_df.drop('Snow_Density', axis=1), val_df.Snow_Density
         X_test, y_test = test_df.drop('Snow_Density', axis=1), test_df.Snow_Density
         X_temp, y_temp = temp_df.drop('Snow_Density', axis=1), temp_df.Snow_Density

         return SplitResult(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            X_temp=X_temp,
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

# def split_data(station_metadata: pd.DataFrame, df: pd.DataFrame, seed: int  = SEED) -> Dict[str, pd.DataFrame]:

#     """
#     A function that splits the data into training (70%), testing (20%) and tuning (10%) sets.

#     Parameters:
#     -----------
#     df : pandas DataFrame
#         A pandas DataFrame containing the data to split.

#     Returns:
#     --------
#     A dictionary containing the training, testing and tuning sets.
#     """

#     strata = station_metadata.Snow_Class

#     temp_stations, test_stations = train_test_split(
#         station_metadata, test_size=0.20, 
#         stratify=strata, random_state=seed
#     )

#     strata2 = temp_stations.Snow_Class

#     train_stations, val_stations = train_test_split(
#         temp_stations, test_size=1/8, 
#         stratify=strata2, random_state=seed
#     )

#     temp_df = df.query("Station_Name in @temp_stations.Station_Name")
#     train_df = df.query("Station_Name in @train_stations.Station_Name")
#     val_df = df.query("Station_Name in @val_stations.Station_Name")
#     test_df = df.query("Station_Name in @test_stations.Station_Name")


#     X_temp, y_temp = temp_df.drop('Snow_Density', axis=1), temp_df.Snow_Density
#     X_train, y_train = train_df.drop('Snow_Density', axis=1), train_df.Snow_Density
#     X_val, y_val = val_df.drop('Snow_Density', axis=1), val_df.Snow_Density
#     X_test, y_test = test_df.drop('Snow_Density', axis=1), test_df.Snow_Density



#     return {
#         'X_train': X_train,
#         'X_test': X_test,
#         'X_val': X_val,
#         'X_temp': X_temp,
#         'y_train': y_train,
#         'y_test': y_test,
#         'y_val': y_val,
#         'y_temp': y_temp
#     }


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

    RMSE = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
    MBE  = np.mean(y_pred- y_true)
    RSQ  = r2_score(y_true=y_true, y_pred=y_pred)
    
    score_df = pd.DataFrame({
        model_name: [RMSE, MBE, RSQ]
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
        eval=evaluate_model(y_true=preds_df[y_true], y_pred=other_preds[col], model_name=col)
        evaluations.append(eval)

    return pd.concat(evaluations, axis=1)
