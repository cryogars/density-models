import datetime
from typing import Dict, Union 

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


def split_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:

    """
    A function that splits the data into training (70%), testing (20%) and tuning (10%) sets.

    Parameters:
    -----------
    df : pandas DataFrame
        A pandas DataFrame containing the data to split.

    Returns:
    --------
    A dictionary containing the training, testing and tuning sets.
    """

    X, y, strata = df.drop('Snow_Density', axis=1), df['Snow_Density'], df['Snow_Class']


    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=strata, random_state=SEED
    )

    strata2 = X_temp['Snow_Class']

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=1/8, stratify=strata2, random_state=SEED
    )


    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val,
        'X_temp': X_temp,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val,
        'y_temp': y_temp
    }


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
            raise ValueError(f"Could not convert {x} to a valid DOY. {e}")
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


