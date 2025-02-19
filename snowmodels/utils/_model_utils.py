import datetime
import numpy as np
import pandas as pd
from typing import Dict, Union 
from ._conversions import ConvertData, OutOfBoundsError

## Machine Learning Libraries
import category_encoders as ce
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

set_config(transform_output="pandas")


# Set the seed for reproducibility
seed = 10


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
        X, y, test_size=0.20, stratify=strata, random_state=seed
    )

    strata2 = X_temp['Snow_Class']

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=1/8, stratify=strata2, random_state=seed
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
   

def preprocessing_pipeline(numeric_features: list, categorical_feature: list, scaler=None, encoder=None) -> ColumnTransformer:

    """
    A function that preprocesses the data by scaling the numeric features and encoding the categorical features.

    Parameters:
    -----------
    numeric_features : list
        A list containing the names of the numeric features.
    
    categorical_feature : list
        A list containing the names of the categorical features.

    scaler : object
        A scaler object that scales the numeric features. Default is None. Uses StandardScaler if None.
    
    encoder : object
        An encoder object that encodes the categorical features. Default is None. Uses TargetEncoder if None.


    Returns:
    --------
    A ColumnTransformer object that preprocesses the data.
    """

    if scaler is None:
        scaler=StandardScaler()
    
    if encoder is None:
        encoder=ce.TargetEncoder(min_samples_leaf=20, smoothing=10)
    
    numeric_transformer=Pipeline(steps=[
        ('scale', scaler) ## Scaling the data
    ])
    
    categorical_transformer=Pipeline(steps=[
        ('encoder', encoder)
    ])
    
    feature_engineering_pipeline= ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_feature)
        ],
        remainder='drop'
    )

    return feature_engineering_pipeline


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


def preprocess_data(data, encoder, numeric_features=['Elevation', 'Snow_Depth', 'DOWY']):

    """
    Transforms a dataset by encoding the 'Snow_Class' feature and retaining specified numeric features.

    This function applies one of two encoding strategies based on the type of encoder provided:
    - If the encoder is an instance of `OneHotEncoder`, it performs one-hot encoding on the 'Snow_Class' column.
      The function fits the encoder on `X_train` and applies the transformation to `X_val` and `X_test`.
    - For other encoders, it assumes a different encoding strategy (e.g., label encoding) and applies it to 'Snow_Class'.

    In both cases, the function retains a specified subset of numeric features in each transformed dataset.

    Parameters
    ----------
    data : dict of pandas.DataFrame
        A dictionary containing the datasets, expected to have keys 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', and 'y_test'.
        Each DataFrame in `X_train`, `X_val`, and `X_test` must include the 'Snow_Class' column.
        
    encoder : sklearn encoder instance
        The encoder to apply to the 'Snow_Class' column. Should be compatible with `fit_transform` and `transform` methods.
        For one-hot encoding, use `OneHotEncoder` from sklearn. Other encoders will be applied differently to the 'Snow_Class' column.
        
    numeric_features : list of str, optional
        A list of numeric feature column names to retain in each transformed dataset, by default ['Elevation', 'Snow_Depth', 'DOWY'].

    Returns
    -------
    dict of pandas.DataFrame
        A dictionary containing the transformed training, validation, and test sets with encoded 'Snow_Class' values:
        
        - 'X_train_transformed': Transformed training set with selected numeric features and encoded 'Snow_Class'.
        - 'X_val_transformed': Transformed validation set with selected numeric features and encoded 'Snow_Class'.
        - 'X_test_transformed': Transformed test set with selected numeric features and encoded 'Snow_Class'.
        - 'y_train': Original target values for the training set.
        - 'y_val': Original target values for the validation set.
        - 'y_test': Original target values for the test set.
        
    Examples
    --------
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> encoder = OneHotEncoder()
    >>> data = {
    ...     'X_train': df_train,
    ...     'X_val': df_val,
    ...     'X_test': df_test,
    ...     'y_train': y_train,
    ...     'y_val': y_val,
    ...     'y_test': y_test
    ... }
    >>> transformed_data = transform_snow_class(data, encoder, numeric_features=['Elevation', 'Snow_Depth', 'DOWY'])
    >>> transformed_data['X_train_transformed'].head()
    """

    if encoder.__class__.__name__ == 'OneHotEncoder':

        X_train_transformed=(
            data['X_train']
            .filter(items=numeric_features)
            .assign(
                **pd.DataFrame(
                    encoder.fit_transform(data['X_train'].Snow_Class.values.reshape(-1,1)), 
                    columns=list(encoder.categories_[0]), 
                    index=data['X_train'].index
                ).to_dict(orient='list')
            )
        )

        X_val_transformed=(
            data['X_val']
            .filter(items=numeric_features)
            .assign(
                **pd.DataFrame(
                    encoder.transform(data['X_val'].Snow_Class.values.reshape(-1,1)), 
                    columns=list(encoder.categories_[0]), 
                    index=data['X_val'].index
                ).to_dict(orient='list')
            )
        )

        X_test_transformed=(
            data['X_test']
            .filter(items=numeric_features)
            .assign(
                **pd.DataFrame(
                    encoder.transform(data['X_test'].Snow_Class.values.reshape(-1,1)), 
                    columns=list(encoder.categories_[0]), 
                    index=data['X_test'].index
                ).to_dict(orient='list')
            )
        )

    else:

        X_train_transformed=(
            data['X_train']
            .filter(items=numeric_features)
            .assign(Snow_Class=encoder.fit_transform(data['X_train'].Snow_Class, data['y_train']))
        )

        X_val_transformed=(
            data['X_val']
            .filter(items=numeric_features)
            .assign(Snow_Class=encoder.transform(data['X_val'].Snow_Class))
        )

        X_test_transformed=(
            data['X_test']
            .filter(items=numeric_features)
            .assign(Snow_Class=encoder.transform(data['X_test'].Snow_Class))
        )


    return {
        'X_train_transformed': X_train_transformed,
        'X_val_transformed': X_val_transformed,
        'X_test_transformed': X_test_transformed,
        'y_train': data['y_train'],
        'y_test': data['y_test'],
        'y_val': data['y_val']
    }