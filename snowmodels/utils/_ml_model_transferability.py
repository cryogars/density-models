import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as rmse



def plot_learning_curve(X, y, xgb_params, random_state=42):
    """
    Plot a learning curve showing model performance (RMSE) with increasing training set sizes.
    
    Parameters:
    ===========

    X : array-like
        Features dataset
    y : array-like
        Target variable
    xgb_params : dict
        XGBoost parameters to use for model training
    random_state : int, default=42
        Random seed for reproducibility
    """
    # Make a copy of the parameters to avoid modifying the original
    model_params = xgb_params.copy()
    
    # First, set aside 20% of data as a fixed test set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Define the percentages of the TRAINING data to use 10%, 20%, 30%...80%
    # Note 10% of total dataset will be 12.5% of the training (80%) data.
    train_subset_percentages = [12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
    
    # Convert to percentages of total data for display
    train_size_percentages = [10, 20, 30, 40, 50, 60, 70, 80]
    
    # Calculate actual sample sizes based on the training set size
    n_train_full = len(X_train_full)
    sizes = [int(percentage / 100 * n_train_full) for percentage in train_subset_percentages] # sizes[-1] == n_train_full
    
    
    rmse_values = []
    
    for size, percentage in zip(sizes, train_size_percentages):
        # For each size, we take a random subset of the full training data
        # These subsets get progressively larger
        
        # Get random indices from the full training set
        n_train_full = len(X_train_full)
        subset_size = min(size, n_train_full)  # Ensure we don't try to take more than available
        
        
        if isinstance(X_train_full, pd.DataFrame):
            # For pandas DataFrame
            indices = np.random.RandomState(random_state).choice(
                np.arange(n_train_full), size=subset_size, replace=False
            )
            idx_values = X_train_full.index[indices]
            X_subset = X_train_full.loc[idx_values]
            
            if isinstance(y_train_full, pd.DataFrame) or isinstance(y_train_full, pd.Series):
                if y_train_full.index.equals(X_train_full.index):
                    y_subset = y_train_full.loc[idx_values]
                else:
                    y_idx_values = y_train_full.index[indices]
                    y_subset = y_train_full.loc[y_idx_values]
            else:
                y_subset = y_train_full[indices]
        else:
            # For numpy arrays
            indices = np.random.RandomState(random_state).choice(
                n_train_full, size=subset_size, replace=False
            )
            X_subset = X_train_full[indices]
            y_subset = y_train_full[indices]
        
        
        dtrain = xgb.DMatrix(X_subset, label=y_subset)
        dtest = xgb.DMatrix(X_test)
        
        
        model = xgb.train(
            model_params,
            dtrain,
            num_boost_round=1498, # from model training in 05_ml_models.ipynb
        )
        
       
        y_pred = model.predict(dtest)
        
        
        rmse_ = rmse(y_true=y_test, y_pred=y_pred) 
        rmse_values.append(rmse_)
    
    return train_size_percentages, rmse_values