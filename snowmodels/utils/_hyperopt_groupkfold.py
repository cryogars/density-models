
"""Module providing a Grouped Kfold hyperparameter tuning."""

import logging
import warnings
from datetime import datetime
import yaml
import torch
import optuna
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import root_mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


warnings.filterwarnings('ignore')
xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration
BOOSTING_MODELS = ['lightgbm', 'xgboost']
SKLEARN_MODELS = ['rf', 'extratrees']

def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"grouped_kfold_optuna_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting hyperparameter optimization with Grouped KFold")
    logger.info("Log file: %s", log_filename)
    return logger, timestamp

def load_config(config_path="hyperparameters.yaml"):
    """Load hyperparameter configuration from YAML file"""
    try:
        with open(file=config_path, mode= 'r', encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as e:
        raise ValueError("Error parsing YAML config") from e

def get_model_class(model_name):
    """Get the model class based on model name"""
    model_map = {
        'rf': RandomForestRegressor,
        'extratrees': ExtraTreesRegressor,
        'lightgbm': None,  # Will use native API
        'xgboost': None,   # Will use native API
    }
    
    if model_name not in model_map:
        available_models = list(model_map.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available_models}")
    
    return model_map[model_name]

def suggest_hyperparameters(trial, model_name, config):
    """Suggest hyperparameters based on config"""
    model_config = config['models'][model_name]
    global_config = config['global']
    params = {}
    
    # Suggest model-specific hyperparameters
    for param_name, param_config in model_config.items():
        if param_config['type'] == 'int':
            if 'step' in param_config:
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high'],
                    step=param_config['step']
                )
            else:
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high']
                )

        elif param_config['type'] == 'float':
            log_scale = param_config.get('log', False)
            params[param_name] = trial.suggest_float(
                param_name, 
                param_config['low'], 
                param_config['high'],
                log=log_scale
            )
        
        elif param_config['type'] == 'categorical':
            params[param_name] = trial.suggest_categorical(
                param_name, 
                param_config['choices']
            )
    
    # Add global and model-specific fixed parameters

    if model_name in SKLEARN_MODELS:
        params.update({
            'random_state': global_config['seed'],
            'n_jobs': global_config['n_jobs']
        })
    
    elif model_name == 'lightgbm':
        params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'seed': global_config['seed'],
            'verbosity': global_config['verbosity'],
            'force_col_wise': True,
            'deterministic': True
        })
    
    elif model_name == 'xgboost':
        params.update({
            'objective': 'reg:squarederror',
            'seed': global_config['seed'],
            'tree_method': 'hist',
            'device': xgb_device,
            'sampling_method': 'gradient_based',
            'verbosity': 0  # XGBoost uses 0 for silent
        })
    
    return params


def create_encoder_pipeline(encoder_type, categorical_column):
    """Create preprocessing pipeline with specified encoder"""
    
    if encoder_type == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        preprocessor = ColumnTransformer(
            transformers=[('cat', encoder, [categorical_column])],
            remainder='passthrough'
        )
    
    elif encoder_type == 'target':
        # Use CV=5 to avoid target leakage
        encoder = TargetEncoder(target_type='continuous', cv=5, random_state=10, smooth="auto")
        preprocessor = ColumnTransformer(
            transformers=[('cat', encoder, [categorical_column])],
            remainder='passthrough'
        )
    
    elif encoder_type == 'catboost':
        encoder = ce.CatBoostEncoder(random_state=10)
        preprocessor = ColumnTransformer(
            transformers=[('cat', encoder, [categorical_column])],
            remainder='passthrough'
        )
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    return preprocessor

def train_with_native_api(X_train, y_train, X_val, y_val, model_name, model_params, config):
    """Train model using native API with validation set and early stopping"""
    
    global_config = config['global']
    early_stopping_rounds = global_config['early_stopping_rounds']
    
    if model_name == 'lightgbm':
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train with early stopping
        model = lgb.train(
            model_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=0)  # Silent training
            ]
        )
        
        # Predict on validation set
        y_pred = model.predict(X_val)
        
    elif model_name == 'xgboost':
        # Create XGBoost matrices
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train with early stopping
        model = xgb.train(
            model_params,
            dtrain,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False  # Silent training
        )
        
        # Predict on validation set
        y_pred = model.predict(dval)
        
    else:
        raise ValueError(f"Native API not supported for model: {model_name}")
    
    return model, y_pred