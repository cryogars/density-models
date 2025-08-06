
"""Module providing a Grouped Kfold hyperparameter tuning."""

import logging
import warnings
import argparse
from typing import Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import yaml
import torch
import optuna
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


warnings.filterwarnings('ignore')
xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration
BOOSTING_MODELS = ['lightgbm', 'xgboost']
SKLEARN_MODELS = ['rf', 'extratrees']


@dataclass
class TrainingData:
    x_train: Any
    y_train: Any
    x_val: Any
    y_val: Any

@dataclass  
class ModelConfig:
    name: str
    params: dict

@dataclass
class TrainingConfig:
    data: TrainingData
    model: ModelConfig
    config: dict

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

def train_with_native_api(training_config: TrainingConfig) -> Tuple[Any, Any]:
    """Train model using native API with validation set and early stopping"""
    
    data = training_config.data
    model_config = training_config.model
    config = training_config.config
    
    global_config = config['global']
    early_stopping_rounds = global_config['early_stopping_rounds']
    
    if model_config.name == 'lightgbm':
        # Create LightGBM datasets
        train_data = lgb.Dataset(data.x_train, label=data.y_train)
        val_data = lgb.Dataset(data.x_val, label=data.y_val, reference=train_data)
        
        # Train with early stopping
        model = lgb.train(
            model_config.params,
            train_data,
            valid_names=['train', 'val'],
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=0)  # Silent training
            ]
        )
        
        # Predict on validation set
        y_pred = model.predict(data.x_val, num_iteration=model.best_iteration)
        
    elif model_config.name == 'xgboost':
        # Create XGBoost matrices
        dtrain = xgb.DMatrix(data.x_train, label=data.y_train)
        dval = xgb.DMatrix(data.x_val, label=data.y_val)

        early_stopping = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name='rmse',
            data_name='valid',
            maximize=False,
            save_best=True
        )
        
        # Train with early stopping
        model = xgb.train(
            model_config.params,
            dtrain,
            evals=[(dtrain, 'train'), (dval, 'val')],
            callbacks=[early_stopping],
            verbose_eval=False  # Silent training
        )
        
        # Predict on validation set
        y_pred = model.predict(dval)
        
    else:
        raise ValueError(f"Native API not supported for model: {model_config.name}")
    
    return model, y_pred

def objective(trial, X, y, groups, categorical_column, encoder_type, model_name, config):
    """Objective function for Optuna optimization"""
    
    # Suggest hyperparameters based on config
    model_params = suggest_hyperparameters(trial, model_name, config)
    
    if model_name in BOOSTING_MODELS:
        # Use native API with manual cross-validation for boosting models
        gkf = GroupKFold(n_splits=10, shuffle=True, random_state=10)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            x_train_fold, x_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Get preprocessor and transform data
            preprocessor = create_encoder_pipeline(encoder_type, categorical_column)
            x_train_processed = preprocessor.fit_transform(x_train_fold, y_train_fold)
            x_val_processed = preprocessor.transform(x_val_fold)
            
            try:
                # Train with native API
                training_data = TrainingData(x_train_processed, y_train_fold, x_val_processed, y_val_fold)
                model_config = ModelConfig(model_name, model_params)
                training_config = TrainingConfig(training_data, model_config, config)
                model, y_pred = train_with_native_api(training_config)
                
                # Calculate RMSE
                rmse = root_mean_squared_error(y_true=y_val_fold, y_pred=y_pred)
                fold_scores.append(rmse)
                
            except RuntimeError as exc:
                # If training fails, return a bad score
                trial.report(float('inf'), fold)
                if trial.should_prune():
                    raise optuna.TrialPruned() from exc
                fold_scores.append(float('inf'))
        
        return np.mean(fold_scores)
        
    if model_name in SKLEARN_MODELS:
        # Use sklearn API with cross_val_score for tree models
        preprocessor = create_encoder_pipeline(encoder_type, categorical_column)
        model_class = get_model_class(model_name)
        model = model_class(**model_params)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        gkf = GroupKFold(n_splits=10, shuffle=True, random_state=10)
        
        scores = cross_val_score(
            pipeline, X, y, 
            cv=gkf, 
            groups=groups, 
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        
        return -scores.mean() 
    
def optimize_encoder(X, y, groups, categorical_column, encoder_type, model_name, config, n_trials, study_name, storage_url):
    """Optimize hyperparameters for a specific encoder and model combination"""
    
    # Create study with database storage
    study = optuna.create_study(
        direction='minimize',
        study_name=f"{study_name}_{model_name}_{encoder_type}",
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config['global']['seed'])
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, groups, categorical_column, encoder_type, model_name, config),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    return study

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization with Grouped KFold')
    
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['rf', 'extratrees', 'lightgbm', 'xgboost'],
        required=True,
        help='Model to optimize (rf, extratrees, lightgbm, xgboost)'
    )
    
    parser.add_argument(
        '--encoders',
        type=str,
        nargs='+',
        default=['onehot', 'target', 'catboost'],
        choices=['onehot', 'target', 'catboost'],
        help='Encoders to compare (default: all)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of trials per encoder (default: 50)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='hyperparameters.yaml',
        help='Path to hyperparameter config file (default: hyperparameters.yaml)'
    )
    
    parser.add_argument(
        '--categorical-column',
        type=str,
        default='category',
        help='Name of categorical column (default: category)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to data files (if not provided, uses example data)'
    )
    
    return parser.parse_args()