
"""Module providing a Grouped Kfold hyperparameter tuning for snow density models."""

import pickle
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Tuple, Optional, Dict, List, Union
import yaml
import torch
import optuna
import numpy as np
import pandas as pd
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

# Model configuration
BOOSTING_MODELS = ['lightgbm', 'xgboost']
SKLEARN_MODELS = ['rf', 'extratrees']

# Global seed will be loaded from config
GLOBAL_SEED = None
XGB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DataSplits:
    """Container for all data splits"""
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_temp: pd.DataFrame  # Combined train+val for CV
    y_train: pd.Series
    y_val: pd.Series
    y_temp: pd.Series  # Combined train+val for CV
    x_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.Series] = None


@dataclass
class FeatureConfig:
    """Configuration for feature selection"""
    numeric_features: List[str]
    categorical_features: List[str]
    group_column: str = "Station_Name"

    @property
    def all_features(self) -> List[str]:
        """Get all feature columns"""
        return self.numeric_features + self.categorical_features

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only the configured features from dataframe"""
        return df.filter(items=self.all_features)


@dataclass
class ModelVariant:
    """Configuration for a model variant"""
    name: str
    feature_config: FeatureConfig
    description: str = ""


@dataclass
class TrainingData:
    """Container for training and validation data"""
    # For CV approach - full dataset
    x: Optional[pd.DataFrame] = None
    y: Optional[pd.Series] = None
    groups: Optional[pd.Series] = None

    # For train/val split approach
    x_train: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    x_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None
    groups_train: Optional[pd.Series] = None
    groups_val: Optional[pd.Series] = None


@dataclass
class ModelConfig:
    """Model configuration with name and parameters"""
    name: str
    params: Dict[str, Any]
    model_class: Optional[Any] = None

    def get_model_instance(self):
        """Create and return a model instance"""
        if self.model_class:
            return self.model_class(**self.params)
        return None


@dataclass
class EncoderConfig:
    """Encoder configuration"""
    encoder_type: str
    categorical_columns: List[str]
    preprocessor: Optional[Any] = None

    def create_preprocessor(self):
        """Create and return the preprocessor pipeline"""
        if not self.categorical_columns:
            # No categorical columns, return passthrough
            self.preprocessor = 'passthrough'
            return self.preprocessor

        if self.encoder_type == 'onehot':
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        elif self.encoder_type == 'target':
            encoder = TargetEncoder(
                target_type='continuous',
                cv=5,
                random_state=GLOBAL_SEED,
                smooth="auto"
            )
        elif self.encoder_type == 'catboost':
            encoder = ce.CatBoostEncoder(random_state=GLOBAL_SEED)
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

        self.preprocessor = ColumnTransformer(
            transformers=[('cat', encoder, self.categorical_columns)],
            remainder='passthrough'
        )
        return self.preprocessor


@dataclass
class CrossValidationConfig:
    """Cross-validation configuration"""
    n_splits: int = 10
    shuffle: bool = True
    random_state: Optional[int] = None

    def get_group_kfold(self):
        """Return GroupKFold instance"""
        return GroupKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state or GLOBAL_SEED
        )


@dataclass
class OptimizationConfig:
    """Optimization configuration for Optuna"""
    n_trials: int = 100
    study_name: str = "optimization_study"
    storage_url: Optional[str] = None
    direction: str = "minimize"
    sampler: Optional[Any] = None

    def create_study(self, suffix: str = ""):
        """Create Optuna study"""
        study_name = f"{self.study_name}_{suffix}" if suffix else self.study_name

        if self.sampler is None:
            self.sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)

        return optuna.create_study(
            direction=self.direction,
            study_name=study_name,
            storage=self.storage_url,
            load_if_exists=True,
            sampler=self.sampler
        )


@dataclass
class TrainingConfig:
    """Complete training configuration"""
    data: TrainingData
    model: ModelConfig
    encoder: EncoderConfig
    cv_config: CrossValidationConfig
    model_variant: ModelVariant
    eval_method: str = "cv"  # "cv" or "validation"
    global_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def early_stopping_rounds(self) -> int:
        """Get early stopping rounds from global config"""
        return self.global_config.get('early_stopping_rounds', 50)

    @property
    def n_jobs(self) -> int:
        """Get number of jobs from global config"""
        return self.global_config.get('n_jobs', -1)

    @property
    def verbosity(self) -> int:
        """Get verbosity level from global config"""
        return self.global_config.get('verbosity', -1)


@dataclass
class ExperimentResults:
    """Container for experiment results"""
    model_name: str
    encoder_type: str
    model_variant: str
    eval_method: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    feature_config: FeatureConfig
    study: Optional[Any] = None
    timestamp: Optional[str] = None


def get_model_variants() -> Dict[str, ModelVariant]:
    """Define all model variants with their feature configurations"""
    variants = {
        'main': ModelVariant(
            name='main',
            feature_config=FeatureConfig(
                numeric_features=['Elevation', 'Snow_Depth', 'DOWY'],
                categorical_features=['Snow_Class'],
                group_column='Station_Name'
            ),
            description='Base model with core features'
        ),
        'climate_7d': ModelVariant(
            name='climate_7d',
            feature_config=FeatureConfig(
                numeric_features=['Elevation', 'Snow_Depth', 'DOWY',
                                'PRECIPITATION_lag_7d', 'TAVG_lag_7d'],
                categorical_features=['Snow_Class'],
                group_column='Station_Name'
            ),
            description='Climate-enhanced model with 7-day lags'
        ),
        'climate_14d': ModelVariant(
            name='climate_14d',
            feature_config=FeatureConfig(
                numeric_features=['Elevation', 'Snow_Depth', 'DOWY',
                                'PRECIPITATION_lag_14d', 'TAVG_lag_14d'],
                categorical_features=['Snow_Class'],
                group_column='Station_Name'
            ),
            description='Climate-enhanced model with 14-day lags'
        )
    }
    return variants

def load_config(config_path="hyperparameters.yaml") -> Dict[str, Any]:
    """Load hyperparameter configuration from YAML file and set global seed"""
    global GLOBAL_SEED

    try:
        with open(file=config_path, mode='r', encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Set global seed from config
        GLOBAL_SEED = config.get('global', {}).get('seed', 42)

        # Set seeds for reproducibility
        np.random.seed(GLOBAL_SEED)
        if torch.cuda.is_available():
            torch.manual_seed(GLOBAL_SEED)
            torch.cuda.manual_seed_all(GLOBAL_SEED)

        return config
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as e:
        raise ValueError("Error parsing YAML config") from e


def get_model_class(model_name: str):
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


def create_model_config(model_name: str, params: Dict[str, Any]) -> ModelConfig:
    """Create ModelConfig instance with appropriate model class"""
    model_class = get_model_class(model_name)
    return ModelConfig(name=model_name, params=params, model_class=model_class)

def suggest_hyperparameters(
        trial: optuna.Trial,
        model_name: str,
        config: Dict[str, Any]
) -> Dict[str, Any]:
    """Suggest hyperparameters based on config"""
    model_config = config['models'][model_name]
    global_config = config['global']
    params = {}

    # Suggest model-specific hyperparameters
    for param_name, param_config in model_config.items():
        if param_config['type'] == 'int':
            if 'step' in param_config:
                params[param_name] = trial.suggest_int(
                    name=param_name,
                    low=param_config['low'],
                    high=param_config['high'],
                    step=param_config['step']
                )
            else:
                params[param_name] = trial.suggest_int(
                    name=param_name,
                    low=param_config['low'],
                    high=param_config['high']
                )

        elif param_config['type'] == 'float':
            log_scale = param_config.get('log', False)
            params[param_name] = trial.suggest_float(
                name=param_name,
                low=param_config['low'],
                high=param_config['high'],
                log=log_scale
            )

        elif param_config['type'] == 'categorical':
            params[param_name] = trial.suggest_categorical(
                name=param_name,
                choices=param_config['choices']
            )

    # Add global and model-specific fixed parameters
    if model_name in SKLEARN_MODELS:
        params.update({
            'random_state': GLOBAL_SEED,
            'n_jobs': global_config['n_jobs']
        })

    elif model_name == 'lightgbm':
        params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'seed': GLOBAL_SEED,
            'verbosity': global_config['verbosity'],
            'force_col_wise': True,
            'deterministic': True
        })

    elif model_name == 'xgboost':
        params.update({
            'objective': 'reg:squarederror',
            'seed': GLOBAL_SEED,
            'tree_method': 'hist',
            'device': XGB_DEVICE,
            'verbosity': 0,  # XGBoost uses 0 for silent
            'sampling_method': 'gradient_based',
        })

    return params

def train_with_native_api(training_config: TrainingConfig) -> Tuple[Any, np.ndarray]:
    """Train model using native API with validation set and early stopping"""

    data = training_config.data
    model_config = training_config.model
    early_stopping_rounds = training_config.early_stopping_rounds

    # Extract num_boost_round from params if present
    params = model_config.params.copy()
    num_boost_round = params.pop('num_boost_round', 1000)

    if model_config.name == 'lightgbm':
        # Create LightGBM datasets
        train_data = lgb.Dataset(data.x_train, label=data.y_train)
        val_data = lgb.Dataset(data.x_val, label=data.y_val, reference=train_data)

        # Train with early stopping
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
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
            data_name='val',
            maximize=False,
            save_best=True
        )

        # Train with early stopping
        model = xgb.train(
            params,
            dtrain,
            num_rounds=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'val')],
            callbacks=[early_stopping],
            verbose_eval=False  # Silent training
        )

        # Predict on validation set
        y_pred = model.predict(dval)

    else:
        raise ValueError(f"Native API not supported for model: {model_config.name}")

    return model, y_pred

def train_sklearn_model(
        training_config: TrainingConfig,
        x: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series
) -> float:
    """Train sklearn model with cross-validation"""
    model_config = training_config.model
    encoder_config = training_config.encoder
    cv_config = training_config.cv_config

    # Create preprocessor and model
    preprocessor = encoder_config.create_preprocessor()
    model = model_config.get_model_instance()

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Perform cross-validation
    gkf = cv_config.get_group_kfold()
    scores = cross_val_score(
        pipeline, x, y,
        cv=gkf,
        groups=groups,
        scoring="neg_root_mean_squared_error",
        n_jobs=training_config.n_jobs
    )

    return -scores.mean()
