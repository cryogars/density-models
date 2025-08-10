
"""Module providing a Grouped Kfold hyperparameter tuning."""

import logging
import warnings
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Tuple, Optional, Dict, List
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

# Model configuration
BOOSTING_MODELS = ['lightgbm', 'xgboost']
SKLEARN_MODELS = ['rf', 'extratrees']

# Global seed will be loaded from config
GLOBAL_SEED = None
XGB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainingData:
    """Container for training and validation data"""
    x_train: Any
    y_train: Any
    x_val: Optional[Any] = None
    y_val: Optional[Any] = None
    groups_train: Optional[Any] = None
    groups_val: Optional[Any] = None

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
    