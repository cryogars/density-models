
"""Module providing a Grouped Kfold hyperparameter tuning."""

import logging
import warnings
from datetime import datetime
import optuna
import yaml
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, make_scorer
import category_encoders as ce

warnings.filterwarnings('ignore')

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