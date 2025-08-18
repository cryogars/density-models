
"""Module providing a Grouped Kfold CV/ Spatial validation hyperparameter tuning for snow density models."""

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

def load_config(config_path: str = "hyperparameters.yaml") -> Dict[str, Any]:
    """Load hyperparameter configuration from YAML file and set global seed"""
    global GLOBAL_SEED

    try:
        with open(config_path, mode='r', encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Set global seed from config
        GLOBAL_SEED = config.get('global', {}).get('seed', 42)

        return config
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as e:
        raise ValueError("Error parsing YAML config") from e

def load_data(data_path: str) -> DataSplits:
    """Load data splits from pickle file"""
    logger = logging.getLogger(__name__)
    logger.info("Loading data from %s", data_path)

    with open(data_path, 'rb') as f:
        data_splits = pickle.load(f)

    # Create DataSplits object
    splits = DataSplits(
        x_train=data_splits.X_train,
        x_val=data_splits.X_val,
        x_temp=data_splits.X_temp,
        y_train=data_splits.y_train,
        y_val=data_splits.y_val,
        y_temp=data_splits.y_temp,
        x_test=getattr(data_splits, 'X_test', None),
        y_test=getattr(data_splits, 'y_test', None)
    )

    logger.info("Data loaded successfully:")
    logger.info("  X_train shape: %s", splits.x_train.shape)
    logger.info("  X_val shape: %s", splits.x_val.shape)
    logger.info("  X_temp shape: %s", splits.x_temp.shape)

    return splits


def prepare_training_data(
    data_splits: DataSplits,
    model_variant: ModelVariant,
    eval_method: str = "cv"
) -> TrainingData:
    """Prepare training data based on evaluation method and model variant"""
    feature_config = model_variant.feature_config

    if eval_method == "cv":
        # Use combined data for cross-validation
        x = feature_config.select_features(data_splits.X_temp)
        y = data_splits.y_temp
        groups = data_splits.X_temp[feature_config.group_column]

        return TrainingData(x=x, y=y, groups=groups)

    if eval_method == "validation":
        # Use pre-split data
        x_train = feature_config.select_features(data_splits.X_train)
        x_val = feature_config.select_features(data_splits.X_val)
        y_train = data_splits.y_train
        y_val = data_splits.y_val

        return TrainingData(
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val
        )

    raise ValueError(f"Unknown evaluation method: {eval_method}")


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
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'val')],
            callbacks=[early_stopping],
            verbose_eval=False  # Silent training
        )

        # Predict on validation set
        y_pred = model.predict(dval)

    else:
        raise ValueError(f"Native API not supported for model: {model_config.name}")

    return model, y_pred

def train_sklearn_cv(training_config: TrainingConfig) -> float:
    """Train sklearn model with cross-validation"""
    data = training_config.data
    model_config = training_config.model
    encoder_config = training_config.encoder
    cv_config = training_config.cv_config

    # Create preprocessor and model
    preprocessor = encoder_config.create_preprocessor()
    model = model_config.get_model_instance()

    # Create pipeline
    if preprocessor == 'passthrough':
        pipeline = model
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

    # Perform cross-validation
    gkf = cv_config.get_group_kfold()
    scores = cross_val_score(
        pipeline, data.x, data.y,
        cv=gkf,
        groups=data.groups,
        scoring="neg_root_mean_squared_error",
        n_jobs=training_config.n_jobs
    )

    return -scores.mean()

def train_sklearn_validation(training_config: TrainingConfig) -> float:
    """Train sklearn model with validation set"""
    data = training_config.data
    model_config = training_config.model
    encoder_config = training_config.encoder

    # Create preprocessor and model
    preprocessor = encoder_config.create_preprocessor()
    model = model_config.get_model_instance()

    # Fit preprocessor and transform data
    if preprocessor == 'passthrough':
        x_train_processed = data.x_train
        x_val_processed = data.x_val
    else:
        x_train_processed = preprocessor.fit_transform(data.x_train, data.y_train)
        x_val_processed = preprocessor.transform(data.x_val)

    # Train model
    model.fit(x_train_processed, data.y_train)

    # Predict and evaluate
    y_pred = model.predict(x_val_processed)
    rmse = root_mean_squared_error(data.y_val, y_pred)

    return rmse


def train_boosting_cv(training_config: TrainingConfig) -> float:
    """Train boosting model with manual cross-validation"""
    data = training_config.data
    cv_config = training_config.cv_config
    encoder_config = training_config.encoder

    gkf = cv_config.get_group_kfold()
    fold_scores = []

    preprocessor = encoder_config.create_preprocessor()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(data.x, data.y, data.groups)):
        x_train_fold = data.x.iloc[train_idx]
        x_val_fold = data.x.iloc[val_idx]
        y_train_fold = data.y.iloc[train_idx]
        y_val_fold = data.y.iloc[val_idx]

        # Transform data
        if preprocessor == 'passthrough':
            x_train_processed = x_train_fold
            x_val_processed = x_val_fold
        else:
            x_train_processed = preprocessor.fit_transform(x_train_fold, y_train_fold)
            x_val_processed = preprocessor.transform(x_val_fold)

        # Update training data in config
        training_config.data.x_train = x_train_processed
        training_config.data.y_train = y_train_fold
        training_config.data.x_val = x_val_processed
        training_config.data.y_val = y_val_fold

        try:
            # Train with native API
            model, y_pred = train_with_native_api(training_config)

            # Calculate RMSE
            rmse = root_mean_squared_error(y_val_fold, y_pred)
            fold_scores.append(rmse)

        except Exception as exc:
            # If training fails, return a bad score
            logging.warning("Training failed for fold %s: %s", fold, exc)
            fold_scores.append(float('inf'))

    return np.mean(fold_scores)

def train_boosting_validation(training_config: TrainingConfig) -> float:
    """Train boosting model with validation set"""
    data = training_config.data
    encoder_config = training_config.encoder

    # Create preprocessor
    preprocessor = encoder_config.create_preprocessor()

    # Transform data
    if preprocessor == 'passthrough':
        x_train_processed = data.x_train
        x_val_processed = data.x_val
    else:
        x_train_processed = preprocessor.fit_transform(data.x_train, data.y_train)
        x_val_processed = preprocessor.transform(data.x_val)

    # Update data for training
    training_config.data.x_train = x_train_processed
    training_config.data.x_val = x_val_processed

    # Train with native API
    model, y_pred = train_with_native_api(training_config)

    # Calculate RMSE
    rmse = root_mean_squared_error(data.y_val, y_pred)

    return rmse

def objective(
    trial: optuna.Trial,
    training_config: TrainingConfig,
    model_name: str,
    config: Dict[str, Any]
) -> float:
    """Objective function for Optuna optimization"""

    # Suggest hyperparameters
    model_params = suggest_hyperparameters(trial, model_name, config)

    # Update model config with suggested params
    training_config.model = create_model_config(model_name, model_params)

    # Train and evaluate based on model type and eval method
    if model_name in SKLEARN_MODELS:
        if training_config.eval_method == "cv":
            score = train_sklearn_cv(training_config)
        else:
            score = train_sklearn_validation(training_config)
    elif model_name in BOOSTING_MODELS:
        if training_config.eval_method == "cv":
            score = train_boosting_cv(training_config)
        else:
            score = train_boosting_validation(training_config)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Report intermediate score for pruning
    trial.report(score, 0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return score

def optimize_configuration(
    data_splits: DataSplits,
    model_variant: ModelVariant,
    model_name: str,
    encoder_type: str,
    eval_method: str,
    config: Dict[str, Any],
    opt_config: OptimizationConfig
) -> ExperimentResults:
    """Optimize hyperparameters for a specific configuration"""
    logger = logging.getLogger(__name__)

    # Prepare training data
    training_data = prepare_training_data(data_splits, model_variant, eval_method)

    # Create encoder config
    encoder_config = EncoderConfig(
        encoder_type=encoder_type,
        categorical_columns=model_variant.feature_config.categorical_features
    )

    # Create CV config
    cv_config = CrossValidationConfig(n_splits=10, shuffle=True, random_state=GLOBAL_SEED)

    # Create training config
    training_config = TrainingConfig(
        data=training_data,
        model=ModelConfig(name=model_name, params={}),  # Will be updated in objective
        encoder=encoder_config,
        cv_config=cv_config,
        model_variant=model_variant,
        eval_method=eval_method,
        global_config=config['global']
    )

    # Create study
    study_suffix = f"{model_variant.name}_{encoder_type}_{eval_method}"
    study = opt_config.create_study(suffix=study_suffix)


    logger.info(
        "Optimizing: %s - %s variant - %s encoder - %s",
        model_name, model_variant.name, encoder_type, eval_method
    )
    logger.info("  Study name: %s", study.study_name)
    logger.info("  Features: %s", model_variant.feature_config.all_features)
    logger.info("  Evaluation: %s", eval_method)

    # Optimize
    study.optimize(
        lambda trial: objective(trial, training_config, model_name, config),
        n_trials=opt_config.n_trials,
        show_progress_bar=True
    )

    # Create results
    results = ExperimentResults(
        model_name=model_name,
        encoder_type=encoder_type,
        model_variant=model_variant.name,
        eval_method=eval_method,
        best_params=study.best_params,
        best_score=study.best_value,
        n_trials=opt_config.n_trials,
        feature_config=model_variant.feature_config,
        study=study,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    logger.info("  Best RMSE: %.4f", results.best_score)

    return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Hyperparameter optimization with Grouped KFold CV or validation set'
    )

    parser.add_argument(
        '--model', 
        type=str,
        choices=['rf', 'extratrees', 'lightgbm', 'xgboost'],
        required=True,
        help='Model to optimize (rf, extratrees, lightgbm, xgboost)'
    )

    parser.add_argument(
        '--model-variants',
        type=str,
        nargs='+',
        default=['main', 'climate_7d', 'climate_14d'],
        choices=['main', 'climate_7d', 'climate_14d'],
        help='Model variants to test (default: all)'
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
        '--eval-methods',
        type=str,
        nargs='+',
        default=['cv'],
        choices=['cv', 'validation'],
        help='Evaluation methods (default: cv)'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of trials per configuration (default: 50)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='hyperparameters.yaml',
        help='Path to hyperparameter config file (default: hyperparameters.yaml)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='../data/data_splits.pkl',
        help='Path to pickle file with data splits (default: ../data/data_splits.pkl)'
    )

    parser.add_argument(
        '--study-name',
        type=str,
        default=None,
        help='Base name for Optuna study (default: {model} name)'
    )

    parser.add_argument(
        '--storage-url',
        type=str,
        default='sqlite:///optuna_studies.db',
        help='Database URL for Optuna study storage (default: sqlite:///optuna_studies.db)'
    )

    return parser.parse_args()

def main():
    """Main execution function"""

    args = parse_arguments()
    logger, timestamp = setup_logging()

    # Load Config
    config = load_config(args.config)
    logger.info("Loaded config from %s", args.config)
    logger.info("Global seed set to: %s", GLOBAL_SEED)

    # Load data
    data_splits = load_data(args.data_path)

    # Get model variants
    all_variants = get_model_variants()
    selected_variants = {k: v for k, v in all_variants.items() if k in args.model_variants}

    # Set default study name if not provided
    if args.study_name is None:
        args.study_name = args.model

    # Create optimization configuration
    opt_config = OptimizationConfig(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage_url=args.storage_url
    )

    # Run experiments
    all_results = []
    total_experiments = (
        len(selected_variants) * len(args.encoders) * len(args.eval_methods)
    )

    logger.info("Running %s optimization experiments", total_experiments)
    logger.info("Model variants: %s", list(selected_variants.keys()))
    logger.info("Encoders: %s", args.encoders)
    logger.info("Evaluation methods: %s", args.eval_methods)
    logger.info("Model: %s", args.model)
    logger.info("="*60)

    experiment_count = 0
    for variant_name, variant in selected_variants.items():
        for encoder_type in args.encoders:
            for eval_method in args.eval_methods:
                experiment_count += 1
                logger.info("\nExperiment %s/%s", experiment_count, total_experiments)
                logger.info("-"*40)

                try:
                    results = optimize_configuration(
                        data_splits=data_splits,
                        model_variant=variant,
                        model_name=args.model,
                        encoder_type=encoder_type,
                        eval_method=eval_method,
                        config=config,
                        opt_config=opt_config
                    )
                    all_results.append(results)

                except Exception as e:
                    logger.error("Experiment failed: %s", e)
                    continue

    # Analyze and save results
    if all_results:
        logger.info("\n%s", "="*60)
        logger.info("EXPERIMENT RESULTS SUMMARY")
        logger.info("="*60)

        # Sort results by score
        sorted_results = sorted(all_results, key=lambda x: x.best_score)

        # Display top 10 results
        logger.info("\nTop 10 Configurations:")
        logger.info("-"*40)
        for i, result in enumerate(sorted_results[:10], start=1):
            logger.info("%2d. %-12s | %-10s | %-10s | RMSE: %.4f",
            i, result.model_variant, result.encoder_type, result.eval_method, result.best_score)

        # Best overall result
        best_result = sorted_results[0]
        logger.info("\n%s", "="*60)
        logger.info("BEST CONFIGURATION")
        logger.info("="*60)
        logger.info("Model Variant: %s", best_result.model_variant)
        logger.info("Model Type:  %s", best_result.model_name)
        logger.info("Encoder:  %s", best_result.encoder_type)
        logger.info("Evaluation:  %s", best_result.eval_method)
        logger.info("Best RMSE: %.4f", best_result.best_score)
        logger.info("Features: %s", best_result.feature_config.all_features)

        logger.info("\nBest Hyperparameters:")
        for param, value in best_result.best_params.items():
            if isinstance(value, float):
                logger.info("  %s: %.6f", param, value)
            else:
                logger.info("  %s: %s", param, value)

        # Compare model variants
        logger.info("\n%s", "="*60)
        logger.info("MODEL VARIANT COMPARISON")
        logger.info("="*60)

        for variant_name in selected_variants.keys():
            variant_results = [r for r in all_results if r.model_variant == variant_name]
            if variant_results:
                best_variant_result = min(variant_results, key=lambda x: x.best_score)
                avg_score = np.mean([r.best_score for r in variant_results])
                logger.info("%-12s: Best RMSE = %.4f, Avg RMSE = %.4f",
                            variant_name, best_variant_result.best_score, avg_score)

        # Compare evaluation methods
        logger.info("\n%s", "="*60)
        logger.info("EVALUATION METHOD COMPARISON")
        logger.info("="*60)

        for eval_method in args.eval_methods:
            method_results = [r for r in all_results if r.eval_method == eval_method]
            if method_results:
                best_method_result = min(method_results, key=lambda x: x.best_score)
                avg_score = np.mean([r.best_score for r in method_results])
                logger.info("%-10s: Best RMSE = %.4f, Avg RMSE = %.4f",
                            eval_method, best_method_result.best_score, avg_score)

        # Compare encoders
        logger.info("\n%s", "="*60)
        logger.info("ENCODER COMPARISON")
        logger.info("="*60)

        for encoder in args.encoders:
            encoder_results = [r for r in all_results if r.encoder_type == encoder]
            if encoder_results:
                best_encoder_result = min(encoder_results, key=lambda x: x.best_score)
                avg_score = np.mean([r.best_score for r in encoder_results])
                logger.info("%-10s: Best RMSE = %.4f, Avg RMSE = %.4f",
                            encoder, best_encoder_result.best_score, avg_score)

        # Save detailed results to YAML
        save_results_to_yaml(all_results, args.model, timestamp, logger)

        # Save best configuration for production use
        save_best_config(best_result, args.model, timestamp, logger)

    else:
        logger.error("No successful experiments completed!")

    return all_results

def save_results_to_yaml(results: List[ExperimentResults], model_name: str,
                         timestamp: str, logger: logging.Logger):
    """Save all results to YAML file"""
    results_filename = f"optimization_results_{model_name}_{timestamp}.yaml"

    results_dict = {
        'model': model_name,
        'timestamp': timestamp,
        'global_seed': GLOBAL_SEED,
        'total_experiments': len(results),
        'results': {}
    }

    for result in results:
        key = f"{result.model_variant}_{result.encoder_type}_{result.eval_method}"
        results_dict['results'][key] = {
            'model_variant': result.model_variant,
            'encoder_type': result.encoder_type,
            'eval_method': result.eval_method,
            'score': float(result.best_score),
            'n_trials': result.n_trials,
            'features': {
                'numeric': result.feature_config.numeric_features,
                'categorical': result.feature_config.categorical_features
            },
            'params': {k: float(v) if isinstance(v, np.floating) else v 
                      for k, v in result.best_params.items()}
        }

    with open(results_filename, 'w', encoding='utf-8') as f:
        yaml.dump(results_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"\nDetailed results saved to: {results_filename}")

def save_best_config(best_result: ExperimentResults, model_name: str,
                    timestamp: str, logger: logging.Logger):
    """Save best configuration for production use"""
    config_filename = f"best_config_{model_name}_{timestamp}.yaml"

    best_config = {
        'model': {
            'type': best_result.model_name,
            'variant': best_result.model_variant,
            'encoder': best_result.encoder_type,
            'eval_method': best_result.eval_method
        },
        'features': {
            'numeric': best_result.feature_config.numeric_features,
            'categorical': best_result.feature_config.categorical_features,
            'group_column': best_result.feature_config.group_column
        },
        'performance': {
            'rmse': float(best_result.best_score),
            'n_trials': best_result.n_trials
        },
        'hyperparameters': {k: float(v) if isinstance(v, np.floating) else v 
                           for k, v in best_result.best_params.items()},
        'metadata': {
            'timestamp': timestamp,
            'seed': GLOBAL_SEED
        }
    }

    with open(config_filename, 'w', encoding='utf-8') as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Best configuration saved to: {config_filename}")
    logger.info("This file can be used to train the final model for production")
