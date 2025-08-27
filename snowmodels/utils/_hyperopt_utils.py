
"""
Module providing hyperparameter tuning for ML density models.
"""
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, assert_never, Literal, Dict, Any
import yaml
import torch
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import root_mean_squared_error as rmse, r2_score

# # Model configuration
BOOSTING_MODELS = ['lightgbm', 'xgboost']
SKLEARN_MODELS = ['rf', 'extratrees']

MAIN_FEATURES = [
    'Elevation', 'Snow_Depth', 'DOWY',
    'Latitude', 'Longitude'
]
CLIMATE_7_FEATURES = ['PRECIPITATION_lag_7d', 'TAVG_lag_7d']
CLIMATE_14_FEATURES = ['PRECIPITATION_lag_14d', 'TAVG_lag_14d']
NOMINAL_FEATURE = ['Snow_Class']
XGB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""

    rmse: float
    r2: float
    mbe: float
    best_iteration: Optional[int] = None


@dataclass(frozen=True)
class GlobalConfig:
    """Container for global config"""

    early_stopping_rounds: int
    verbosity: int
    seed: Optional[int] = None
    n_jobs: Optional[int] = None

@dataclass(frozen=True)
class DataSplits:
    """Container for datasplits"""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    x_train_nogeo: Optional[pd.DataFrame] = None
    x_val_nogeo: Optional[pd.DataFrame] = None


@dataclass
class ParamSpec:
    """Container for hyperparameter spec"""

    type: str
    low: Optional[float | int] = None
    high: Optional[float | int] = None
    step: Optional[int] = None
    log: bool = False


@dataclass
class ModelConfig:
    """
        Dataclass wrapper for model-specific hyperparameter configurations.

        Provides dict-like access (`__getitem__`, `.items()`, etc.) while still
        maintaining type safety with ParamSpec objects.

        Example:
        --------
        >>> model_cfg = config.models["rf"]
        >>> n_estimators = model_cfg["n_estimators"]
        >>> print(n_estimators.low, n_estimators.high)

        Attributes
        ----------
        params : Dict[str, ParamSpec]
            Mapping of hyperparameter names to their specifications.
    """
    params: Dict[str, ParamSpec] = field(default_factory=dict)

    def __getitem__(self, key: str) -> ParamSpec:
        """Allow dict-like access: model_cfg['n_estimators'] → ParamSpec."""
        return self.params[key]

    def __iter__(self):
        """Iterate over parameter names."""
        return iter(self.params)

    def items(self):
        """Return (name, ParamSpec) pairs."""
        return self.params.items()

    def keys(self):
        """Return parameter names."""
        return self.params.keys()

    def values(self):
        """Return ParamSpec objects only."""
        return self.params.values()

@dataclass
class HyperparamConfig:
    """Container for configurations"""

    global_cfg: GlobalConfig
    models: Dict[str, ModelConfig] = field(default_factory=dict)

@dataclass
class FinalModelConfig:
    """Container for final model configuration"""
    name: str
    variant: str
    encoder: str
    mode: str


@dataclass
class InlineDefaultConfig:
    """Container for inline default config"""

    final_model_config: FinalModelConfig
    timestamp: str
    folder: Path

@dataclass
class InlineTuneConfig:
    """Container for inline tuning config"""

    final_model_config: FinalModelConfig
    timestamp: str
    folder: Path
    storage_url: str
    n_trials: int


@dataclass
class PerformanceConfig:
    """Container for performance configuration"""
    metrics: EvaluationMetrics
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentResults:
    """Container for experiment results"""
    model: FinalModelConfig
    performance: PerformanceConfig
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Objective:
    """Optuna Objective for model tuning.

    Returns
    -------
    float
        RMSE score for the given trial (to be minimized).
    """

    data: DataSplits
    model_name: str
    config: HyperparamConfig

    def __call__(self, trial: optuna.Trial):

        """Run training and evaluation for a single Optuna trial."""

        # Suggest trial-specific hyperparameters
        params = suggest_hyperparameters(
            trial=trial, model_name=self.model_name,
            config=self.config
        )

        # Train + evaluate
        results = train_and_evaluate_models(
            model_name=self.model_name, params=params,
            cfg=self.config.global_cfg,
            data=self.data
        )

        # Report intermediate objective value for pruning
        trial.report(results.metrics.rmse, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Attach useful metrics to trial for later inspection
        trial.set_user_attr("r2", results.metrics.r2)
        trial.set_user_attr("mbe", results.metrics.mbe)
        best_iter = getattr(results.metrics, "best_iteration")
        if best_iter is not None:
            trial.set_user_attr('best_iteration', best_iter)

        return results.metrics.rmse


@dataclass(frozen=True)
class DefaultTrainer:
    """A class for default tuning"""
    cfg: GlobalConfig
    data: DataSplits
    model: Literal['rf', 'extratrees', 'lightgbm', 'xgboost']

    def __post_init__(self):
        """Validate the model after initialization"""

        try:
            if self.model in ['rf', 'extratrees', 'lightgbm', 'xgboost']:
                pass
            else:
                assert_never(self.model)
        except AssertionError as ase:
            raise ValueError(
                f"{self.model} not supported. Choose from {SKLEARN_MODELS + BOOSTING_MODELS}"
            ) from ase

    def default_params(self) -> Dict[str, Any]:
        """Get default params"""
        if self.model in SKLEARN_MODELS:
            return {
                'random_state': self.cfg.seed,
                'n_jobs': self.cfg.n_jobs
            }

        if self.model == 'lightgbm':
            return {
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': self.cfg.seed,
                "force_col_wise": True,
                "verbosity": self.cfg.verbosity
            }

        if self.model == 'xgboost':
            return {
                'objective': 'reg:squarederror',
                'random_state': self.cfg.seed,
                "device": XGB_DEVICE
            }

        return None


    def default_results(self) -> PerformanceConfig:

        """Train and evaluate ml models using default settings"""

        return train_and_evaluate_models(
            model_name=self.model,
            params=self.default_params(),
            cfg=self.cfg, data=self.data
        )

def compute_metrics_for_logging(y_true, y_pred, best_iteration=None) -> EvaluationMetrics:
    """A function that computes RMSE, R², and MBE"""

    return EvaluationMetrics(
        rmse = rmse(y_true=y_true, y_pred=y_pred),
        r2 = r2_score(y_true=y_true, y_pred=y_pred),
        mbe = float(np.mean(y_pred - y_true)),
        best_iteration = best_iteration
    )


def train_and_evaluate_models(
        model_name: Literal['rf', 'extratrees', 'lightgbm', 'xgboost'],
        params: Dict[str, Any], cfg: GlobalConfig, data: DataSplits
) -> PerformanceConfig:
    """A container to train all models"""

    if model_name == 'extratrees':
        model = ExtraTreesRegressor(**params)
        model.fit(
            data.x_train, data.y_train
        )
        y_pred = model.predict(data.x_val)

        return PerformanceConfig(
            metrics=compute_metrics_for_logging(y_true=data.y_val, y_pred=y_pred),
            hyperparameters=params
        )

    if model_name == 'rf':

        model = RandomForestRegressor(**params)
        model.fit(
            data.x_train, data.y_train
        )

        y_pred = model.predict(data.x_val)

        return PerformanceConfig(
            metrics=compute_metrics_for_logging(y_true=data.y_val, y_pred=y_pred),
            hyperparameters=params
        )

    if model_name == 'lightgbm':
        train_data = lgb.Dataset(data.x_train, label=data.y_train)
        val_data = lgb.Dataset(data.x_val, label=data.y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=cfg.early_stopping_rounds, verbose=False
                ),
                lgb.log_evaluation(period=0)
            ],
            num_boost_round=1500
        )

        y_pred = model.predict(data.x_val, num_iteration=model.best_iteration)

        return PerformanceConfig(
            metrics=compute_metrics_for_logging(
                y_true=data.y_val, y_pred=y_pred, best_iteration=model.best_iteration
            ),
            hyperparameters=params
        )

    if model_name == 'xgboost':

        logger = logging.getLogger(__name__)
        dtrain = xgb.DMatrix(data.x_train, label=data.y_train)
        dval = xgb.DMatrix(data.x_val, label=data.y_val)
        logger.info("XGBoost will run on %s", XGB_DEVICE.upper())

        early_stopping = xgb.callback.EarlyStopping(
            rounds=cfg.early_stopping_rounds,
            metric_name='rmse',
            data_name='val',
            maximize=False,
            save_best=True
        )

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1500,
            evals=[(dtrain, 'train'), (dval, 'val')],
            callbacks=[early_stopping],
            verbose_eval=False
        )

        y_pred = model.predict(dval)

        return PerformanceConfig(
            metrics=compute_metrics_for_logging(
                y_true=data.y_val, y_pred=y_pred, best_iteration=model.best_iteration
            ),
            hyperparameters=params
        )

    return None


def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s  %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    return logger, timestamp

def ecnoder_preprocessor(
        encoder: Literal['onehot', 'catboost', 'target'],
        cfg: GlobalConfig, data: DataSplits
) -> DataSplits:
    """A function to preprocess using the specified encoder"""
    logger = logging.getLogger(__name__)
    seed = cfg.seed
    logger.info("Encoding Snow Class using %s Encoding", encoder.upper())

    try:
        if encoder == 'onehot':
            snow_class_encoder = OneHotEncoder(
                handle_unknown='error', drop=None, sparse_output=False
            )

        elif encoder == 'catboost':
            snow_class_encoder = CatBoostEncoder(
                handle_unknown='error', random_state = seed
            )

        elif encoder == 'target':
            snow_class_encoder = TargetEncoder(
                target_type = 'continuous', cv=5,
                random_state = seed, smooth = 'auto'
            )

        else:
            assert_never(encoder)
    except AssertionError as ase:
        raise ValueError(
            f"Unknown encoder type: {encoder}. Choose onehot, target, or catboost"
        ) from ase

    x_cat_train = snow_class_encoder.fit_transform(data.x_train.Snow_Class.to_frame(), data.y_train)
    x_cat_test = snow_class_encoder.transform(data.x_val.Snow_Class.to_frame())

    x_train_prep = pd.concat(
        [data.x_train.drop('Snow_Class', axis=1), x_cat_train],
        axis=1
    )
    x_test_prep = pd.concat(
        [data.x_val.drop('Snow_Class', axis=1), x_cat_test],
        axis=1
    )

    splits = DataSplits(
        x_train = x_train_prep,
        x_val = x_test_prep,
        y_train = data.y_train,
        y_val = data.y_val,
        x_train_nogeo = x_train_prep.drop(columns=['Latitude', 'Longitude'], axis=1),
        x_val_nogeo = x_test_prep.drop(columns=['Latitude', 'Longitude'], axis=1)
    )

    logger.info("Snow Class encoding finished!")

    return splits

def model_variant_selector(
        variant: Literal['main', 'climate_7', 'climate_14'],
        data: DataSplits
) -> DataSplits:
    """Select and validate model variant"""
    logger = logging.getLogger(__name__)
    logger.info("Performing feature selection...")

    try:
        if variant == 'main':
            selected_features = MAIN_FEATURES + NOMINAL_FEATURE

        elif variant == 'climate_7d':
            selected_features = MAIN_FEATURES + NOMINAL_FEATURE + CLIMATE_7_FEATURES

        elif variant == 'climate_14d':
            selected_features = MAIN_FEATURES + NOMINAL_FEATURE + CLIMATE_14_FEATURES

        else:
            assert_never(variant)
    except AssertionError as ase:
        raise ValueError(
            f"Unknown model variant: {variant}. Choose main, climate_7, or climate_14"
        ) from ase


    selected_data = DataSplits(
        x_train = data.x_train.filter(items=selected_features),
        x_val = data.x_val.filter(items=selected_features),
        y_train = data.y_train,
        y_val = data.y_val
    )
    return selected_data

def load_data(
        data_path: str, final_eval: bool = True
) -> DataSplits:
    """Load data splits from a pickle file"""

    logger = logging.getLogger(__name__)
    logger.info("Loading data from %s", data_path)

    with open(f'{data_path}', 'rb') as f:
        data_splits = pickle.load(f)

    if final_eval:
        x_train = data_splits.X_temp
        x_test = data_splits.X_test
        y_train = data_splits.y_temp
        y_test = data_splits.y_test
    else:
        x_train = data_splits.X_train
        x_test = data_splits.X_val
        y_train = data_splits.y_train
        y_test = data_splits.y_val

    splits = DataSplits(
        x_train = x_train,
        x_val = x_test,
        y_train = y_train,
        y_val = y_test
    )

    logger.info("Data loaded successfully!")
    logger.info("For tuning? %s", not final_eval)

    return splits

def apply_backend_defaults(model_name: str, global_cfg: GlobalConfig) -> Dict[str, Any]:
    """Add defaults specific to sklearn, lightgbm, or xgboost."""
    if model_name in SKLEARN_MODELS:
        return {
            "random_state": global_cfg.seed,
            "n_jobs": global_cfg.n_jobs,
        }

    if model_name == "lightgbm":
        return {
            "objective": "regression",
            "metric": "rmse",
            "seed": global_cfg.seed,
            "verbosity": global_cfg.verbosity,
            "force_col_wise": True,
            "deterministic": True
        }

    if model_name == "xgboost":
        sampling_method = "gradient_based" if XGB_DEVICE == "cuda" else "uniform"
        return {
            "objective": "reg:squarederror",
            "seed": global_cfg.seed,
            "tree_method": "hist",
            "device": XGB_DEVICE,
            "verbosity": 0,
            "sampling_method": sampling_method
        }

    return {}


def suggest_hyperparameters(
    trial: optuna.Trial,
    model_name: str,
    config: HyperparamConfig
) -> Dict[str, Any]:
    """Suggest hyperparameters for a given model using dataclass config."""
    params = {}

    # Fetch the model's ParamSpecs
    model_params = config.models[model_name]

    for pname, pspec in model_params.items():
        if pspec.type == "int":
            if pspec.step:
                params[pname] = trial.suggest_int(
                    pname, pspec.low, pspec.high, step=pspec.step
                )
            else:
                params[pname] = trial.suggest_int(
                    pname, pspec.low, pspec.high
                )

        elif pspec.type == "float":
            params[pname] = trial.suggest_float(
                pname, float(pspec.low), float(pspec.high), log=pspec.log
            )

        else:
            raise ValueError(
                f"Unknown hyperparameter type '{pspec.type}' "
                f"for parameter '{pname}' in model '{model_name}'"
            )

    # Add global / backend-specific defaults
    params.update(apply_backend_defaults(model_name, config.global_cfg))

    return params

def load_config(config_path: str = "hyperparameters.yaml") -> HyperparamConfig:
    """Load hyperparameter configuration (and global options) from YAML file"""

    try:

        with open(config_path, mode='r', encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)

        # parse global section
        global_config = GlobalConfig(**raw_config['global'])

        # parse models section
        models = {}
        for model_name, params in raw_config["models"].items():
            param_specs = {
                pname: ParamSpec(**pconfig) for pname, pconfig in params.items()
            }
            models[model_name] = ModelConfig(param_specs)

        return HyperparamConfig(global_cfg=global_config, models=models)

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as e:
        raise ValueError("Error parsing YAML config") from e


def create_folders(tune_mode: str) -> None:

    """Create results folder depending on tuning mode and return its Path."""

    folder = Path("results/tune") if tune_mode == "tune" else Path("results/default")
    folder.mkdir(parents=True, exist_ok=True)

    return folder

def parse_arguments():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        description = "Hyperparameter optimization using Optuna and a validation set"
    )

    parser.add_argument(
        '--models', 
        nargs='+',
        type=str,
        default=['extratrees'],
        choices=['rf', 'extratrees', 'lightgbm', 'xgboost'],
        help='Model(s) to optimize'
    )

    parser.add_argument(
        '--variants', type = str,
        nargs = '+',
        default = ['main', 'climate_7d', 'climate_14d'],
        choices = ['main', 'climate_7d', 'climate_14d'],
        help = 'model variants to compare (defaults to all)'
    )

    parser.add_argument(
        '--encoders', type = str,
        nargs = '+',
        default = ['onehot', 'target', 'catboost'],
        choices = ['onehot', 'target', 'catboost'],
        help = 'Encoders to compare (defaults to all)'
    )

    parser.add_argument(
        '--data-path',
        type = str,
        default = "../data/data_splits.pkl",
        help='Path to pickle file with data splits (default: ../data/data_splits.pkl)'
    )

    parser.add_argument(
        '--config-path',
        type=str,
        default="hyperparameters.yaml",
        help='Path to hyperparameter config file (default: hyperparameters.yaml)'
    )

    parser.add_argument(
        '--tuning-mode',
        type=str,
        default = 'default',
        choices = ['default', 'tune'],
        help='Whether to tune or use default configuration'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of trials per configuration (default: 100)'
    )

    parser.add_argument(
        '--storage-url',
        type=str,
        default='sqlite:///optuna_studies.db',
        help='Database URL for Optuna study storage (default: sqlite:///optuna_studies.db)'
    )

    return parser.parse_args()


def run_default_inline(
        global_cfg: GlobalConfig,
        data: DataSplits,
        inline_default: InlineDefaultConfig
) -> ExperimentResults:
    """Function to run default config inline"""

    logger = logging.getLogger(__name__)
    logger.info("Using default configuration")

    model = inline_default.final_model_config.name
    variant = inline_default.final_model_config.variant
    mode = inline_default.final_model_config.mode
    encoder = inline_default.final_model_config.encoder

    filename = f"{model}_{variant}_{mode}_" \
        f"{encoder}_{inline_default.timestamp}.yaml"
    results_filename = inline_default.folder / filename

    results = (
        DefaultTrainer(
            cfg=global_cfg, data=data,
            model=model
        )
        .default_results()
    )

    logger.info("Trainin finished with the following metrics:")
    if results.metrics.best_iteration is not None:
        logger.info(
            "===>>> Best Iteration: %.4f",
            results.metrics.best_iteration
        )
    logger.info("===>>> RMSE: %.4f", results.metrics.rmse)
    logger.info("===>>> R²: %.4f", results.metrics.r2)
    logger.info("===>>> MBE: %.4f\n", results.metrics.mbe)
    logger.info('Saving results to %s.', results_filename)


    experiment_results=asdict(
        ExperimentResults(
            model=inline_default.final_model_config,
            performance=results,
            config={
                'seed': global_cfg.seed,
                'timestamp': inline_default.timestamp
            }
        )
    )

    with open(results_filename, "w", encoding="utf-8") as f:
        yaml.dump(
            experiment_results, f, default_flow_style=False,
            sort_keys=False, allow_unicode=True
        )

def tune_inline(
        data: DataSplits,
        cfg: HyperparamConfig,
        inline_tune: InlineTuneConfig
):
    """Function to tune hyperparameters inline"""
    logger = logging.getLogger(__name__)

    logger.info("Tunning for optimal hyperparametr configuration...")
    logger.info('Number of trials: %s', inline_tune.n_trials)

    model = inline_tune.final_model_config.name
    variant = inline_tune.final_model_config.variant
    encoder = inline_tune.final_model_config.encoder

    filename = f"{model}_{variant}_{inline_tune.final_model_config.mode}_" \
        f"{encoder}_{inline_tune.timestamp}.yaml"
    results_filename = inline_tune.folder / filename

    objective = Objective(
        data=data,
        model_name=model,
        config=cfg
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model}_{variant}_{encoder}",
        storage=inline_tune.storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.global_cfg.seed)
    )

    study.optimize(
        objective, n_trials=inline_tune.n_trials,
        show_progress_bar=True
    )

    evaluation_metrics = EvaluationMetrics(
        rmse=study.best_trial.value,
        r2=study.best_trial.user_attrs['r2'],
        mbe=study.best_trial.user_attrs["mbe"],
        best_iteration=study.best_trial.user_attrs.get("best_iteration", None)
    )

    performance_config = PerformanceConfig(
        metrics=evaluation_metrics,
        hyperparameters=study.best_trial.params
    )

    experiment_results = ExperimentResults(
        model=inline_tune.final_model_config,
        performance=performance_config,
        config={
            'seed': cfg.global_cfg.seed,
            'timestamp': inline_tune.timestamp
        }
    )

    logger.info("Trainin finished with the following metrics:")
    logger.info("===>>> Best RMSE: %.4f", evaluation_metrics.rmse)
    logger.info("===>>> Best R²: %.4f", evaluation_metrics.r2)
    logger.info("===>>> Best MBE: %.4f\n", evaluation_metrics.mbe)
    logger.info('Saving results to %s.', results_filename)

    with open(results_filename, "w", encoding="utf-8") as f:
        yaml.dump(
            asdict(experiment_results), f, default_flow_style=False,
            sort_keys=False, allow_unicode=True
        )

def main():
    """Main execution function"""

    logger, timestamp = setup_logging()
    all_args = parse_arguments()
    raw_data=load_data(data_path=all_args.data_path, final_eval=False)
    all_config = load_config(config_path=all_args.config_path)
    folder = create_folders(all_args.tuning_mode)

    for variant in all_args.variants:

        logger.info("="*59)
        logger.info("Starting data preprocessing for %s model variants.", variant.upper())
        logger.info("="*59)

        selected_variant = model_variant_selector(variant=variant, data=raw_data)

        for encoder in all_args.encoders:

            processed_data=ecnoder_preprocessor(
                encoder=encoder,
                cfg=all_config.global_cfg,
                data=selected_variant
            )

            logger.info(
                "Data preparation for '%s' with '%s' encoder complete!",
                variant.upper(), encoder.upper()
            )
            logger.info("Features: %s", processed_data.x_train.columns.to_list())
            logger.info("  X_train shape: %s", processed_data.x_train.shape)

            for model in all_args.models:
                logger.info("="*20)
                logger.info("Training %s", model.upper())
                logger.info("="*20)

                final_model_config = FinalModelConfig(
                    name=model,
                    encoder=encoder,
                    mode=all_args.tuning_mode,
                    variant=variant
                )

                if all_args.tuning_mode == "default":
                    inline_default = InlineDefaultConfig(
                        final_model_config=final_model_config,
                        timestamp=timestamp,
                        folder=folder
                    )

                    run_default_inline(
                        global_cfg=all_config.global_cfg,
                        data=processed_data, 
                        inline_default=inline_default
                    )
                else:
                    inline_tune = InlineTuneConfig(
                        final_model_config=final_model_config,
                        timestamp=timestamp,
                        folder=folder,
                        storage_url=all_args.storage_url,
                        n_trials=all_args.n_trials
                    )

                    tune_inline(
                        data=processed_data, cfg=all_config,
                        inline_tune=inline_tune
                    )


    logger.info("Modeling done, see %s for results", folder)


if __name__ == "__main__":

    main()
