import optuna
import logging
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
from sklearn import set_config
from typing import List, Tuple
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

set_config(transform_output="pandas")
warnings.filterwarnings('ignore')

class DefaultTuner:
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        cat_col: str = 'Snow_Class',
        num_cols: List[str] = ['Elevation', 'Snow_Depth', 'DOWY'],
        random_state: int = 42
    ):
        # Select only the features we want to use
        self.X_train = X_train[num_cols + [cat_col]]
        self.X_val = X_val[num_cols + [cat_col]]
        self.y_train = y_train
        self.y_val = y_val
        self.cat_col = cat_col
        self.num_cols = num_cols
        self.random_state = random_state
        
    def prepare_data(self, encoder_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data with specified encoder"""
        if encoder_name == 'onehot':
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        elif encoder_name == 'catboost':
            encoder = ce.CatBoostEncoder(cols=[self.cat_col])
        elif encoder_name == 'target':
            encoder = ce.TargetEncoder(cols=[self.cat_col], min_samples_leaf=20, smoothing=10)
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
        # Encode categorical features
        X_cat_train = encoder.fit_transform(self.X_train[[self.cat_col]], self.y_train)
        X_cat_val = encoder.transform(self.X_val[[self.cat_col]])
        
        # Combine with numerical features
        X_train = pd.concat([self.X_train[self.num_cols], X_cat_train], axis=1)
        X_val = pd.concat([self.X_val[self.num_cols], X_cat_val], axis=1)
        
        return X_train.values, X_val.values, self.y_train.values, self.y_val.values

    def train_lightgbm(self, X_train, X_val, y_train, y_val):
        """Train LightGBM with early stopping"""
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Default parameters
        params = {
            'objective': 'regression',
            'metric': ['rmse'],
            'random_state': self.random_state
        }

        #evals_result = {} # in case I need to store the eval results for plotting
        
        early_stopping = [
            lgb.early_stopping(stopping_rounds=50), 
            # lgb.log_evaluation(100), # to log eval
            # lgb.record_evaluation(evals_result) # in case I need to store the eval results for plotting
        ]

        # Train with early stopping
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=early_stopping
        )
        
        # Get predictions
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        score = r2_score(y_true=y_val, y_pred=y_pred)
        rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
        
        return {
            'score': score,
            'rmse': rmse,
            'best_iteration': model.best_iteration,
            'params': params
        }

    def train_xgboost(self, X_train, X_val, y_train, y_val):
        """Train XGBoost with early stopping"""
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Default parameters
        params = {
            'objective': 'reg:squarederror',
            'random_state': self.random_state
        }

        early_stopping= xgb.callback.EarlyStopping(
            rounds=50,
            metric_name='rmse',
            data_name='valid',
            maximize=False,
            save_best=True
        )
        
        # Train with early stopping
        # evals_result = {}  # in case I need to store the eval results for plotting
        model = xgb.train(
            params,
            dtrain,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            num_boost_round=1000,
            callbacks=[early_stopping],
            verbose_eval=False,
            # evals_result=evals_result
        )
        
        # Get predictions
        y_pred = model.predict(dval)
        score = r2_score(y_true=y_val, y_pred=y_pred)
        rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
        
        return {
            'score': score,
            'rmse': rmse,
            'best_iteration': model.best_iteration,
            'params': params
        }

    def train_sklearn_model(self, model, X_train, X_val, y_train, y_val):
        """Train sklearn model (ExtraTrees, RandomForest)"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = r2_score(y_true=y_val, y_pred=y_pred)
        rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
        
        return {
            'score': score,
            'rmse': rmse,
            'params': model.get_params()
        }

    def run_default_models(self):
        """Run all models with default hyperparameters"""
        encoders = ['onehot', 'catboost', 'target']
        
        baseline_results = {}
        print("\nRunning baseline models with default parameters...")
        
        for encoder_name in encoders:
            X_train, X_val, y_train, y_val = self.prepare_data(encoder_name)
            
            # Train LightGBM and XGBoost using native API
            lgb_results = self.train_lightgbm(X_train, X_val, y_train, y_val)
            xgb_results = self.train_xgboost(X_train, X_val, y_train, y_val)
            
            # Train sklearn models
            et_results = self.train_sklearn_model(
                ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1),
                X_train, X_val, y_train, y_val
            )
            rf_results = self.train_sklearn_model(
                RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
                X_train, X_val, y_train, y_val
            )
            
            # Store results
            baseline_results[f"lightgbm_{encoder_name}"] = lgb_results
            baseline_results[f"xgboost_{encoder_name}"] = xgb_results
            baseline_results[f"extra_trees_{encoder_name}"] = et_results
            baseline_results[f"random_forest_{encoder_name}"] = rf_results
            
            # Print results for this encoder
            print(f"\nResults with {encoder_name} encoder:")
            print(f"LightGBM - R²: {lgb_results['score']:.4f}, RMSE: {lgb_results['rmse']:.4f}, Best iteration: {lgb_results['best_iteration']}")
            print(f"XGBoost - R²: {xgb_results['score']:.4f}, RMSE: {xgb_results['rmse']:.4f}, Best iteration: {xgb_results['best_iteration']}")
            print(f"ExtraTrees - R²: {et_results['score']:.4f}, RMSE: {et_results['rmse']:.4f}")
            print(f"RandomForest - R²: {rf_results['score']:.4f}, RMSE: {rf_results['rmse']:.4f}")
        
        # Find best model
        best_model = max(baseline_results.items(), key=lambda x: x[1]['score'])
        print("\nBest model configuration:")
        print(f"Model: {best_model[0]}")
        print(f"R² score: {best_model[1]['score']:.4f}")
        print(f"RMSE: {best_model[1]['rmse']:.4f}")
        if 'best_iteration' in best_model[1]:
            print(f"Best iteration: {best_model[1]['best_iteration']}")
        
        return baseline_results

class ComprehensiveOptimizer(DefaultTuner):
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        model_name: str,
        cat_col: str = 'Snow_Class',
        num_cols: List[str] = ['Elevation', 'Snow_Depth', 'DOWY'],
        random_state: int = 42,
        log_file: str = None
    ):
        # Initialize parent class
        super().__init__(X_train, X_val, y_train, y_val, cat_col, num_cols, random_state)
        
        # Additional initialization
        self.model_name = model_name.lower()
        if self.model_name not in ['random_forest', 'xgboost', 'lightgbm', 'extra_trees']:
            raise ValueError("model_name must be one of: 'random_forest', 'xgboost', 'lightgbm', 'extra_trees'")
        
        # Setup logging
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"optimization_{self.model_name}_{timestamp}.log"
        
        # Configure logging
        self.logger = logging.getLogger(f"optimizer_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        
        # File handler for detailed logging
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        self.logger.addHandler(fh)
        
        # Console handler for summary info
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        self.logger.addHandler(ch)

    def optimize_random_forest(self, trial: optuna.Trial, X_train, X_val, y_train, y_val) -> float:
        """Optimize Random Forest hyperparameters"""

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
        
        # Add R² to trial info
        score = r2_score(y_true=y_val, y_pred=y_pred)
        trial.set_user_attr('r2', score)
        
        return rmse  # Return RMSE for minimization

    def optimize_extra_trees(self, trial: optuna.Trial, X_train, X_val, y_train, y_val) -> float:
        """Optimize Extra Trees hyperparameters"""

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        model = ExtraTreesRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
        
        # Add R² to trial info
        score = r2_score(y_true=y_val, y_pred=y_pred)
        trial.set_user_attr('r2', score)
        
        return rmse  # Return RMSE for minimization

    def optimize_xgboost(self, trial: optuna.Trial, X_train, X_val, y_train, y_val) -> float:
        """Optimize XGBoost hyperparameters"""
        params = {
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 2, 25),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e2),
            'max_bin': trial.suggest_int('max_bin', 255, 6000),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'device': 'cuda',
            'tree_method': 'hist',
            'subsample': 1,
            'sampling_method': 'gradient_based',
            'random_state': self.random_state
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        early_stopping = xgb.callback.EarlyStopping(
            rounds=50,
            metric_name='rmse',
            data_name='valid',
            maximize=False,
            save_best=True
        )
        
        # Train with early stopping
        model = xgb.train(
            params,
            dtrain,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            num_boost_round=1500,
            callbacks=[early_stopping],
            verbose_eval=False
        )
        
        # Get predictions using best model
        # y_pred = model[: model.best_iteration].predict(dval)
        y_pred = model.predict(dval)
        rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
        
        # Add metrics to trial info
        score = r2_score(y_true=y_val, y_pred=y_pred)
        trial.set_user_attr('r2', score)
        trial.set_user_attr('best_iteration', model.best_iteration)
        
        return rmse  # Return RMSE for minimization

    def optimize_lightgbm(self, trial: optuna.Trial, X_train, X_val, y_train, y_val) -> float:
        """Optimize LightGBM hyperparameters"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': self.random_state,
            'verbosity': -1,
            'max_depth': trial.suggest_int('max_depth', 2, 25),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e2),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
            'max_bin': trial.suggest_int('max_bin', 255, 6000),
            'bagging_fraction': 1,
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [lgb.early_stopping(stopping_rounds=50)]
        
        # Train with early stopping
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            callbacks=callbacks,
            num_boost_round=1500
        )
        
        # Get predictions using best iteration
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
        
        # Add metrics to trial info
        score = r2_score(y_true=y_val, y_pred=y_pred)
        trial.set_user_attr('r2', score)
        trial.set_user_attr('best_iteration', model.best_iteration)
        
        return rmse  # Return RMSE for minimization

    def optimize(self, n_trials: int = 100, storage: str = "sqlite:///optuna.db"):
        """Run optimization for all encoders"""
        encoders = ['onehot', 'catboost', 'target']
        all_results = {}
        
        self.logger.info(f"\nStarting optimization for {self.model_name}")
        self.logger.info(f"Number of trials: {n_trials}")
        self.logger.info(f"Storage: {storage}")
        
        # Select optimization function based on model name
        if self.model_name == 'random_forest':
            optimize_func = self.optimize_random_forest
        elif self.model_name == 'extra_trees':
            optimize_func = self.optimize_extra_trees
        elif self.model_name == 'xgboost':
            optimize_func = self.optimize_xgboost
        else:  # lightgbm
            optimize_func = self.optimize_lightgbm
        
        # Run optimization for each encoder
        for encoder_name in encoders:
            self.logger.info(f"\nOptimizing {self.model_name} with {encoder_name} encoder...")
            
            # Prepare data for this encoder
            X_train, X_val, y_train, y_val = self.prepare_data(encoder_name)
            
            # Create study
            study = optuna.create_study(
                study_name=f"{self.model_name}_{encoder_name}",
                storage=storage,
                direction="minimize",
                load_if_exists=True
            )
            
            # Create objective function closure
            objective = lambda trial: optimize_func(trial, X_train, X_val, y_train, y_val)
            
            # Run optimization
            study.optimize(objective, n_trials=n_trials) # use n_jobs=-1 for LigGBM if you have multiple cores.
            
            # Store results
            all_results[f"{self.model_name}_{encoder_name}"] = {
                'best_params': study.best_trial.params,
                'best_rmse': study.best_trial.value,
                'best_r2': study.best_trial.user_attrs['r2'],
                'study': study
            }
            
            if self.model_name in ['xgboost', 'lightgbm']:
                all_results[f"{self.model_name}_{encoder_name}"]["best_iteration"] = \
                    study.best_trial.user_attrs['best_iteration']
            
            # Log detailed results to file
            self.logger.debug(f"\nDetailed results for {self.model_name} with {encoder_name} encoder:")
            self.logger.debug(f"Best RMSE: {study.best_trial.value:.4f}")
            self.logger.debug(f"Best R² score: {study.best_trial.user_attrs['r2']:.4f}")
            if self.model_name in ['xgboost', 'lightgbm']:
                self.logger.debug(f"Best iteration: {study.best_trial.user_attrs['best_iteration']}")
            self.logger.debug("\nBest parameters:")
            for param, value in study.best_trial.params.items():
                self.logger.debug(f"  {param}: {value}")
        
        # Find overall best configuration (minimum RMSE)
        best_config = min(all_results.items(), key=lambda x: x[1]['best_rmse'])
        
        # Print summary to console
        self.logger.info("\n" + "="*50)
        self.logger.info("Optimization Summary")
        self.logger.info("="*50)
        self.logger.info(f"Best configuration: {best_config[0]}")
        self.logger.info(f"Best RMSE: {best_config[1]['best_rmse']:.4f}")
        self.logger.info(f"Best R² score: {best_config[1]['best_r2']:.4f}")
        
        return all_results


class ClimateOptimizer(DefaultTuner):
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        model_name: str,
        cat_col: str = 'Snow_Class',
        num_cols: List[str] = ['Elevation', 'Snow_Depth', 'DOWY',
                              'PRECIPITATION_lag_14d', 'TAVG_lag_14d'],
        random_state: int = 42,
        log_file: str = None
    ):
        # Initialize parent class
        super().__init__(X_train, X_val, y_train, y_val, cat_col, num_cols, random_state)
        
        # Additional initialization
        self.model_name = model_name.lower()
        if self.model_name not in ['random_forest', 'xgboost', 'lightgbm', 'extra_trees']:
            raise ValueError("model_name must be one of: 'random_forest', 'xgboost', 'lightgbm', 'extra_trees'")
        
        # Setup logging
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"climate_optimization_{self.model_name}_{timestamp}.log"
        
        # Configure logging
        self.logger = logging.getLogger(f"climate_optimizer_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        
        # File handler for detailed logging
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        self.logger.addHandler(fh)
        
        # Console handler for summary info
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        self.logger.addHandler(ch)

    def optimize_random_forest(self, trial: optuna.Trial, X_train, X_val, y_train, y_val) -> float:
        """Optimize Random Forest hyperparameters"""

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
        
        # Add R² to trial info
        score = r2_score(y_true=y_val, y_pred=y_pred)
        trial.set_user_attr('r2', score)
        
        return rmse  # Return RMSE for minimization

    def optimize_extra_trees(self, trial: optuna.Trial, X_train, X_val, y_train, y_val) -> float:
        """Optimize Extra Trees hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        model = ExtraTreesRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
        
        # Add R² to trial info
        score = r2_score(y_true=y_val, y_pred=y_pred)
        trial.set_user_attr('r2', score)
        
        return rmse  # Return RMSE for minimization

    def optimize_xgboost(self, trial: optuna.Trial, X_train, X_val, y_train, y_val) -> float:
        """Optimize XGBoost hyperparameters with fixed n_estimators and early stopping"""

        params = {
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 2, 25),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e2),
            'max_bin': trial.suggest_int('max_bin', 255, 6000),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'device': 'cuda',
            'tree_method': 'hist',
            'subsample': 1,
            'sampling_method': 'gradient_based',
            'random_state': self.random_state
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        early_stopping = xgb.callback.EarlyStopping(
            rounds=50,
            metric_name='rmse',
            data_name='valid',
            maximize=False,
            save_best=True
        )
        
        # Train with early stopping and fixed n_estimators
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1500,  # Fixed large number of trees
            evals=[(dtrain, 'train'), (dval, 'valid')],
            callbacks=[early_stopping],
            verbose_eval=False
        )
        
        # Get predictions using best model
        y_pred = model.predict(dval)
        rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
        
        # Add metrics to trial info
        score = r2_score(y_true=y_val, y_pred=y_pred)
        trial.set_user_attr('r2', score)
        trial.set_user_attr('best_iteration', model.best_iteration)
        
        return rmse  # Return RMSE for minimization

    def optimize_lightgbm(self, trial: optuna.Trial, X_train, X_val, y_train, y_val) -> float:
        """Optimize LightGBM hyperparameters with fixed n_estimators and early stopping"""

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': self.random_state,
            'verbosity': -1,
            'max_depth': trial.suggest_int('max_depth', 2, 25),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e2),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
            'max_bin': trial.suggest_int('max_bin', 255, 6000),
            'bagging_fraction': 1,
        }
        
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [lgb.early_stopping(stopping_rounds=50)]
        
        # Train with early stopping and fixed n_estimators
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1500,  # Fixed large number of trees
            valid_sets=[train_data, val_data],
            callbacks=callbacks
        )
        
        # Get predictions using best iteration
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
        
        # Add metrics to trial info
        score = r2_score(y_true=y_val, y_pred=y_pred)
        trial.set_user_attr('r2', score)
        trial.set_user_attr('best_iteration', model.best_iteration)
        
        return rmse  # Return RMSE for minimization

    def optimize(self, n_trials: int = 100, storage: str = "sqlite:///optuna.db"):
        """Run optimization using only target encoding"""
        self.logger.info(f"\nStarting climate model optimization for {self.model_name}")
        self.logger.info(f"Number of trials: {n_trials}")
        self.logger.info(f"Storage: {storage}")
        
        # Select optimization function based on model name
        if self.model_name == 'random_forest':
            optimize_func = self.optimize_random_forest
        elif self.model_name == 'extra_trees':
            optimize_func = self.optimize_extra_trees
        elif self.model_name == 'xgboost':
            optimize_func = self.optimize_xgboost
        else:  # lightgbm
            optimize_func = self.optimize_lightgbm
        
        # Prepare data with target encoding only
        X_train, X_val, y_train, y_val = self.prepare_data('target')
        
        # Create study with pruner to reduce output
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
            interval_steps=10
        )
        
        study = optuna.create_study(
            study_name=f"{self.model_name}_climate",
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            pruner=pruner
        )
        
        # Create objective function closure
        objective = lambda trial: optimize_func(trial, X_train, X_val, y_train, y_val)
        
        # Run optimization with minimal output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False) # n_jobs=-1
        
        # Store results
        results = {
            'best_params': study.best_trial.params,
            'best_rmse': study.best_trial.value,
            'best_r2': study.best_trial.user_attrs['r2'],
            'study': study
        }
        
        if self.model_name in ['xgboost', 'lightgbm']:
            results['best_iteration'] = study.best_trial.user_attrs['best_iteration']
        
        # Print summary to console
        self.logger.info("\n" + "="*50)
        self.logger.info("Climate Model Optimization Summary")
        self.logger.info("="*50)
        self.logger.info(f"Best RMSE: {results['best_rmse']:.4f}")
        self.logger.info(f"Best R² score: {results['best_r2']:.4f}")
        if self.model_name in ['xgboost', 'lightgbm']:
            self.logger.info(f"Best iteration: {results['best_iteration']}")
        self.logger.info("\nBest parameters:")
        for param, value in results['best_params'].items():
            self.logger.info(f"  {param}: {value}")
        
        return results