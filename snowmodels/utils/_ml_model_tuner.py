import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
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
                ExtraTreesRegressor(random_state=self.random_state),
                X_train, X_val, y_train, y_val
            )
            rf_results = self.train_sklearn_model(
                RandomForestRegressor(random_state=self.random_state),
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