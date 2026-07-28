#!/usr/bin/env python3
"""
Hierarchical Model for the Data Warehouse architecture.

Implements hierarchical models for the Data Warehouse architecture with the ML Consumer pattern,
using DuckDB views for direct queries and connection pooling for performance.
"""

import time

import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrameGroupBy.*')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

_actual_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _actual_project_root not in sys.path:
    sys.path.insert(0, _actual_project_root)

from core.validation import (assert_splits_disjoint, audit_feature_set,
                             canonical_fold)

#: Name this module's paradigm answers to, used wherever an artifact or a
#: diagnostic has to say which of the three produced it.
PARADIGM = 'sql_engine'
from core.base_architecture import BaseArchitectureML
#: Names of the target's autoregressive columns, derived from the single place
#: that declares them. Named rather than matched by substring: the exemption
#: used to key off `_lag_`, which would silently excuse any feature whose name
#: happened to contain it.
_TARGET_LAGS = [f'{BaseArchitectureML.TARGET_STEM}_lag_{order}'
                for order in BaseArchitectureML.TARGET_LAG_ORDERS]

from core.models.hierarchical import (
    simple_hierarchical_model as shared_simple_hierarchical_model,
    write_feature_audit as shared_write_feature_audit,
    write_imputation_report as shared_write_imputation_report,
    write_prediction_artifact as shared_write_prediction_artifact)
from core.validation import impute_from_training_window
from core.scientific_config import SCIENTIFIC_CONFIG, RANDOM_SEED, setup_reproducibility

setup_reproducibility()

try:
    from core.config import get_absolute_output_path
except ImportError:
    def get_absolute_output_path(relative_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        return os.path.join(project_root, 'outputs', relative_path)

connection_manager_path = os.path.join(project_root, 'src', 'collection', 'sql_engine')
if connection_manager_path not in sys.path:
    sys.path.append(connection_manager_path)

try:
    from collection.sql_engine.connection_manager import DuckDBConnectionManager, SQLProcessingError
except ImportError:
    try:
        from connection_manager import DuckDBConnectionManager, SQLProcessingError
    except ImportError as e:
        raise ImportError(f"Could not import DuckDBConnectionManager: {e}")

class HierarchicalModelSQLFirst:
    """
    Hierarchical Model for the Data Warehouse architecture.

    Implements the ML Data Warehouse Consumer pattern with direct queries to the
    views, using 13 selected scientific features.
    """
    
    def __init__(self):
        print("Initialising DuckDB Hierarchical Model")

        self.target_col = 'dropout_rate_sql_engine'
        #: (fold_id, report) for each fold, written out at the end.
        #: The extent of the fold-scoped imputation appeared in no
        #: artifact: the reports were produced and discarded.
        self._imputation_reports = []
        #: P3 audit report for the final feature set, written at the end.
        self._feature_audits = []
        self._cleared_by_selection = []

        print("   Pattern: ML Consumer with views")
        
        dataset_name = os.environ.get('DATASET_NAME', 'worldbank')
        self.folds_path = get_absolute_output_path("ml_pipeline/architectures/sql_engine/prep/temporal_folds_sql_engine.json")
        self.results_path = get_absolute_output_path("ml_pipeline/architectures/sql_engine/models/hierarchical_results")
        self.db_path = get_absolute_output_path(f'collection/sql_engine/{dataset_name}_data.duckdb')
        
        os.makedirs(self.results_path, exist_ok=True)
        
        if not os.path.exists(self.folds_path):
            raise FileNotFoundError(f"Data Warehouse folds not found: {self.folds_path}")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"DuckDB Data Warehouse not found: {self.db_path}")
        
        try:
            self.conn_manager = DuckDBConnectionManager(
                db_path=self.db_path,
                max_retries=3,
                retry_delay=1.0
            )
            print(f"   Connection Manager: {self.db_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialise Connection Manager: {e}")
        with open(self.folds_path, 'r') as f:
            self.folds_config = json.load(f)
            self.folds = self.folds_config['folds']
        
        self._verify_views()
        self._load_data_summary()
    
    def _verify_views(self):
        """Check that the required views exist in the Data Warehouse."""
        print("   Checking views...")

        try:
            count = self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
            if count > 0:
                print(f"   Views checked: {count} records")
            else:
                raise RuntimeError("Setup views are empty")
        except SQLProcessingError as e:
            raise RuntimeError(f"Error checking the setup views: {e}")
    
    def _load_data_summary(self):
        """Load the data summary via direct queries to the views."""
        print("   Loading data summary...")
        
        try:
            total_records = self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
            min_year = self.conn_manager.execute_scalar("SELECT MIN(year) FROM analytics_wide")
            max_year = self.conn_manager.execute_scalar("SELECT MAX(year) FROM analytics_wide")
            total_countries = self.conn_manager.execute_scalar("SELECT COUNT(DISTINCT entity_id) FROM analytics_wide")
            
            print(f"   Data: {total_records} observations")
            print(f"   Period: {min_year}-{max_year}")
            print(f"   Countries: {total_countries}")
            print(f"   Target: {self.target_col}")
            print(f"   Folds: {len(self.folds)}")
            
            target_exists = self.conn_manager.execute_scalar(f"""
                SELECT COUNT(*) > 0 
                FROM information_schema.columns 
                WHERE table_name = 'analytics_wide' 
                AND column_name = '{self.target_col}'
            """)
            
            if not target_exists:
                raise ValueError(f"Target {self.target_col} not found in the analytics_wide table")
            
            target_mean = self.conn_manager.execute_scalar(f"SELECT AVG({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL")
            target_std = self.conn_manager.execute_scalar(f"SELECT STDDEV({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL")
            target_min = self.conn_manager.execute_scalar(f"SELECT MIN({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL")
            target_max = self.conn_manager.execute_scalar(f"SELECT MAX({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL")
            
            print(f"   Target stats: mean={target_mean:.2f}%, std={target_std:.2f}%")
            print(f"   Target range: [{target_min:.2f}%, {target_max:.2f}%]")
                
        except SQLProcessingError as e:
            raise RuntimeError(f"Error loading the summary via views: {e}")
    
    def _load_ml_fold_data(self, fold_id: int, split: str) -> pd.DataFrame:
        """Load the fold data via direct queries to the views."""
        view_name = f"vw_fold_{fold_id}_{split}"

        try:
            query = f"""
                SELECT *
                FROM {view_name}
                WHERE {self.target_col} IS NOT NULL
                ORDER BY entity_id, year
            """
            
            df = self.conn_manager.execute_sql(query)
            
            if df.empty:
                raise SQLProcessingError(f"No data returned from view {view_name}")
            
            return df
            
        except SQLProcessingError as e:
            raise RuntimeError(f"Error loading data for fold {fold_id} split {split}: {e}")
    
    def _prepare_features(self, train_clean):
        """Prepare the feature list based on the mode."""
        selection_path = get_absolute_output_path("ml_pipeline/architectures/sql_engine/prep/feature_selection_sql_engine.json")

        if os.path.exists(selection_path):
            # Normal mode: load the selected features
            with open(selection_path, 'r') as f:
                selection_data = json.load(f)
            available_features = selection_data['selected_features']
            # What the P3 re-audit may skip: selection already applied the
            # proxy ceiling to these, over the full panel, and aborts there.
            self._cleared_by_selection = list(available_features)
            print(f"   {len(available_features)} features from feature selection")
            print(f"   Method: {selection_data.get('selection_method', 'N/A')}")
        else:
            raise FileNotFoundError(f"Feature selection not found: {selection_path}. Run setup.py first.")

        if 'dropout_rate_lag_2' in train_clean.columns and 'dropout_rate_lag_2' not in available_features:
            available_features.append('dropout_rate_lag_2')
        if 'dropout_rate_lag_3' in train_clean.columns and 'dropout_rate_lag_3' not in available_features:
            available_features.append('dropout_rate_lag_3')

        all_columns = list(train_clean.columns)
        available_features = [feat for feat in available_features if feat in all_columns]


        return available_features
    
    def _prepare_data(self, data, available_features, *,
                      return_years: bool = False):
        """
        Materialise a fold already loaded from the database.

        Materialisation only. Every statistic -- the median that fills missing
        values -- lives in core.validation.impute_from_training_window, fitted on
        the fold's training window. Three implementations of one statistic are
        three chances for the paradigms to compute different things, and the
        equivalence claim assumes they differ only in how they move data.

        No reference parameter: materialising does not need the training
        window, only fitting a statistic does.

        The fold's view already applies WHERE target IS NOT NULL and ORDER BY
        entity_id, year -- the same subset and the same order the other
        paradigms produce in Python.
        """
        return canonical_fold(data[available_features], data[self.target_col],
                              data['entity_id'], data['year'],
                              paradigm=PARADIGM, return_years=return_years)
    
    def simple_hierarchical_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 countries_train: pd.Series, countries_test: pd.Series,
                                 residual_shrinkage: float = 0.8) -> Dict:
        """Delegates to the shared implementation (core.models.hierarchical).

        The three paradigms computed this identically -- verified by AST and by
        bitwise equality of the predictions on the same input.
        """
        return shared_simple_hierarchical_model(
            X_train, y_train, X_test, y_test, countries_train, countries_test,
            residual_shrinkage=residual_shrinkage, architecture=PARADIGM)
    
    def random_forest_hierarchical(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 countries_train: pd.Series, countries_test: pd.Series,
                                 max_depth: int = 6, min_samples_leaf: int = 8) -> Dict:
        """
        Hierarchical Random Forest with country effects as features.
        """
        country_means = {}
        global_mean = y_train.mean()

        for country in countries_train.unique():
            country_mask = countries_train == country
            country_y = y_train[country_mask]
            country_means[country] = country_y.mean()

        X_train_augmented = X_train.copy()
        X_test_augmented = X_test.copy()
        
        train_country_effects = [country_means.get(country, global_mean) for country in countries_train]
        test_country_effects = [country_means.get(country, global_mean) for country in countries_test]
        
        X_train_augmented['country_effect'] = train_country_effects
        X_test_augmented['country_effect'] = test_country_effects
        
        # Same hyperparameters as the Data Lake for a fair comparison
        _hm = SCIENTIFIC_CONFIG['hierarchical_model']
        rf_model = RandomForestRegressor(
            n_estimators=_hm['rf_n_estimators'],
            max_depth=max_depth,
            min_samples_split=_hm['rf_min_samples_split'],
            min_samples_leaf=min_samples_leaf,
            max_features=_hm['rf_max_features'],
            random_state=RANDOM_SEED,
            n_jobs=_hm['rf_n_jobs']
        )
        
        rf_model.fit(X_train_augmented, y_train)
        predictions = rf_model.predict(X_test_augmented)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        feature_names = list(X_train_augmented.columns)
        feature_importance = dict(zip(feature_names, rf_model.feature_importances_))

        return {
            'model_name': 'random_forest_hierarchical',
            'architecture': 'sql_engine',
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'predictions': predictions.tolist(),
            'y_true': y_test.tolist(),
            'entities': [str(c) for c in countries_test],
            'feature_importance': {k: float(v) for k, v in feature_importance.items()},
            'country_effects': {str(k): float(v) for k, v in country_means.items()},
            'regularization_applied': (
                f"Regularised: n_est={_hm['rf_n_estimators']}, "
                f"depth={max_depth}, split={_hm['rf_min_samples_split']}, "
                f"leaf={min_samples_leaf}"),
            'country_effect_importance': feature_importance.get('country_effect', 0),
            'rf_params': {'n_estimators': 200, 'max_depth': int(max_depth), 'min_samples_split': 15, 'min_samples_leaf': int(min_samples_leaf)},
            'features_count': X_train_augmented.shape[1]
        }
    
    def run_fold_analysis(self, fold_info: Dict) -> Dict:
        """Run the full analysis for a fold via the ML Data Warehouse Consumer pattern."""
        # Decomposed latency: loading the fold belongs to the engine,
        # the fit is common to the three paradigms, which materialise in
        # pandas before scikit-learn. Measuring the whole stage
        # charged the paradigm a share it does not control.
        _load_t0 = time.perf_counter()
        fold_id = fold_info['fold_id']
        print(f"\nFold {fold_id}: Train({fold_info['train_start']}-{fold_info['train_end']}) -> Val({fold_info['val_start']}-{fold_info['val_end']}) -> Test({fold_info['test_start']}-{fold_info['test_end']})")
        
        try:
            train_data = self._load_ml_fold_data(fold_id, 'train')
            val_data = self._load_ml_fold_data(fold_id, 'val')
            test_data = self._load_ml_fold_data(fold_id, 'test')
            
            print(f"   Data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
            print(f"   Gaps: Train-Val={fold_info['val_start']-fold_info['train_end']-1}yr, Val-Test={fold_info['test_start']-fold_info['val_end']-1}yr")

        except Exception as e:
            print(f"   Error loading data for fold {fold_id}: {e}")
            return {}
        
        if len(train_data) == 0 or len(test_data) == 0:
            print(f"   Fold {fold_id}: Insufficient data")
            return {}
        
        available_features = self._prepare_features(train_data)


        X_train, y_train, countries_train, years_train = self._prepare_data(
            train_data, available_features, return_years=True)
        X_val, y_val, countries_val, years_val = self._prepare_data(
            val_data, available_features, return_years=True)
        X_test, y_test, countries_test, years_test = self._prepare_data(
            test_data, available_features, return_years=True)

        # L1.1 at row granularity. P1 and P2 keep the windows apart; nothing
        # kept a row of the test window out of the training frame, and pasting
        # one there passed every check this framework had.
        assert_splits_disjoint(
            {'train': (countries_train, years_train),
             'val': (countries_val, years_val),
             'test': (countries_test, years_test)},
            paradigm=PARADIGM)
        
        _fold_load_s = time.perf_counter() - _load_t0

        # P3 re-audit, on the matrix this fold's model is about to fit. Placed
        # between the two timers on purpose: it is verification overhead, not
        # work the paradigm does, and charging it to either segment would put an
        # audit cost inside a published latency. Only sql_engine used to pay it,
        # and only inside fold_load_s.
        self._feature_audits.append((fold_id, audit_feature_set(
            X_train, y_train,
            autoregressive=_TARGET_LAGS,
            unaudited_by_selection=[
                column for column in X_train.columns
                if column not in self._cleared_by_selection
                and column not in _TARGET_LAGS],
            config=SCIENTIFIC_CONFIG)))
        _fit_t0 = time.perf_counter()

        # P5: imputation and scaler fitted exclusively on the training data
        # (Kaufman et al. 2012). Imputation comes first because the scaler does
        # not accept missing values, and both statistics come from the same window.
        (X_train, X_val, X_test), imputation_report = \
            impute_from_training_window(X_train, X_val, X_test)
        self._imputation_reports.append((fold_id, imputation_report))
        if imputation_report['columns_without_training_observation']:
            print(f"   [WARN] No observation in the training window, left missing: "
                  f"{imputation_report['columns_without_training_observation']}")

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        print(f"   Running hierarchical models:")

        # Hierarchical models
        # HPO: hyperparameter selection via grid search on the validation
        # set. Final model retrained on the full training set for evaluation
        # on the test set. Prevents leakage (Kapoor & Narayanan, 2023).
        models = {}

        # 1. Simple Hierarchical (tuning of residual_shrinkage)
        best_shrink = 0.8
        best_val_r2 = -1e9
        for rs in SCIENTIFIC_CONFIG['hierarchical_model']['residual_shrinkage_grid']:
            tmp = self.simple_hierarchical_model(
                X_train_scaled, y_train, X_val_scaled, y_val,
                countries_train, countries_val, residual_shrinkage=rs
            )
            if tmp['r2'] > best_val_r2:
                best_val_r2 = tmp['r2']
                best_shrink = rs
                val_simple = tmp
        test_simple = self.simple_hierarchical_model(
            X_train_scaled, y_train, X_test_scaled, y_test,
            countries_train, countries_test, residual_shrinkage=best_shrink
        )
        models['simple_hierarchical'] = {'val': val_simple, 'test': test_simple}

        # 2. Random Forest Hierarchical (light tuning)
        best_params = (6, 8)
        best_val_r2 = -1e9
        _hm = SCIENTIFIC_CONFIG['hierarchical_model']
        for depth in _hm['rf_max_depth_grid']:
            for leaf in _hm['rf_min_samples_leaf_grid']:
                tmp = self.random_forest_hierarchical(
                    X_train_scaled, y_train, X_val_scaled, y_val,
                    countries_train, countries_val,
                    max_depth=depth, min_samples_leaf=leaf
                )
                if tmp['r2'] > best_val_r2:
                    best_val_r2 = tmp['r2']
                    best_params = (depth, leaf)
                    val_rf = tmp
        test_rf = self.random_forest_hierarchical(
            X_train_scaled, y_train, X_test_scaled, y_test,
            countries_train, countries_test,
            max_depth=best_params[0], min_samples_leaf=best_params[1]
        )
        models['random_forest_hierarchical'] = {'val': val_rf, 'test': test_rf}
        
        # Gap analysis
        print(f"   Hierarchical results (Val -> Test):")
        simple_gap = val_simple['r2'] - test_simple['r2']
        rf_gap = val_rf['r2'] - test_rf['r2']
        rf_country_imp = val_rf.get('country_effect_importance', 0)
        
        print(f"      Simple Hierarchical: Val R²={val_simple['r2']:.3f}, Test R²={test_simple['r2']:.3f}, Gap={simple_gap:+.3f}")
        print(f"      Random Forest:       Val R²={val_rf['r2']:.3f}, Test R²={test_rf['r2']:.3f}, Gap={rf_gap:+.3f}")
        print(f"         Country Effect: {rf_country_imp:.3f} (Target: 0.2-0.4)")
        
        # Gap interpretation
        if abs(simple_gap) <= 0.15:
            print(f"      Simple: Gap within target")
        elif abs(simple_gap) <= 0.2:
            print(f"      Simple: Moderate gap")
        else:
            print(f"      Simple: High gap")

        if abs(rf_gap) <= 0.15:
            print(f"      RF: Gap within target")
        elif abs(rf_gap) <= 0.2:
            print(f"      RF: Moderate gap")
        else:
            print(f"      RF: High gap")
        
        _fit_predict_s = time.perf_counter() - _fit_t0

        return {
            'fold_load_s': _fold_load_s,
            'fit_predict_s': _fit_predict_s,
            'fold_id': fold_id,
            'architecture': 'sql_engine',
            'total_features': len(available_features),
            'models': models
        }
    
    def _write_prediction_artifact(self, all_results: Dict) -> None:
        """Delegates to the shared implementation."""
        shared_write_prediction_artifact(all_results, architecture=PARADIGM)
        shared_write_imputation_report(
            self._imputation_reports, architecture=PARADIGM)
        shared_write_feature_audit(
            self._feature_audits, architecture=PARADIGM)

    def run_hierarchical_analysis(self):
        """Run the full hierarchical analysis via the ML Data Warehouse Consumer."""
        print("DuckDB hierarchical analysis")
        print("   RidgeCV (Hoerl & Kennard 1970), James-Stein shrinkage (Efron & Morris 1975)")
        
        try:
            _meta = SCIENTIFIC_CONFIG['hierarchical_model']
            all_results = {
                'architecture': 'sql_engine',
                'version': 'hierarchical_analysis',
                'target': self.target_col,
                'mode': 'normal',
                'corrections_applied': {
                    'simple_hierarchical': (
                    f"RidgeCV with alphas logspace("
                    f"{_meta['ridge_alpha_log10_start']}, "
                    f"{_meta['ridge_alpha_log10_stop']}, "
                    f"{_meta['ridge_alpha_count']}) + James-Stein shrinkage"),
                    'random_forest': (
                    f"Regularised: n_est={_meta['rf_n_estimators']}, "
                    f"depth in {tuple(_meta['rf_max_depth_grid'])}, "
                    f"split={_meta['rf_min_samples_split']}, "
                    f"leaf in {tuple(_meta['rf_min_samples_leaf_grid'])}"),
                    'regularization_approach': 'RidgeCV (Hoerl & Kennard 1970) + Shrinkage (Efron & Morris 1975)'
                },
                'folds': []
            }
            
            for fold_info in self.folds_config['folds']:
                _fold_t0 = time.perf_counter()
                fold_results = self.run_fold_analysis(fold_info)
                if fold_results:
                    fold_results['fold_duration_s'] = time.perf_counter() - _fold_t0
                    all_results['folds'].append(fold_results)
                    
                    # Log the results
                    fold_id = fold_info['fold_id']
                    if 'models' in fold_results:
                        for model_name, model_results in fold_results['models'].items():
                            val_r2 = model_results['val']['r2']
                            test_r2 = model_results['test']['r2']
                            gap = val_r2 - test_r2
                            print(f"   {model_name}: Val R²={val_r2:.3f}, Test R²={test_r2:.3f}, Gap={gap:+.3f}")
            
            # Aggregate performance
            if all_results['folds']:
                print(f"\nDuckDB aggregate performance:")
                
                for model_name in ['simple_hierarchical', 'random_forest_hierarchical']:
                    val_r2s = []
                    test_r2s = []
                    
                    for fold in all_results['folds']:
                        if 'models' in fold and model_name in fold['models']:
                            val_r2s.append(fold['models'][model_name]['val']['r2'])
                            test_r2s.append(fold['models'][model_name]['test']['r2'])
                    
                    if val_r2s and test_r2s:
                        val_mean = np.mean(val_r2s)
                        test_mean = np.mean(test_r2s)
                        test_std = np.std(test_r2s)
                        gap_mean = val_mean - test_mean
                        
                        print(f"\n   {model_name}:")
                        print(f"      Val:  R² = {val_mean:.3f}")
                        print(f"      Test: R² = {test_mean:.3f} ± {test_std:.3f}")
                        print(f"      Gap:  {gap_mean:+.3f}")
                        
                        # Gap analysis
                        abs_gap = abs(gap_mean)
                        if abs_gap <= 0.15:
                            print(f"      Effective regularisation - gap within target")
                        elif abs_gap <= 0.2:
                            print(f"      Acceptable gap")
                        else:
                            print(f"      Additional regularisation required")

                print(f"\nDuckDB hierarchical summary")
            
            self._write_prediction_artifact(all_results)

            results_file = f"{self.results_path}/hierarchical_analysis_sql_engine_results.json"
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"\nResults saved: {results_file}")
            
            return all_results
            
        except Exception as e:
            # Re-raises: returning a dictionary with 'folds': [] made the
            # orchestrator and the benchmark treat the failure as a fast and
            # successful run, and the equivalence gate does not see a
            # paradigm that wrote no vector at all.
            print(f"Error in the hierarchical analysis: {e}")
            raise
        finally:
            # Cleanup connection
            if hasattr(self, 'conn_manager') and self.conn_manager:
                try:
                    self.conn_manager.close_connection()
                    print("   Connection Manager closed")
                except Exception as e:
                    print(f"   Error closing Connection Manager: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hierarchical Model Data Warehouse')
    args = parser.parse_args()

    try:
        model = HierarchicalModelSQLFirst()
        results = model.run_hierarchical_analysis()
        print(f"\nDuckDB hierarchical analysis completed!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
