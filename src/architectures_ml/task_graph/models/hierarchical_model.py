#!/usr/bin/env python3
"""
Hierarchical Model for the Data Lake Architecture.

Implements hierarchical models for the Data Lake architecture with distributed
processing and schema-on-read for school dropout prediction.

Optimized with batch compute to reduce Dask calls.
"""

import time

import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask
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

core_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'core')
core_path = os.path.abspath(core_path)
if core_path not in sys.path:
    sys.path.append(core_path)

from config import get_absolute_output_path

project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.append(project_root)
from core.injection import active as injection_active
from core.injection import duplicate_evaluation_rows
from core.validation import (assert_splits_disjoint, audit_feature_set,
                             canonical_fold)

#: Name this module's paradigm answers to, used wherever an artifact or a
#: diagnostic has to say which of the three produced it.
PARADIGM = 'task_graph'
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

class HierarchicalModelTaskGraph:
    """
    Hierarchical Model for the Data Lake Architecture.

    Implements hierarchical models with distributed processing and schema-on-read.
    """
    
    def __init__(self):
        print("Initializing Dask Hierarchical Model")

        self.target_col = 'dropout_rate_task_graph'
        #: (fold_id, report) for each fold, written out at the end.
        #: The extent of the fold-scoped imputation appeared in no
        #: artifact: the reports were produced and discarded.
        self._imputation_reports = []
        #: P3 audit report of the final feature set, written out at the end.
        self._feature_audits = []
        self._cleared_by_selection = []
        self._injection_records = []
        self._disjointness_records = []

        self._setup_normal_mode()

        self.results_path = get_absolute_output_path("ml_pipeline/architectures/task_graph/models/hierarchical_results")
        os.makedirs(self.results_path, exist_ok=True)

        self._load_normal_summary()
    
    def _setup_normal_mode(self):
        """Setup for normal mode."""
        self.data_path = get_absolute_output_path("ml_pipeline/architectures/task_graph/prep/master_data_task_graph.parquet")
        self.folds_path = get_absolute_output_path("ml_pipeline/architectures/task_graph/prep/temporal_folds_task_graph.json")
        
        if not os.path.exists(self.data_path) or not os.path.exists(self.folds_path):
            raise FileNotFoundError("Data Lake data not found")
        
        print("   Loading data with Dask...")
        self.ddf = dd.read_parquet(self.data_path, engine='pyarrow')
        self._needs_persist = True
        with open(self.folds_path, 'r') as f:
            self.folds_config = json.load(f)
            self.folds = self.folds_config['folds']
    
    def _load_normal_summary(self):
        """Load the summary of the normal data with optimized batch compute."""
        # Batch compute for basic statistics
        stats_batch = {
            'year_min': self.ddf['year'].min(),
            'year_max': self.ddf['year'].max(),
            'n_countries': self.ddf['entity_id'].nunique(),
            'unique_countries': self.ddf['entity_id'].unique()
        }
        computed_stats = dask.compute(stats_batch)[0]
        
        n_records = self.ddf.shape[0].compute()
        print(f"   Data: {n_records} records, {self.ddf.npartitions} partitions")
        print(f"   Period: {computed_stats['year_min']}-{computed_stats['year_max']}")
        print(f"   Countries: {computed_stats['n_countries']}")
        print(f"   Target: {self.target_col}")
        print(f"   Folds: {len(self.folds)}")
        
        if self.target_col not in self.ddf.columns:
            raise ValueError(f"Target {self.target_col} not found in the data")
        

        
        selection_path = get_absolute_output_path("ml_pipeline/architectures/task_graph/prep/feature_selection_task_graph.json")
        if os.path.exists(selection_path):
            try:
                with open(selection_path, 'r') as f:
                    selection_data = json.load(f)
                # Indexed, not .get with a default: the empty list let this
                # paradigm train on the target's lags alone, and the audit
                # downstream passed for want of any exogenous feature to fail
                # on.
                selected = selection_data['selected_features']
                # What the P3 re-audit may skip: selection already applied
                # the proxy ceiling to these, over the full panel.
                self._cleared_by_selection = list(selected)
                # Include the target lag if it exists
                if 'dropout_rate_lag_2' in self.ddf.columns and 'dropout_rate_lag_2' not in selected:
                    selected.append('dropout_rate_lag_2')
                if 'dropout_rate_lag_3' in self.ddf.columns and 'dropout_rate_lag_3' not in selected:
                    selected.append('dropout_rate_lag_3')
                # Filter by existence
                existing = [c for c in selected if c in self.ddf.columns]
                self.available_features = existing
                print(f"Available features (scientific selection): {len(self.available_features)}")
            except Exception as e:
                print(f"Failed to load feature selection: {e}")
                raise
        else:
            raise FileNotFoundError(f"Feature selection not found: {selection_path}. Run setup.py first.")
    
    def _prepare_data(self, data_ddf, *, return_years: bool = False):
        """
        Materialize a fold with a single batched compute.

        Materialization only. Every statistic -- the median that fills missing
        values -- lives in core.validation.impute_from_training_window, fitted on
        the fold's training window. Three implementations of one statistic are
        three chances for the paradigms to compute different things, and the
        equivalence claim assumes they differ only in how they move data.

        No reference parameter: materializing does not need the training window,
        only fitting a statistic does.
        """
        X_ddf = data_ddf[self.available_features]

        final_data = {
            'X': X_ddf,
            'y': data_ddf[self.target_col],
            'countries': data_ddf['entity_id'],
            'year': data_ddf['year'],
        }
        computed_final = dask.compute(final_data)[0]

        X_out, y_out, c_out, yr_out = (
            computed_final['X'], computed_final['y'],
            computed_final['countries'], computed_final['year'])
        valid_mask = y_out.notna()
        X_out, y_out, c_out, yr_out = (X_out[valid_mask], y_out[valid_mask],
                                       c_out[valid_mask], yr_out[valid_mask])

        # Positional, not label-based. Each Dask partition carries its own
        # index, so after compute the labels repeat across partitions, and
        # .loc[sort_idx] selects every row matching each label: six rows in,
        # twelve out. The fit would succeed on a silently doubled fold.
        order = np.lexsort((yr_out.to_numpy(), c_out.to_numpy()))
        X_out, y_out, c_out, yr_out = (X_out.iloc[order], y_out.iloc[order],
                                       c_out.iloc[order], yr_out.iloc[order])
        return canonical_fold(X_out, y_out, c_out, yr_out,
                              paradigm=PARADIGM, return_years=return_years)
    
    def simple_hierarchical_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 countries_train: pd.Series, countries_test: pd.Series,
                                 residual_shrinkage: float = 0.8) -> Dict:
        """Delegates to the shared implementation (core.models.hierarchical).

        The three paradigms computed this identically -- verified by AST and by
        bitwise equality of the predictions over the same input.
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
        
        print(f"      Random Forest: {X_train_augmented.shape[1]} total features ({X_train_augmented.shape[1]-1} base + 1 country_effect) × {X_train_augmented.shape[0]} samples")
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        feature_names = list(X_train_augmented.columns)
        feature_importance = dict(zip(feature_names, rf_model.feature_importances_))

        return {
            'model_name': 'random_forest_hierarchical',
            'architecture': 'task_graph',
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'predictions': predictions.tolist(),
            'y_true': y_test.tolist(),
            'entities': [str(c) for c in countries_test],
            'feature_importance': {k: float(v) for k, v in feature_importance.items()},
            'country_effects': {str(k): float(v) for k, v in country_means.items()},
            'regularization_applied': (
                f"Regularized: n_est={_hm['rf_n_estimators']}, "
                f"depth={max_depth}, split={_hm['rf_min_samples_split']}, "
                f"leaf={min_samples_leaf}"),
            'rf_params': {'n_estimators': 200, 'max_depth': int(max_depth), 'min_samples_split': 15, 'min_samples_leaf': int(min_samples_leaf)},
            'country_effect_importance': feature_importance.get('country_effect', 0),
            'features_count': X_train_augmented.shape[1]
        }
    
    def run_fold_analysis(self, fold_info: Dict) -> Dict:
        """Run the complete hierarchical analysis for a specific fold."""
        # Decomposed latency: loading the fold belongs to the engine,
        # the fit is common to all three paradigms, which materialize into
        # pandas before scikit-learn. Measuring the whole stage
        # charged the paradigm with a share it does not control.
        _load_t0 = time.perf_counter()
        fold_id = fold_info['fold_id']
        print(f"\nAnalyzing Fold {fold_id} Dask (NORMAL)...")

        train_ddf = self.ddf[
            (self.ddf['year'] >= fold_info['train_start']) &
            (self.ddf['year'] <= fold_info['train_end'])
        ]
        train_ddf = train_ddf[
            ~((train_ddf['year'] >= fold_info['train_gap_start']) &
              (train_ddf['year'] <= fold_info['train_gap_end']))
        ]
        val_ddf = self.ddf[
            (self.ddf['year'] >= fold_info['val_start']) &
            (self.ddf['year'] <= fold_info['val_end'])
        ]
        test_ddf = self.ddf[
            (self.ddf['year'] >= fold_info['test_start']) &
            (self.ddf['year'] <= fold_info['test_end'])
        ]
        test_ddf = test_ddf[
            ~((test_ddf['year'] >= fold_info['val_gap_start']) &
              (test_ddf['year'] <= fold_info['val_gap_end']))
        ]
        n_train, n_val, n_test = dask.compute(
            train_ddf.shape[0], val_ddf.shape[0], test_ddf.shape[0]
        )
        print(f"Normal Data: Train={n_train}, Val={n_val}, Test={n_test}")

        X_train, y_train, countries_train, years_train = self._prepare_data(
            train_ddf, return_years=True)
        X_val, y_val, countries_val, years_val = self._prepare_data(
            val_ddf, return_years=True)
        X_test, y_test, countries_test, years_test = self._prepare_data(
            test_ddf, return_years=True)

        # L1.1 at row granularity. P1 and P2 keep the windows apart; nothing
        # kept a row of the test window out of the training frame, and pasting
        # one there passed every check this framework had.
        # The arm's declared violation, if this run is an arm at all. Read once
        # and passed explicitly to everything that has to know: a gate that
        # consulted the environment on its own would soften without the call
        # site saying so.
        _injection = injection_active()
        if _injection is not None and _injection.klass == 'C3':
            (X_train, y_train, countries_train, years_train), _c3 = \
                duplicate_evaluation_rows(
                    X_train, y_train, countries_train, years_train,
                    X_test, y_test, countries_test, years_test,
                    spec=_injection, fold_id=fold_id)
            self._injection_records.append((fold_id, _c3))

        self._disjointness_records.append((fold_id, assert_splits_disjoint(
            {'train': (countries_train, years_train),
             'val': (countries_val, years_val),
             'test': (countries_test, years_test)},
            paradigm=PARADIGM, injection=_injection)))
        
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

        # P5: imputation and scaler fitted exclusively on the training set
        # (Kaufman et al. 2012). Imputation comes first because the scaler does
        # not accept missing values, and both statistics come from the same window.
        (X_train, X_val, X_test), imputation_report = \
            impute_from_training_window(X_train, X_val, X_test)
        self._imputation_reports.append((fold_id, imputation_report))
        if imputation_report['columns_without_training_observation']:
            print(f"   [WARN] No observation in training, left missing: "
                  f"{imputation_report['columns_without_training_observation']}")

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        print(f"   {len(self.available_features)} features, Train={X_train_scaled.shape}, Val={X_val_scaled.shape}, Test={X_test_scaled.shape}")
        print(f"   Countries: Train={countries_train.nunique()}, Val={countries_val.nunique()}, Test={countries_test.nunique()}")
        
        # Hierarchical models
        # HPO: hyperparameter selection via grid search on the validation set.
        # Final model retrained on the full training set for evaluation on the
        # test set. Prevents leakage (Kapoor & Narayanan, 2023).
        models = {}

        # 1. Simple Hierarchical (residual_shrinkage tuned on validation)
        best_shrink = 0.8
        best_val_r2 = -1e9
        for rs in SCIENTIFIC_CONFIG['hierarchical_model']['residual_shrinkage_grid']:
            tmp = self.simple_hierarchical_model(X_train_scaled, y_train, X_val_scaled, y_val, countries_train, countries_val, residual_shrinkage=rs)
            if tmp['r2'] > best_val_r2:
                best_val_r2 = tmp['r2']
                best_shrink = rs
                val_simple = tmp
        test_simple = self.simple_hierarchical_model(X_train_scaled, y_train, X_test_scaled, y_test, countries_train, countries_test, residual_shrinkage=best_shrink)
        models['simple_hierarchical'] = {'val': val_simple, 'test': test_simple}
        
        # 2. Random Forest Hierarchical (light tuning on validation)
        best_params = (6, 8)
        best_val_r2 = -1e9
        _hm = SCIENTIFIC_CONFIG['hierarchical_model']
        for depth in _hm['rf_max_depth_grid']:
            for leaf in _hm['rf_min_samples_leaf_grid']:
                tmp = self.random_forest_hierarchical(X_train_scaled, y_train, X_val_scaled, y_val, countries_train, countries_val, max_depth=depth, min_samples_leaf=leaf)
                if tmp['r2'] > best_val_r2:
                    best_val_r2 = tmp['r2']
                    best_params = (depth, leaf)
                    val_rf = tmp
        test_rf = self.random_forest_hierarchical(X_train_scaled, y_train, X_test_scaled, y_test, countries_train, countries_test, max_depth=best_params[0], min_samples_leaf=best_params[1])
        models['random_forest_hierarchical'] = {'val': val_rf, 'test': test_rf}
        
        # Gap analysis
        print(f"\n   Hierarchical results (Val -> Test):")
        simple_gap = val_simple['r2'] - test_simple['r2']
        rf_gap = val_rf['r2'] - test_rf['r2']
        rf_country_imp = val_rf.get('country_effect_importance', 0)
        
        print(f"      Simple Hierarchical: Val R²={val_simple['r2']:.3f}, Test R²={test_simple['r2']:.3f}, Gap={simple_gap:+.3f}")
        print(f"      Random Forest:       Val R²={val_rf['r2']:.3f}, Test R²={test_rf['r2']:.3f}, Gap={rf_gap:+.3f}")
        print(f"         Country Effect: {rf_country_imp:.3f} (Target: 0.2-0.4)")
        
        # Gap interpretation
        if abs(simple_gap) <= 0.15:
            print(f"      Simple: Gap corrected - within the scientific target")
        elif abs(simple_gap) <= 0.2:
            print(f"      Simple: Moderate gap - acceptable for educational data")
        else:
            print(f"      Simple: Gap still high - needs additional regularization")
            
        if abs(rf_gap) <= 0.15:
            print(f"      RF: Gap corrected - excellent regularization applied")
        elif abs(rf_gap) <= 0.2:
            print(f"      RF: Moderate gap - adequate regularization")
        else:
            print(f"      RF: Gap still high - consider additional regularization")
        
        _fit_predict_s = time.perf_counter() - _fit_t0

        return {
            'fold_load_s': _fold_load_s,
            'fit_predict_s': _fit_predict_s,
            'fold_id': fold_id,
            'architecture': 'task_graph',
            'mode': 'normal',
            'total_features': len(self.available_features),
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
        """Run the complete hierarchical analysis for the Data Lake architecture."""
        if getattr(self, '_needs_persist', False):
            self.ddf = self.ddf.persist()
            self._needs_persist = False
        print("Dask hierarchical analysis")
        print("   RidgeCV (Hoerl & Kennard 1970), Shrinkage James-Stein (Efron & Morris 1975)")

        _meta = SCIENTIFIC_CONFIG['hierarchical_model']
        all_results = {
            'architecture': 'task_graph',
            'version': 'hierarchical_analysis',
            'mode': 'normal',
            'target': self.target_col,
            'total_features': len(self.available_features),
            'corrections_applied': {
                'simple_hierarchical': (
                    f"RidgeCV with alphas logspace("
                    f"{_meta['ridge_alpha_log10_start']}, "
                    f"{_meta['ridge_alpha_log10_stop']}, "
                    f"{_meta['ridge_alpha_count']}) + Shrinkage James-Stein"),
                'random_forest': (
                    f"Regularized: n_est={_meta['rf_n_estimators']}, "
                    f"depth in {tuple(_meta['rf_max_depth_grid'])}, "
                    f"split={_meta['rf_min_samples_split']}, "
                    f"leaf in {tuple(_meta['rf_min_samples_leaf_grid'])}"),
                'regularization_approach': 'RidgeCV (Hoerl & Kennard 1970) + Shrinkage (Efron & Morris 1975)'
            },
            'folds': []
        }
        
        valid_folds = 0
        for fold_info in self.folds:
            _fold_t0 = time.perf_counter()
            fold_results = self.run_fold_analysis(fold_info)
            if fold_results is not None:
                fold_results['fold_duration_s'] = time.perf_counter() - _fold_t0
                all_results['folds'].append(fold_results)
                valid_folds += 1
                
                # Log of the results
                fold_id = fold_info['fold_id']
                for model_name, model_results in fold_results['models'].items():
                    val_r2 = model_results['val']['r2']
                    test_r2 = model_results['test']['r2']
                    gap = val_r2 - test_r2
                    print(f"   {model_name}: Val R²={val_r2:.3f}, Test R²={test_r2:.3f}, Gap={gap:+.3f}")
        
        if valid_folds == 0:
            print("No valid fold processed!")
            return all_results
        
        # Aggregate performance
        print("Aggregate SCHEMA-ON-READ performance:")
        
        for model_name in ['simple_hierarchical', 'random_forest_hierarchical']:
            val_r2s = [fold['models'][model_name]['val']['r2'] for fold in all_results['folds'] 
                      if model_name in fold['models']]
            test_r2s = [fold['models'][model_name]['test']['r2'] for fold in all_results['folds']
                       if model_name in fold['models']]
            
            if len(test_r2s) > 0:
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
                    print(f"      Effective regularization - Gap within the scientific target")
                elif abs_gap <= 0.2:
                    print(f"      Acceptable gap")
                else:
                    print(f"      Needs additional regularization")
                
                all_results[f'{model_name}_summary'] = {
                    'val_mean_r2': float(val_mean),
                    'test_mean_r2': float(test_mean),
                    'test_std_r2': float(test_std),
                    'generalization_gap': float(gap_mean)
                }
        
        print("SCHEMA-ON-READ hierarchical summary")
        
        # Best model
        if 'simple_hierarchical_summary' in all_results and 'random_forest_hierarchical_summary' in all_results:
            simple_test = all_results['simple_hierarchical_summary']['test_mean_r2']
            rf_test = all_results['random_forest_hierarchical_summary']['test_mean_r2']
            
            if rf_test > simple_test:
                print(f"   Best model: Hierarchical Random Forest")
                print(f"   Test R²: {rf_test:.3f}")
            else:
                print(f"   Best model: Simple Hierarchical")
                print(f"   Test R²: {simple_test:.3f}")

        self._write_prediction_artifact(all_results)

        results_file = f"{self.results_path}/hierarchical_analysis_task_graph_results_normal.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nDask results (NORMAL) saved: {results_file}")
        
        return all_results

if __name__ == "__main__":
    # Without an exit status the failure reaches the pipeline as a success:
    # subprocess check=True reads only the return code.
    try:
        model = HierarchicalModelTaskGraph()
        results = model.run_hierarchical_analysis()
        print("\nDask hierarchical analysis complete!")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
