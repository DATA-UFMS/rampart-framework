#!/usr/bin/env python3
"""Baseline model analysis for the Data Lake architecture."""
import time
import traceback

import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask
import json
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrameGroupBy.*')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from core.config import get_absolute_output_path
from core.models.hierarchical import (
    write_baseline_predictions as shared_write_baseline_predictions)
from core.prediction_store import PredictionRecorder, predictions_path
from core.scientific_config import SCIENTIFIC_CONFIG, setup_reproducibility

setup_reproducibility()



def _best_by_val_r2(fold_results: dict):
    """Best baseline by validation R2, ignoring the undefined ones.

    `max` compares against NaN returning False, so it was enough for the first
    item to have an undefined R2 for it to be elected the best -- and
    `best_test_r2` and `generalization_gap` derived from it. The choice came to
    depend on the dictionary insertion order, not on performance.
    """
    import math

    scored = [(name, data['val_r2']) for name, data in fold_results.items()
              if isinstance(data, dict) and 'val_r2' in data
              and data['val_r2'] is not None
              and not math.isnan(float(data['val_r2']))]
    if not scored:
        raise ValueError(
            "No baseline has a defined validation R2 in this fold; there is no "
            "best baseline to report."
        )
    return max(scored, key=lambda pair: pair[1])

class BaselineModelAnalysisTaskGraph:
    """Baseline model analysis for the Data Lake architecture."""

    def __init__(self):
        self._prediction_recorder = PredictionRecorder('task_graph')
        """Initialise the baseline analysis for the Data Lake architecture."""
        print("Initialising Dask baseline analysis")

        self.data_path = get_absolute_output_path("ml_pipeline/architectures/task_graph/prep/master_data_task_graph.parquet")
        self.folds_path = get_absolute_output_path("ml_pipeline/architectures/task_graph/prep/temporal_folds_task_graph.json")
        self.results_path = get_absolute_output_path("ml_pipeline/architectures/task_graph/models/baseline_results")
        
        os.makedirs(self.results_path, exist_ok=True)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data Lake data not found: {self.data_path}")
        if not os.path.exists(self.folds_path):
            raise FileNotFoundError(f"Data Lake folds not found: {self.folds_path}")

        print("   Loading data with Dask...")
        self.ddf = dd.read_parquet(self.data_path)
        self._needs_persist = True
        with open(self.folds_path, 'r') as f:
            self.folds_config = json.load(f)
            self.folds = self.folds_config['folds']
        
        self.target_col = 'dropout_rate_task_graph'
        self._load_data_summary()
    
    def _load_data_summary(self):
        """Load a summary of the data."""
        stats_tasks = {
            'total_rows': self.ddf.index.size,
            'year_min': self.ddf['year'].min(),
            'year_max': self.ddf['year'].max(),
            'countries_count': self.ddf['entity_id'].nunique(),
            'target_describe': self.ddf[self.target_col].describe(),
            'negative_target_count': (self.ddf[self.target_col] < 0).sum()
        }
        
        imputed_cols = [col for col in self.ddf.columns if '_imputed' in col and col != f'{self.target_col}_imputed']
        if imputed_cols:
            total_imputed = self.ddf[imputed_cols].sum(axis=1)
            stats_tasks['imputed_count'] = (total_imputed > 0).sum()
        
        keys = list(stats_tasks.keys())
        values = list(stats_tasks.values())
        computed_values = dask.compute(*values)
        computed_stats = dict(zip(keys, computed_values))
        
        total_rows = computed_stats['total_rows']
        year_min = computed_stats.get('year_min')
        year_max = computed_stats.get('year_max')
        countries_count = computed_stats.get('countries_count')
        target_stats = computed_stats.get('target_describe')
        negative_target = computed_stats.get('negative_target_count', 0)
        
        print(f"   Data loaded: {total_rows} records, {len(self.ddf.columns)} columns")
        print(f"   Period: {year_min}-{year_max}")
        print(f"   Countries: {countries_count}")
        print(f"   Target: {self.target_col}")
        print(f"   Folds: {len(self.folds)}")

        print(f"   Target stats: mean={target_stats['mean']:.2f}%, std={target_stats['std']:.2f}%")

        if imputed_cols:
            imputed_count = int(computed_stats['imputed_count'])
            print(f"   Data with imputation: {imputed_count}/{total_rows} ({(imputed_count/total_rows)*100:.1f}%)")

        if negative_target > 0:
            print(f"   Invalid target: {negative_target} negative values detected")
        else:
            target_min = target_stats['min']
            target_max = target_stats['max']
            print(f"   Valid target: range [{target_min:.2f}%, {target_max:.2f}%]")

        if not imputed_cols:
            print(f"   No imputation column found - using the original data")
        
        self._cached_basic_stats = computed_stats
    
    def analyze_target_distribution(self) -> Dict:
        """Analyse the Data Lake target distribution."""
        print(f"\nDask target distribution analysis")
        
        analysis = {}
        
        if hasattr(self, '_cached_basic_stats'):
            target_describe = self._cached_basic_stats['target_describe']
            year_min = self._cached_basic_stats['year_min']
            year_max = self._cached_basic_stats['year_max']
            
            missing_stats_batch = {
                'missing_count': self.ddf[self.target_col].isna().sum(),
                'missing_rate': self.ddf[self.target_col].isna().mean(),
                'unique_years': self.ddf['year'].nunique()
            }
            computed_missing = dask.compute(missing_stats_batch)[0]
            
            target_stats = {
                'architecture': 'task_graph',
                'target_variable': self.target_col,
                'mean': float(target_describe['mean']),
                'std': float(target_describe['std']),
                'min': float(target_describe['min']),
                'max': float(target_describe['max']),
                'missing_count': int(computed_missing['missing_count']),
                'missing_rate': float(computed_missing['missing_rate'])
            }
            unique_years = computed_missing['unique_years']
        else:
            target_stats_batch = {
                'target_describe': self.ddf[self.target_col].describe(),
                'missing_count': self.ddf[self.target_col].isna().sum(),
                'missing_rate': self.ddf[self.target_col].isna().mean(),
                'year_min': self.ddf['year'].min(),
                'year_max': self.ddf['year'].max(),
                'unique_years': self.ddf['year'].nunique()
            }
            computed_target = dask.compute(target_stats_batch)[0]
            
            target_describe = computed_target['target_describe']
            target_stats = {
                'architecture': 'task_graph',
                'target_variable': self.target_col,
                'mean': float(target_describe['mean']),
                'std': float(target_describe['std']),
                'min': float(target_describe['min']),
                'max': float(target_describe['max']),
                'missing_count': int(computed_target['missing_count']),
                'missing_rate': float(computed_target['missing_rate'])
            }
            year_min = computed_target['year_min']
            year_max = computed_target['year_max']
            unique_years = computed_target['unique_years']
        
        print(f"   Dask target ({self.target_col}):")
        print(f"      Mean: {target_stats['mean']:.2f}%")
        print(f"      SD: {target_stats['std']:.2f}%")
        print(f"      Range: {target_stats['min']:.2f}% - {target_stats['max']:.2f}%")
        print(f"      Missing: {target_stats['missing_count']} ({target_stats['missing_rate']:.1%})")
        
        analysis['target_stats'] = target_stats
        
        if unique_years > 1:
            temporal_stats_ddf = self.ddf.groupby('year')[self.target_col].agg([
                'count', 'mean', 'std', 'min', 'max'
            ])
            temporal_stats = temporal_stats_ddf.compute().round(2)
            
            print(f"\n   Dask temporal evolution:")
            if year_min in temporal_stats.index:
                print(f"      First year ({year_min}): {temporal_stats.loc[year_min, 'mean']:.1f}%")
            if year_max in temporal_stats.index:
                print(f"      Last year ({year_max}): {temporal_stats.loc[year_max, 'mean']:.1f}%")

            trend = temporal_stats['mean'].iloc[-1] - temporal_stats['mean'].iloc[0]
            print(f"      Trend: {trend:.1f}% over {year_max - year_min} years")
            
            analysis['temporal_stats'] = temporal_stats.to_dict()
        
        country_stats_ddf = self.ddf.groupby('entity_id')[self.target_col].agg([
            'count', 'mean', 'std', 'min', 'max'
        ])
        country_stats = country_stats_ddf.compute().round(2)
        
        print(f"\n   Variation by country (Dask):")
        print(f"      Lowest dropout: {country_stats['mean'].min():.1f}% ({country_stats['mean'].idxmin()})")
        print(f"      Highest dropout: {country_stats['mean'].max():.1f}% ({country_stats['mean'].idxmax()})")
        print(f"      Variation across countries: {country_stats['mean'].std():.1f}% (std)")
        
        analysis['country_stats'] = country_stats.to_dict()
        
        return analysis
    
    def _write_prediction_artifact(self) -> None:
        """Delegates to the shared implementation."""
        shared_write_baseline_predictions(self._prediction_recorder,
                                         architecture='task_graph')

    def test_baseline_models(self) -> Dict:
        """Test baseline models with walk-forward temporal validation."""
        print(f"\nBaselines with temporal validation")
        
        baseline_results = {}
        
        for fold_id, fold in enumerate(self.folds):
            _fold_t0 = time.perf_counter()
            # Initialised here, and not only at the boundary: in the SQL engine
            # the boundary sits inside a try, and depending on control flow
            # to define a name is how a NameError is produced.
            # None means not measured, and not zero, which would enter the sums.
            _fold_load_s = None
            _fit_t0 = _fold_t0
            print(f"\nFold {fold_id}: Train({fold['train_start']}-{fold['train_end']}) ->Val({fold['val_start']}-{fold['val_end']}) ->Test({fold['test_start']}-{fold['test_end']})")
            
            train_ddf = self.ddf[(self.ddf['year'] >= fold['train_start']) & (self.ddf['year'] <= fold['train_end'])]
            train_ddf = train_ddf[~((train_ddf['year'] >= fold['train_gap_start']) & (train_ddf['year'] <= fold['train_gap_end']))]
            val_ddf = self.ddf[(self.ddf['year'] >= fold['val_start']) & (self.ddf['year'] <= fold['val_end'])]
            test_ddf = self.ddf[(self.ddf['year'] >= fold['test_start']) & (self.ddf['year'] <= fold['test_end'])]
            test_ddf = test_ddf[~((test_ddf['year'] >= fold['val_gap_start']) & (test_ddf['year'] <= fold['val_gap_end']))]

            cols = ['entity_id', 'year', self.target_col]
            train_raw, val_raw, test_raw = dask.compute(
                train_ddf[cols].dropna(subset=[self.target_col]),
                val_ddf[cols].dropna(subset=[self.target_col]),
                test_ddf[cols].dropna(subset=[self.target_col]),
            )
            train_clean = train_raw.sort_values(['entity_id', 'year']).reset_index(drop=True)
            val_clean = val_raw.sort_values(['entity_id', 'year']).reset_index(drop=True)
            test_clean = test_raw.sort_values(['entity_id', 'year']).reset_index(drop=True)
            # Boundary of the decomposition: above is fold materialisation, which
            # belongs to the engine; below is the baseline fit, common to all three.
            _fold_load_s = time.perf_counter() - _fold_t0
            _fit_t0 = time.perf_counter()

            train_len, val_len, test_len = len(train_clean), len(val_clean), len(test_clean)
            print(f"   Data: Train={train_len}, Val={val_len}, Test={test_len}")
            print(f"    Gaps: Train-Val={fold['val_start']-fold['train_end']-1}yr, Val-Test={fold['test_start']-fold['val_end']-1}yr")

            if train_len == 0 or test_len == 0:
                print(f"   Fold {fold_id}: Insufficient data")
                continue

            y_train = train_clean[self.target_col].values
            y_val = val_clean[self.target_col].values
            y_test = test_clean[self.target_col].values
            global_mean = float(y_train.mean())

            def _mase_scale_from_train(df):
                try:
                    if df is None or df.empty:
                        return None
                    diffs = []
                    for _, g in df.sort_values(['entity_id','year']).groupby('entity_id'):
                        s = g[self.target_col].values
                        if len(s) >= 2:
                            d = np.abs(np.diff(s))
                            if len(d) > 0:
                                diffs.append(d)
                    if not diffs:
                        return None
                    return float(np.mean(np.concatenate(diffs)))
                except Exception:
                    return None

            mase_scale = _mase_scale_from_train(train_clean)

            fold_results = {}
            
            # Baseline 1: Global Mean
            
            val_pred_global = np.full(len(y_val), global_mean)
            test_pred_global = np.full(len(y_test), global_mean)
            
            val_r2_global = r2_score(y_val, val_pred_global)
            test_r2_global = r2_score(y_test, test_pred_global)
            
            fold_results['global_mean'] = {
                'val_r2': float(val_r2_global),
                'test_r2': float(test_r2_global),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred_global))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred_global))),
                'test_wape': float((np.abs(y_test - test_pred_global)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)),
                'test_mase': (float(np.mean(np.abs(y_test - test_pred_global))) / mase_scale) if (mase_scale and mase_scale > 0) else None,
                'mase_scale_train': mase_scale,
                'method': 'global_mean'
            }
            
            # Baseline 2: Linear Trend
            X_train_time = train_clean[['year']].values
            X_val_time = val_clean[['year']].values
            X_test_time = test_clean[['year']].values
            
            trend_model = LinearRegression()
            trend_model.fit(X_train_time, y_train)
            
            val_pred_trend = trend_model.predict(X_val_time)
            test_pred_trend = trend_model.predict(X_test_time)
            
            val_r2_trend = r2_score(y_val, val_pred_trend)
            test_r2_trend = r2_score(y_test, test_pred_trend)
            
            fold_results['linear_trend'] = {
                'val_r2': float(val_r2_trend),
                'test_r2': float(test_r2_trend),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred_trend))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred_trend))),
                'test_wape': float((np.abs(y_test - test_pred_trend)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)),
                'test_mase': (float(np.mean(np.abs(y_test - test_pred_trend))) / mase_scale) if (mase_scale and mase_scale > 0) else None,
                'mase_scale_train': mase_scale,
                'slope': float(trend_model.coef_[0]),
                'method': 'linear_trend'
            }
            
            # Baseline 3: Naive with Lag
            MIN_LAG = int(SCIENTIFIC_CONFIG.get('temporal_gap_years', 2))
            print(f"      Naive baseline...")

            val_pred_naive = []

            for _, val_row in val_clean.iterrows():
                country = val_row['entity_id']
                val_year = val_row['year']

                country_train = train_clean[train_clean['entity_id'] == country]
                country_hist = country_train[country_train['year'] <= val_year - MIN_LAG]

                if len(country_hist) > 0:
                    naive_val = country_hist.sort_values('year').iloc[-1][self.target_col]
                else:
                    naive_val = global_mean

                val_pred_naive.append(naive_val)

            test_pred_naive = []
            combined_clean = pd.concat([train_clean, val_clean], ignore_index=True)
            combined_mean = combined_clean[self.target_col].mean()

            for _, test_row in test_clean.iterrows():
                country = test_row['entity_id']
                test_year = test_row['year']

                country_combined = combined_clean[combined_clean['entity_id'] == country]
                country_hist = country_combined[country_combined['year'] <= test_year - MIN_LAG]

                if len(country_hist) > 0:
                    naive_test = country_hist.sort_values('year').iloc[-1][self.target_col]
                else:
                    naive_test = combined_mean

                test_pred_naive.append(naive_test)
            
            val_pred_naive = np.array(val_pred_naive)
            test_pred_naive = np.array(test_pred_naive)
            
            val_r2_naive = r2_score(y_val, val_pred_naive)
            test_r2_naive = r2_score(y_test, test_pred_naive)
            
            fold_results['naive_with_lag'] = {
                'val_r2': float(val_r2_naive),
                'test_r2': float(test_r2_naive),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred_naive))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred_naive))),
                'test_wape': float((np.abs(y_test - test_pred_naive)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)),
                'test_mase': (float(np.mean(np.abs(y_test - test_pred_naive))) / mase_scale) if (mase_scale and mase_scale > 0) else None,
                'mase_scale_train': mase_scale,
                'min_lag_years': MIN_LAG,
                'method': 'naive_persistence_with_scientific_lag'
            }
            
            # Baseline 4: Cross-Country Average
            print(f"      Cross-Country baseline...")

            val_pred_cross = []
            for _, val_row in val_clean.iterrows():
                country = val_row['entity_id']
                val_year = val_row['year']

                year_data = train_clean[train_clean['year'] <= val_year - MIN_LAG]

                if len(year_data) > 0:
                    country_means_dict = year_data.groupby('entity_id')[self.target_col].mean()
                    other_countries = country_means_dict[country_means_dict.index != country]

                    if len(other_countries) > 0:
                        cross_val = other_countries.mean()
                    else:
                        cross_val = global_mean
                else:
                    cross_val = global_mean

                val_pred_cross.append(cross_val)

            test_pred_cross = []
            for _, test_row in test_clean.iterrows():
                country = test_row['entity_id']
                test_year = test_row['year']

                year_data = combined_clean[combined_clean['year'] <= test_year - MIN_LAG]

                if len(year_data) > 0:
                    country_means_dict = year_data.groupby('entity_id')[self.target_col].mean()
                    other_countries = country_means_dict[country_means_dict.index != country]

                    if len(other_countries) > 0:
                        cross_test = other_countries.mean()
                    else:
                        cross_test = combined_mean
                else:
                    cross_test = combined_mean

                test_pred_cross.append(cross_test)
            
            val_pred_cross = np.array(val_pred_cross)
            test_pred_cross = np.array(test_pred_cross)
            
            val_r2_cross = r2_score(y_val, val_pred_cross)
            test_r2_cross = r2_score(y_test, test_pred_cross)
            
            fold_results['cross_entity'] = {
                'val_r2': float(val_r2_cross),
                'test_r2': float(test_r2_cross),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred_cross))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred_cross))),
                'test_wape': float((np.abs(y_test - test_pred_cross)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)),
                'test_mase': (float(np.mean(np.abs(y_test - test_pred_cross))) / mase_scale) if (mase_scale and mase_scale > 0) else None,
                'mase_scale_train': mase_scale,
                'min_lag_years': MIN_LAG,
                'method': 'cross_entity_average_excluding_target'
            }
            
            print(f"   Results (Val | Test):")
            print(f"      Global Mean:      R²={val_r2_global:.3f} | {test_r2_global:.3f}")
            print(f"      Linear Trend:     R²={val_r2_trend:.3f} | {test_r2_trend:.3f}")  
            print(f"      Naive+Lag>=2yr:   R²={val_r2_naive:.3f} | {test_r2_naive:.3f}")
            print(f"      Cross-Country:    R²={val_r2_cross:.3f} | {test_r2_cross:.3f}")
            
            best_val_baseline, best_val_r2 = _best_by_val_r2(fold_results)
            
            best_test_r2 = fold_results[best_val_baseline]['test_r2']
            generalization_gap = best_val_r2 - best_test_r2
            
            fold_results['best_baseline'] = {
                'model': best_val_baseline,
                'val_r2': best_val_r2,
                'test_r2': best_test_r2,
                'generalization_gap': generalization_gap
            }
            
            print(f"   Best baseline: {best_val_baseline} (Val: {best_val_r2:.3f} ->Test: {best_test_r2:.3f}, Gap: {generalization_gap:+.3f})")

            abs_gap = abs(generalization_gap)
            if abs_gap <= 0.05:
                print(f"      Excellent stability: Very low gap (<=0.05)")
            elif abs_gap <= 0.1:
                print(f"      Good stability: Gap within expectations (<=0.10)")
            elif abs_gap <= 0.15:
                print(f"      Moderate gap: Acceptable temporal variation ({abs_gap:.3f})")
            else:
                print(f"      High gap: Possible temporal instability ({abs_gap:.3f})")
            
            self._prediction_recorder.record(
                fold=fold_id, model='global_mean', y_true=y_test,
                y_pred=test_pred_global, entities=test_clean['entity_id'])
            self._prediction_recorder.record(
                fold=fold_id, model='linear_trend', y_true=y_test,
                y_pred=test_pred_trend, entities=test_clean['entity_id'])
            self._prediction_recorder.record(
                fold=fold_id, model='naive_with_lag', y_true=y_test,
                y_pred=test_pred_naive, entities=test_clean['entity_id'])
            self._prediction_recorder.record(
                fold=fold_id, model='cross_entity', y_true=y_test,
                y_pred=test_pred_cross, entities=test_clean['entity_id'])

            fold_results['fold_duration_s'] = time.perf_counter() - _fold_t0
            fold_results['fold_load_s'] = _fold_load_s
            fold_results['fit_predict_s'] = time.perf_counter() - _fit_t0
            baseline_results[f'fold_{fold_id}'] = fold_results

        self._write_prediction_artifact()

        return baseline_results

    def analyze_predictability(self, baseline_results: Dict) -> Dict:
        """Predictability analysis of the baseline models."""
        print("\nDask predictability analysis")
        
        baselines = ['global_mean', 'linear_trend', 'naive_with_lag', 'cross_entity']
        all_test_scores = {}
        all_val_scores = {}
        generalization_gaps = {}
        
        for baseline in baselines:
            test_r2_scores = []
            val_r2_scores = []
            gaps = []
            
            for fold_key in baseline_results:
                fold_data = baseline_results[fold_key]
                if baseline in fold_data:
                    test_r2_scores.append(fold_data[baseline]['test_r2'])
                    val_r2_scores.append(fold_data[baseline]['val_r2'])
                    gaps.append(fold_data[baseline]['val_r2'] - fold_data[baseline]['test_r2'])
            
            if test_r2_scores:
                all_test_scores[baseline] = {
                    'mean_r2': float(np.mean(test_r2_scores)),
                    'std_r2': float(np.std(test_r2_scores)),
                    'min_r2': float(np.min(test_r2_scores)),
                    'max_r2': float(np.max(test_r2_scores)),
                    'scores': test_r2_scores
                }
                
                all_val_scores[baseline] = {
                    'mean_r2': float(np.mean(val_r2_scores)),
                    'std_r2': float(np.std(val_r2_scores))
                }
                
                generalization_gaps[baseline] = {
                    'mean_gap': float(np.mean(gaps)),
                    'std_gap': float(np.std(gaps)),
                    'gaps': gaps
                }
        
        print("   Out-of-sample performance (TEST SET) of the baselines:")
        for baseline, stats in all_test_scores.items():
            val_stats = all_val_scores[baseline]
            gap_stats = generalization_gaps[baseline]
            print(f"      {baseline:20} | Test: R²={stats['mean_r2']:.3f}±{stats['std_r2']:.3f} | Val: R²={val_stats['mean_r2']:.3f} | Gap: {gap_stats['mean_gap']:+.3f}")
        
        if all_test_scores:
            best_baseline_overall = max(all_test_scores.keys(), key=lambda x: all_test_scores[x]['mean_r2'])
            best_mean_test_r2 = all_test_scores[best_baseline_overall]['mean_r2']
            best_mean_val_r2 = all_val_scores[best_baseline_overall]['mean_r2']
            best_generalization_gap = generalization_gaps[best_baseline_overall]['mean_gap']
            
            print(f"\n   Best baseline: {best_baseline_overall}")
            print(f"      Validation performance: R² = {best_mean_val_r2:.3f}")
            print(f"      Test performance:       R² = {best_mean_test_r2:.3f}")
            print(f"      Generalization gap:     {best_generalization_gap:+.3f}")
            
            predictability_analysis = {
                'architecture': 'task_graph',
                'methodology': 'scientific_baselines_with_temporal_lags',
                'validation_scores': all_val_scores,
                'test_scores': all_test_scores,
                'generalization_gaps': generalization_gaps,
                'best_baseline': best_baseline_overall,
                'best_test_r2': best_mean_test_r2,
                'best_val_r2': best_mean_val_r2,
                'generalization_gap': best_generalization_gap,
                'predictability_level': 'unknown'
            }

            # Predictability classification
            if best_mean_test_r2 < 0:
                predictability_analysis['predictability_level'] = 'very_low'
                print(f"   Very low predictability: R²_test < 0")
                print(f"      Interpretation: Model worse than a constant baseline")
            elif best_mean_test_r2 < 0.05:
                predictability_analysis['predictability_level'] = 'very_low'
                print(f"   Very low predictability: R²_test = {best_mean_test_r2:.3f}")
                print(f"      Interpretation: Almost no predictive power")
            elif best_mean_test_r2 < 0.15:
                predictability_analysis['predictability_level'] = 'low'
                print(f"   Low predictability: R²_test = {best_mean_test_r2:.3f}")
                print(f"      Interpretation: Limited predictive power")
            elif best_mean_test_r2 < 0.35:
                predictability_analysis['predictability_level'] = 'moderate'
                print(f"   Moderate predictability: R²_test = {best_mean_test_r2:.3f}")
                print(f"      Interpretation: Reasonable predictive power")
            else:
                predictability_analysis['predictability_level'] = 'good'
                print(f"   Good predictability: R²_test = {best_mean_test_r2:.3f}")
                print(f"      Interpretation: Good predictive power")
            
            avg_generalization_gap = np.mean([gap_data['mean_gap'] for gap_data in generalization_gaps.values()])
            abs_avg_gap = abs(avg_generalization_gap)
            
            if abs_avg_gap <= 0.05:
                print(f"   Excellent stability: Very low mean gap ({avg_generalization_gap:+.3f})")
                stability_level = "excellent"
            elif abs_avg_gap <= 0.1:
                print(f"   Good stability: Mean gap within expectations ({avg_generalization_gap:+.3f})")
                stability_level = "good"
            elif abs_avg_gap <= 0.15:
                print(f"   Moderate stability: Acceptable temporal variation ({avg_generalization_gap:+.3f})")
                stability_level = "moderate"
            else:
                print(f"   Instability detected: High mean gap ({avg_generalization_gap:+.3f})")
                print(f"      Possible overfitting or strong temporal variation")
                stability_level = "low"
            
            predictability_analysis['stability_analysis'] = {
                'avg_generalization_gap': float(avg_generalization_gap),
                'stability_level': stability_level
            }
            
        else:
            predictability_analysis = {
                'architecture': 'task_graph',
                'baseline_scores': {},
                'predictability_level': 'unknown'
            }
        
        return predictability_analysis
    
    def save_results(self, target_analysis: Dict, baseline_results: Dict,
                    predictability_analysis: Dict):
        """Save the results of the Data Lake analysis."""
        print(f"\nSaving Dask results...")
        
        full_results = {
            'architecture': 'task_graph',
            'target_variable': self.target_col,
            'data_source': self.data_path,
            'target_distribution_analysis': target_analysis,
            'baseline_model_results': baseline_results,
            'predictability_analysis': predictability_analysis,
            'summary': {
                'total_folds_analyzed': len(baseline_results),
                'best_baseline_model': predictability_analysis.get('best_baseline', 'unknown'),
                'best_baseline_r2': predictability_analysis.get('best_test_r2', 0),
                'predictability_level': predictability_analysis.get('predictability_level', 'unknown'),
            }
        }
        
        results_file = f"{self.results_path}/baseline_analysis_task_graph_results.json"
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"   Dask results saved: {results_file}")
        
        return full_results
    
    def run_complete_analysis(self):
        """Run the complete Data Lake baseline analysis."""
        if getattr(self, '_needs_persist', False):
            self.ddf = self.ddf.persist()
            self._needs_persist = False
        print(f"Complete analysis - Dask architecture")
        
        try:
            target_analysis = self.analyze_target_distribution()
            baseline_results = self.test_baseline_models()
            predictability_analysis = self.analyze_predictability(baseline_results)
            results = self.save_results(target_analysis, baseline_results, 
                                       predictability_analysis)
            
            print(f"\nExecutive summary - Dask architecture:")
            print(f"   Research: Comparison of ML architectures for dropout")
            print(f"   Architecture: Dask (Distributed Processing)")
            print(f"   Target: {self.target_col}")
            print(f"   Predictability: {predictability_analysis.get('predictability_level', 'unknown').upper()}")
            print(f"   Best baseline: {predictability_analysis.get('best_baseline', 'unknown')}")
            print(f"   Test R²: {predictability_analysis.get('best_test_r2', 0):.3f}")
            
            gap = predictability_analysis.get('generalization_gap', 0)
            
            if abs(gap) <= 0.05:
                gap_status = f"Gap: {gap:+.3f} (excellent stability)"
            elif abs(gap) <= 0.1:
                gap_status = f"Gap: {gap:+.3f} (good stability)"
            elif abs(gap) <= 0.15:
                gap_status = f"Gap: {gap:+.3f} (moderate stability)"
            else:
                gap_status = f"Gap: {gap:+.3f} (requires attention)"

            print(f"   {gap_status}")

            stability = predictability_analysis.get('stability_analysis', {}).get('stability_level', 'unknown')
            print(f"   Stability: {stability}")
            
            return results
            
        except Exception as e:
            print(f"\nError in the Dask analysis: {e}")
            traceback.print_exc()
            return {
                'architecture': 'task_graph',
                'status': 'failed',
                'error': str(e)
            }

if __name__ == "__main__":
    analyzer = BaselineModelAnalysisTaskGraph()
    results = analyzer.run_complete_analysis()
