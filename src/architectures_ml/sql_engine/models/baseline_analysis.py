#!/usr/bin/env python3
"""
Baseline model analysis for the Data Warehouse architecture.

Module for comparative analysis using the ML Data Warehouse Consumer pattern
with direct queries to views and walk-forward temporal validation with gaps.

Technical summary:
- ML Data Warehouse Consumer pattern (Connection Manager + temporal views)
- Temporal validation with gaps (minimum of 2 years) to prevent leakage
- Baseline models: global mean, linear trend, naive with lag, cross-country
- Access via native SQL (DuckDB) with no file I/O during training
"""
import time

import pandas as pd
import numpy as np
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

core_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'core')
core_path = os.path.abspath(core_path)
if core_path not in sys.path:
    sys.path.append(core_path)

project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import get_absolute_output_path
from core.models.hierarchical import (
    write_baseline_predictions as shared_write_baseline_predictions)
from core.prediction_store import PredictionRecorder, predictions_path
from core.scientific_config import SCIENTIFIC_CONFIG, setup_reproducibility

setup_reproducibility()

sql_engine_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'collection', 'sql_engine')
sql_engine_path = os.path.abspath(sql_engine_path)
if sql_engine_path not in sys.path:
    sys.path.append(sql_engine_path)

from connection_manager import DuckDBConnectionManager, SQLProcessingError



def _best_by_val_r2(fold_results: dict):
    """Best baseline by validation R2, ignoring the undefined ones.

    `max` compares against NaN by returning False, so it was enough for the
    first item to have an undefined R2 for it to be elected the best -- and
    `best_test_r2` and `generalization_gap` derived from it. The choice came to
    depend on insertion order in the dictionary, not on performance.
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

class BaselineModelAnalysisSqlEngine:
    """
    ML Data Warehouse Consumer - Baseline Analysis.

    Implements the ML Data Warehouse Consumer pattern for the scientific
    analysis of baseline models with temporal validation, using:
    - Direct queries via the Connection Manager
    - Consumption of Feature Store views for ML training
    - Connection pooling for performance in ML workloads
    - Elimination of file I/O during training

    Attributes:
        folds_path (str): Path to the temporal fold configuration
        results_path (str): Directory in which to save results
        db_path (str): Path to the DuckDB database
        conn_manager (DuckDBConnectionManager): Connection manager
        target_col (str): Name of the target column for prediction
        folds (list): List of temporal fold configurations
    """
    
    def __init__(self):
        self._prediction_recorder = PredictionRecorder('sql_engine')
        """Initializes the baseline analysis for the Data Warehouse architecture."""
        print("Initializing DuckDB baseline analysis")
        
        dataset_name = os.environ.get('DATASET_NAME', 'worldbank')
        self.folds_path = get_absolute_output_path("ml_pipeline/architectures/sql_engine/prep/temporal_folds_sql_engine.json")
        self.results_path = get_absolute_output_path("ml_pipeline/architectures/sql_engine/models")
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
            raise RuntimeError(f"Failed to initialize the Connection Manager: {e}")
        with open(self.folds_path, 'r') as f:
            self.folds_config = json.load(f)
            self.folds = self.folds_config['folds']
        
        self.target_col = 'dropout_rate_sql_engine'
        self._ensure_target_column()
        
        self._verify_feature_store_views()
        self._load_data_summary_from_views()

    def _ensure_target_column(self) -> None:
        """Ensures the target column exists in analytics_wide.

        If absent, creates and populates it as
        100 - target_source_rate, preserving NULLs/invalid values.
        """
        try:
            exists = self.conn_manager.execute_scalar(
                f"SELECT COUNT(*) > 0 FROM information_schema.columns WHERE table_name = 'analytics_wide' AND column_name = '{self.target_col}'"
            )
            if not exists:
                print(f"   Target '{self.target_col}' missing - creating column...")
                self.conn_manager.execute_sql_no_return(
                    f"ALTER TABLE analytics_wide ADD COLUMN IF NOT EXISTS {self.target_col} DOUBLE"
                )
                self.conn_manager.execute_sql_no_return(
                    f"""
                    UPDATE analytics_wide
                    SET {self.target_col} = CASE
                        WHEN target_source_rate IS NULL THEN NULL
                        WHEN target_source_rate < 0 OR target_source_rate > 100 THEN NULL
                        ELSE 100 - target_source_rate
                    END
                    """
                )
                print("   Target created/populated in analytics_wide")
        except SQLProcessingError as e:
            raise RuntimeError(f"Failed to ensure the target column: {e}")
    
    def _verify_feature_store_views(self):
        """Check whether the temporal views exist in the Data Warehouse."""
        print("   Checking temporal views...")
        
        try:
            table_exists = self.conn_manager.execute_scalar(f"""
                SELECT COUNT(*) > 0 
                FROM information_schema.tables 
                WHERE table_name = 'analytics_wide'
            """)
            
            if not table_exists:
                raise RuntimeError(f"Base table not found: analytics_wide")
            
            views_found = 0
            for fold_id in range(len(self.folds)):
                for split in ['train', 'val', 'test']:
                    view_name = f"vw_fold_{fold_id}_{split}"
                    view_exists = self.conn_manager.execute_scalar(f"""
                        SELECT COUNT(*) > 0 
                        FROM information_schema.views 
                        WHERE table_name = '{view_name}'
                    """)
                    
                    if view_exists:
                        views_found += 1
            
            if views_found == 0:
                print(f"   Warning: no temporal view found")
                print(f"   Run setup.py first to create the temporal views")
                print(f"   Using the base table as a fallback")
            else:
                print(f"   Temporal views checked: {views_found} views found")
            
        except SQLProcessingError as e:
            raise RuntimeError(f"Error checking the temporal views: {e}")
    
    def _load_data_summary_from_views(self):
        """Load the data summary via direct queries to the views."""
        print("   Loading the data summary via views...")
        
        try:
            # Prefer the feature store view (guarantees the target is present)
            view_base = 'vw_selected_features'
            total_records = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM {view_base} WHERE {self.target_col} IS NOT NULL")
            min_year = self.conn_manager.execute_scalar(f"SELECT MIN(year) FROM {view_base} WHERE {self.target_col} IS NOT NULL")
            max_year = self.conn_manager.execute_scalar(f"SELECT MAX(year) FROM {view_base} WHERE {self.target_col} IS NOT NULL")
            total_countries = self.conn_manager.execute_scalar(f"SELECT COUNT(DISTINCT entity_id) FROM {view_base} WHERE {self.target_col} IS NOT NULL")
            target_mean = self.conn_manager.execute_scalar(f"SELECT AVG({self.target_col}) FROM {view_base} WHERE {self.target_col} IS NOT NULL") or 0
            target_std = self.conn_manager.execute_scalar(f"SELECT STDDEV({self.target_col}) FROM {view_base} WHERE {self.target_col} IS NOT NULL") or 0
            target_min = self.conn_manager.execute_scalar(f"SELECT MIN({self.target_col}) FROM {view_base} WHERE {self.target_col} IS NOT NULL") or 0
            target_max = self.conn_manager.execute_scalar(f"SELECT MAX({self.target_col}) FROM {view_base} WHERE {self.target_col} IS NOT NULL") or 0
            negative_target = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM {view_base} WHERE {self.target_col} < 0") or 0
            target_exists = self.conn_manager.execute_scalar(f"SELECT COUNT(*) > 0 FROM information_schema.columns WHERE table_name = '{view_base}' AND column_name = '{self.target_col}'")
            
            print(f"   Data via views: {total_records} observations")
            print(f"   Period: {min_year}-{max_year}")
            print(f"   Countries: {total_countries}")
            print(f"   Target: {self.target_col}")
            print(f"   Folds: {len(self.folds)}")
            
            if not target_exists:
                raise ValueError(f"Target {self.target_col} not found in the analytics_wide table")
            
            print(f"   Target stats: mean={target_mean:.2f}%, std={target_std:.2f}%")
            
            if negative_target > 0:
                print(f"   Invalid target: {negative_target} negative values detected")
            else:
                print(f"   Valid target: range [{target_min:.2f}%, {target_max:.2f}%]")
            
            self._cached_target_stats = {
                'mean': target_mean,
                'std': target_std,
                'min': target_min,
                'max': target_max,
                'total_records': total_records,
                'min_year': min_year,
                'max_year': max_year,
                'total_countries': total_countries
            }
                
        except SQLProcessingError as e:
            raise RuntimeError(f"Error loading the summary via views: {e}")
    
    def _load_ml_fold_data(self, fold_id: int, split: str) -> pd.DataFrame:
        """
        Load the fold data via the ML Data Warehouse Consumer pattern with SQL-First dynamic queries.

        Implements the ML Data Warehouse Consumer pattern:
        - Direct queries to the temporal views after collinearity filtering
        - Dynamic feature discovery via information_schema (100% SQL)
        - Connection pooling for performance in ML workloads
        - Keeps the SQL-first paradigm of the Data Warehouse

        Args:
            fold_id: ID of the temporal fold
            split: 'train', 'val', or 'test'

        Returns:
            DataFrame with the ML features available after collinearity filtering
        """
        cache_key = f"fold_{fold_id}_{split}"
        if not hasattr(self, '_fold_data_cache'):
            self._fold_data_cache = {}
        
        if cache_key in self._fold_data_cache:
            return self._fold_data_cache[cache_key]
        
        fold = self.folds[fold_id]
        
        if split == 'train':
            year_start, year_end = fold['train_start'], fold['train_end']
        elif split == 'val':
            year_start, year_end = fold['val_start'], fold['val_end']
        elif split == 'test':
            year_start, year_end = fold['test_start'], fold['test_end']
        else:
            raise ValueError(f"Invalid split: {split}")
        
        try:
            view_name = f"vw_fold_{fold_id}_{split}"
            
            view_exists = self.conn_manager.execute_scalar(f"""
                SELECT COUNT(*) > 0
                FROM information_schema.views
                WHERE table_name = '{view_name}'
            """)
            
            if view_exists:
                print(f"      Using temporal view: {view_name}")
                
                # Dynamic feature discovery via SQL
                try:
                    available_features_query = f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = '{view_name}'
                        AND column_name NOT IN ('entity_id', 'year', '{self.target_col}')
                        ORDER BY column_name
                    """
                    
                    feature_result = self.conn_manager.execute_sql(available_features_query)
                    available_features = feature_result['column_name'].tolist()
                    
                    if available_features:
                        feature_list = ', '.join(available_features)
                        print(f"      Features discovered via SQL: {len(available_features)} columns")
                        
                        # Dynamic query with the features actually available (100% SQL)
                        query = f"""
                            SELECT
                                entity_id,
                                year,
                                {self.target_col},
                                {feature_list}
                            FROM {view_name}
                            WHERE {self.target_col} IS NOT NULL
                            ORDER BY entity_id, year
                        """
                    else:
                        print(f"      No feature discovered, using basic columns")
                        query = f"""
                            SELECT
                                entity_id,
                                year,
                                {self.target_col}
                            FROM {view_name}
                            WHERE {self.target_col} IS NOT NULL
                            ORDER BY entity_id, year
                        """
                        
                except SQLProcessingError as e:
                    print(f"      Error discovering features via SQL: {e}")
                    print(f"      Fallback: using a query with basic columns")
                    query = f"""
                        SELECT
                            entity_id,
                            year,
                            {self.target_col}
                        FROM {view_name}
                        WHERE {self.target_col} IS NOT NULL
                        ORDER BY entity_id, year
                    """
                    
            else:
                print(f"      Fallback: view {view_name} not found")
                print(f"      Using the base table: analytics_wide (years {year_start}-{year_end})")
                print(f"      Run setup.py first to create the temporal views")
                
                # Feature discovery from the base table (SQL-first)
                try:
                    base_features_query = f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = 'analytics_wide'
                        AND column_name NOT IN ('entity_id', 'year', '{self.target_col}', 'data_source',
                                               'data_completeness_score', 'etl_batch_id', 'collection_timestamp',
                                               'entity_name', 'entity_stratum')
                        AND data_type IN ('DOUBLE', 'INTEGER', 'FLOAT', 'DECIMAL', 'NUMERIC')
                        ORDER BY column_name
                    """
                    
                    base_feature_result = self.conn_manager.execute_sql(base_features_query)
                    base_available_features = base_feature_result['column_name'].tolist()
                    
                    if base_available_features:
                        base_feature_list = ', '.join(base_available_features)
                        print(f"      Features discovered from the base table: {len(base_available_features)} columns")
                        
                        query = f"""
                            SELECT
                                entity_id,
                                year,
                                {self.target_col},
                                {base_feature_list}
                            FROM analytics_wide
                            WHERE {self.target_col} IS NOT NULL
                              AND year >= {year_start}
                              AND year <= {year_end}
                            ORDER BY entity_id, year
                        """
                    else:
                        query = f"""
                            SELECT
                                entity_id,
                                year,
                                {self.target_col}
                            FROM analytics_wide
                            WHERE {self.target_col} IS NOT NULL
                              AND year >= {year_start}
                              AND year <= {year_end}
                            ORDER BY entity_id, year
                        """
                        
                except SQLProcessingError as e:
                    print(f"      Error discovering features from the base table: {e}")
                    query = f"""
                        SELECT
                            entity_id,
                            year,
                            {self.target_col}
                        FROM analytics_wide
                        WHERE {self.target_col} IS NOT NULL
                          AND year >= {year_start}
                          AND year <= {year_end}
                        ORDER BY entity_id, year
                    """
            
            df = self.conn_manager.execute_sql(query)
            
            if df.empty:
                data_source = view_name if view_exists else f"analytics_wide (years {year_start}-{year_end})"
                raise SQLProcessingError(f"No data returned from {data_source}")
            
            self._fold_data_cache[cache_key] = df
            
            feature_count = len([col for col in df.columns if col not in ['entity_id', 'year', self.target_col]])
            
            if view_exists:
                print(f"      Data from the temporal view: {len(df)} records, {feature_count} features")
            else:
                print(f"      Data from the base table: {len(df)} records, {feature_count} features")
            
            return df
            
        except SQLProcessingError as e:
            raise RuntimeError(f"Error loading data for fold {fold_id} split {split}: {e}")
    
    def cleanup(self):
        """Release the connection manager's resources."""
        if hasattr(self, 'conn_manager') and self.conn_manager:
            try:
                self.conn_manager.close_connection()
                print("   Connection Manager closed")
            except Exception as e:
                print(f"   Error closing the Connection Manager: {e}")
    
    def analyze_target_distribution(self) -> Dict:
        """
        Analyze the target distribution via Data Warehouse views.

        Returns:
            Dict: Statistics of the target distribution, including temporal and per-country
        """
        print(f"\nDuckDB target distribution analysis")
        
        analysis = {}
        
        try:
            if hasattr(self, '_cached_target_stats'):
                cached_stats = self._cached_target_stats
                
                missing_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM analytics_wide WHERE {self.target_col} IS NULL")
                total_count = self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
                
                target_stats = {
                    'architecture': 'sql_engine',
                    'target_variable': self.target_col,
                    'mean': cached_stats['mean'],
                    'std': cached_stats['std'],
                    'min': cached_stats['min'],
                    'max': cached_stats['max'],
                    'missing_count': int(missing_count),
                    'missing_rate': float(missing_count / total_count if total_count > 0 else 0)
                }
            else:
                target_mean = self.conn_manager.execute_scalar(f"SELECT AVG({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL") or 0
                target_std = self.conn_manager.execute_scalar(f"SELECT STDDEV({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL") or 0
                target_min = self.conn_manager.execute_scalar(f"SELECT MIN({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL") or 0
                target_max = self.conn_manager.execute_scalar(f"SELECT MAX({self.target_col}) FROM analytics_wide WHERE {self.target_col} IS NOT NULL") or 0
                missing_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM analytics_wide WHERE {self.target_col} IS NULL")
                total_count = self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
                
                target_stats = {
                    'architecture': 'sql_engine',
                    'target_variable': self.target_col,
                    'mean': float(target_mean),
                    'std': float(target_std),
                    'min': float(target_min),
                    'max': float(target_max),
                    'missing_count': int(missing_count),
                    'missing_rate': float(missing_count / total_count if total_count > 0 else 0)
                }
            
            print(f"   Target ({self.target_col}):")
            print(f"      Mean: {target_stats['mean']:.2f}%")
            print(f"      Std: {target_stats['std']:.2f}%")
            print(f"      Range: {target_stats['min']:.2f}% - {target_stats['max']:.2f}%")
            print(f"      Missing: {target_stats['missing_count']} ({target_stats['missing_rate']:.1%})")
            
            analysis['target_stats'] = target_stats
            
            temporal_query = f"""
                SELECT 
                    year,
                    COUNT(*) as count,
                    AVG({self.target_col}) as mean,
                    STDDEV({self.target_col}) as std,
                    MIN({self.target_col}) as min,
                    MAX({self.target_col}) as max
                FROM analytics_wide 
                WHERE {self.target_col} IS NOT NULL
                GROUP BY year
                ORDER BY year
            """
            temporal_df = self.conn_manager.execute_sql(temporal_query)
            
            if not temporal_df.empty:
                if hasattr(self, '_cached_target_stats'):
                    first_year = self._cached_target_stats['min_year']
                    last_year = self._cached_target_stats['max_year']
                else:
                    first_year = self.conn_manager.execute_scalar(f"SELECT MIN(year) FROM analytics_wide WHERE {self.target_col} IS NOT NULL")
                    last_year = self.conn_manager.execute_scalar(f"SELECT MAX(year) FROM analytics_wide WHERE {self.target_col} IS NOT NULL")
                
                first_mean = self.conn_manager.execute_scalar(f"SELECT AVG({self.target_col}) FROM analytics_wide WHERE year = {first_year} AND {self.target_col} IS NOT NULL")
                last_mean = self.conn_manager.execute_scalar(f"SELECT AVG({self.target_col}) FROM analytics_wide WHERE year = {last_year} AND {self.target_col} IS NOT NULL")
                
                print(f"\n   DuckDB temporal evolution:")
                print(f"      First year ({first_year}): {first_mean:.1f}%")
                print(f"      Last year ({last_year}): {last_mean:.1f}%")
                
                trend = last_mean - first_mean
                print(f"      Trend: {trend:.1f}% over {last_year - first_year} years")
                
                analysis['temporal_stats'] = temporal_df.to_dict('records')
            
            country_query = f"""
                SELECT 
                    entity_id,
                    COUNT(*) as count,
                    AVG({self.target_col}) as mean,
                    STDDEV({self.target_col}) as std,
                    MIN({self.target_col}) as min,
                    MAX({self.target_col}) as max
                FROM analytics_wide 
                WHERE {self.target_col} IS NOT NULL
                GROUP BY entity_id
                ORDER BY mean
            """
            country_df = self.conn_manager.execute_sql(country_query)
            
            if not country_df.empty:
                min_dropout_mean = self.conn_manager.execute_scalar(f"SELECT MIN(avg_dropout) FROM (SELECT entity_id, AVG({self.target_col}) as avg_dropout FROM analytics_wide WHERE {self.target_col} IS NOT NULL GROUP BY entity_id)")
                max_dropout_mean = self.conn_manager.execute_scalar(f"SELECT MAX(avg_dropout) FROM (SELECT entity_id, AVG({self.target_col}) as avg_dropout FROM analytics_wide WHERE {self.target_col} IS NOT NULL GROUP BY entity_id)")
                min_dropout_country_code = self.conn_manager.execute_scalar(f"SELECT entity_id FROM (SELECT entity_id, AVG({self.target_col}) as avg_dropout FROM analytics_wide WHERE {self.target_col} IS NOT NULL GROUP BY entity_id ORDER BY avg_dropout LIMIT 1)")
                max_dropout_country_code = self.conn_manager.execute_scalar(f"SELECT entity_id FROM (SELECT entity_id, AVG({self.target_col}) as avg_dropout FROM analytics_wide WHERE {self.target_col} IS NOT NULL GROUP BY entity_id ORDER BY avg_dropout DESC LIMIT 1)")
                country_variation = self.conn_manager.execute_scalar(f"SELECT STDDEV(avg_dropout) FROM (SELECT entity_id, AVG({self.target_col}) as avg_dropout FROM analytics_wide WHERE {self.target_col} IS NOT NULL GROUP BY entity_id)")
                
                print(f"\n   Variation across countries:")
                print(f"      Lowest dropout: {min_dropout_mean:.1f}% ({min_dropout_country_code})")
                print(f"      Highest dropout: {max_dropout_mean:.1f}% ({max_dropout_country_code})")
                print(f"      Variation between countries: {country_variation:.1f}% (std)")
                
                analysis['country_stats'] = country_df.to_dict('records')
            
            return analysis
            
        except SQLProcessingError as e:
            print(f"   [ERROR] Distribution analysis: {e}")
            return {
                'architecture': 'sql_engine',
                'error': str(e),
                'target_stats': {}
            }
    
    def _write_prediction_artifact(self) -> None:
        """Delegates to the shared implementation."""
        shared_write_baseline_predictions(self._prediction_recorder,
                                         architecture='sql_engine')

    def test_baseline_models(self) -> Dict:
        """
        Test the scientific baseline models via the Data Warehouse.

        Methodological corrections implemented:
        - Minimum lag of 2 years to avoid temporal leakage
        - Correct scientific walk-forward validation
        - Appropriate temporal gaps between the sets

        Returns:
            Dict: Baseline results with temporal validation
        """
        print(f"\nBaselines with temporal validation")
        
        baseline_results = {}
        
        for fold_id, fold in enumerate(self.folds):
            _fold_t0 = time.perf_counter()
            # Initialized here, and not only at the boundary: in the SQL engine
            # the boundary sits inside a try, and depending on control flow to
            # define a name is how a NameError gets produced. None means not
            # measured, and not zero, which would enter the sums.
            _fold_load_s = None
            _fit_t0 = _fold_t0
            print(f"\nFold {fold_id}: Train({fold['train_start']}-{fold['train_end']}) -> Val({fold['val_start']}-{fold['val_end']}) -> Test({fold['test_start']}-{fold['test_end']})")

            try:
                train_data = self._load_ml_fold_data(fold_id, 'train')
                val_data = self._load_ml_fold_data(fold_id, 'val')
                test_data = self._load_ml_fold_data(fold_id, 'test')

                print(f"   Data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
                print(f"   Gaps: Train-Val={fold['val_start']-fold['train_end']-1}yr, Val-Test={fold['test_start']-fold['val_end']-1}yr")
                print(f"   Features available: {len(train_data.columns)} columns")
                
                train_clean = train_data
                val_clean = val_data
                test_clean = test_data
                # Boundary of the decomposition: above is materialization of the
                # fold, which belongs to the engine; below is the fitting of the
                # baselines, common to all three.
                _fold_load_s = time.perf_counter() - _fold_t0
                _fit_t0 = time.perf_counter()
                
            except Exception as e:
                print(f"   Error loading data for fold {fold_id}: {e}")
                continue
            
            if len(train_clean) == 0 or len(test_clean) == 0:
                print(f"   Fold {fold_id}: insufficient data")
                continue
            
            y_train = train_clean[self.target_col]
            y_val = val_clean[self.target_col] 
            y_test = test_clean[self.target_col]

            # MASE scale from the training window (absolute differences per country)
            def _mase_scale_from_train(df):
                try:
                    if df is None or len(df) == 0:
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
            
            global_mean = y_train.mean()
            
            val_pred_global = np.full(len(y_val), global_mean)
            test_pred_global = np.full(len(y_test), global_mean)
            
            val_r2_global = r2_score(y_val, val_pred_global)
            test_r2_global = r2_score(y_test, test_pred_global)
            
            fold_results['global_mean'] = {
                'val_r2': float(val_r2_global),
                'test_r2': float(test_r2_global),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred_global))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred_global))),
                'test_wape': float((np.abs(y_test - test_pred_global)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)) if hasattr(y_test, 'sum') else None,
                'test_mase': (float(np.mean(np.abs(y_test - test_pred_global))) / mase_scale) if (mase_scale and mase_scale > 0) else None,
                'mase_scale_train': mase_scale,
                'method': 'global_mean'
            }
            
            X_train_time = train_clean[['year']].values
            trend_model = LinearRegression()  # LinearRegression does not accept random_state
            trend_model.fit(X_train_time, y_train)
            
            X_val_time = val_clean[['year']].values
            X_test_time = test_clean[['year']].values
            
            val_pred_trend = trend_model.predict(X_val_time)
            test_pred_trend = trend_model.predict(X_test_time)
            
            val_r2_trend = r2_score(y_val, val_pred_trend)
            test_r2_trend = r2_score(y_test, test_pred_trend)
            
            fold_results['linear_trend'] = {
                'val_r2': float(val_r2_trend),
                'test_r2': float(test_r2_trend),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred_trend))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred_trend))),
                'test_wape': float((np.abs(y_test - test_pred_trend)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)) if hasattr(y_test, 'sum') else None,
                'test_mase': (float(np.mean(np.abs(y_test - test_pred_trend))) / mase_scale) if (mase_scale and mase_scale > 0) else None,
                'mase_scale_train': mase_scale,
                'slope': float(trend_model.coef_[0]),
                'method': 'linear_trend'
            }
            
            MIN_LAG = int(SCIENTIFIC_CONFIG.get('temporal_gap_years', 2))
            
            print(f"      Naive baseline...")
            
            val_pred_naive = []
            for _, val_row in val_clean.iterrows():
                country = val_row['entity_id']
                val_year = val_row['year']
                
                country_history = train_clean[
                    (train_clean['entity_id'] == country) & 
                    (train_clean['year'] <= val_year - MIN_LAG)
                ].sort_values('year')
                
                if len(country_history) > 0:
                    naive_val = country_history[self.target_col].iloc[-1]
                else:
                    naive_val = global_mean
                
                val_pred_naive.append(naive_val)
            
            test_pred_naive = []
            combined_history = pd.concat([train_clean, val_clean], ignore_index=True)
            
            for _, test_row in test_clean.iterrows():
                country = test_row['entity_id']
                test_year = test_row['year']
                
                country_history = combined_history[
                    (combined_history['entity_id'] == country) & 
                    (combined_history['year'] <= test_year - MIN_LAG)
                ].sort_values('year')
                
                if len(country_history) > 0:
                    naive_test = country_history[self.target_col].iloc[-1]
                else:
                    naive_test = combined_history[self.target_col].mean()
                
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
                'min_lag_years': MIN_LAG,
                'method': 'naive_persistence_with_scientific_lag'
            }
            
            print(f"      Cross-Country baseline...")
            
            val_pred_cross = []
            for _, val_row in val_clean.iterrows():
                country = val_row['entity_id']
                val_year = val_row['year']

                year_data = train_clean[
                    train_clean['year'] <= val_year - MIN_LAG
                ]

                if len(year_data) > 0:
                    country_means = year_data.groupby('entity_id')[self.target_col].mean()
                    other_countries = country_means[country_means.index != country]
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

                year_data = combined_history[
                    combined_history['year'] <= test_year - MIN_LAG
                ]

                if len(year_data) > 0:
                    country_means = year_data.groupby('entity_id')[self.target_col].mean()
                    other_countries = country_means[country_means.index != country]
                    if len(other_countries) > 0:
                        cross_test = other_countries.mean()
                    else:
                        cross_test = combined_history[self.target_col].mean()
                else:
                    cross_test = combined_history[self.target_col].mean()

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
                'test_wape': float((np.abs(y_test - test_pred_cross)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)) if hasattr(y_test, 'sum') else None,
                'test_mase': (float(np.mean(np.abs(y_test - test_pred_cross))) / mase_scale) if (mase_scale and mase_scale > 0) else None,
                'mase_scale_train': mase_scale,
                'min_lag_years': MIN_LAG,
                'method': 'cross_entity_average_excluding_target'
            }
            
            # Aggregate WAPE/MASE for Naive
            try:
                test_wape_naive = float((np.abs(y_test - test_pred_naive)).sum() / np.maximum(np.abs(y_test).sum(), 1e-12)) if hasattr(y_test, 'sum') else None
                test_mase_naive = (float(np.mean(np.abs(y_test - test_pred_naive))) / mase_scale) if (mase_scale and mase_scale > 0) else None
            except Exception:
                test_wape_naive = None
                test_mase_naive = None
            fold_results['naive_with_lag'].update({
                'test_wape': test_wape_naive,
                'test_mase': test_mase_naive,
                'mase_scale_train': mase_scale
            })
            
            print(f"   Results (Val | Test):")
            print(f"      Global Mean:      R²={val_r2_global:.3f} | {test_r2_global:.3f}")
            print(f"      Linear Trend:     R²={val_r2_trend:.3f} | {test_r2_trend:.3f}")  
            print(f"      Naive+Lag>=2yr:   R²={val_r2_naive:.3f} | {test_r2_naive:.3f}")
            print(f"      Cross-Country:    R²={val_r2_cross:.3f} | {test_r2_cross:.3f}")
            
            # Best baseline on validation
            best_val_baseline, best_val_r2 = _best_by_val_r2(fold_results)
            
            # Performance of the best baseline on the test set
            best_test_r2 = fold_results[best_val_baseline]['test_r2']
            generalization_gap = best_val_r2 - best_test_r2
            
            fold_results['best_baseline'] = {
                'model': best_val_baseline,
                'val_r2': best_val_r2,
                'test_r2': best_test_r2,
                'generalization_gap': generalization_gap
            }
            
            print(f"   Best baseline: {best_val_baseline} (Val: {best_val_r2:.3f} ->Test: {best_test_r2:.3f}, Gap: {generalization_gap:+.3f})")
            
            # More nuanced analysis of the generalization gap
            abs_gap = abs(generalization_gap)
            if abs_gap <= 0.05:
                print(f"      Excellent stability: very low gap (<=0.05)")
            elif abs_gap <= 0.1:
                print(f"      Good stability: gap within expectation (<=0.10)")
            elif abs_gap <= 0.15:
                print(f"      Moderate gap: acceptable temporal variation ({abs_gap:.3f})")
            else:
                print(f"      High gap: possible temporal instability ({abs_gap:.3f})")
            
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
        """
        Analyze the scientific predictability of the Data Warehouse baselines.

        Args:
            baseline_results: Baseline results with temporal validation

        Returns:
            Dict: Aggregated predictability analysis with generalization gaps
        """
        print("\nDuckDB predictability analysis")
        
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
                    # Test scores (main metric)
                    test_r2_scores.append(fold_data[baseline]['test_r2'])
                    # Validation scores (for comparison)
                    val_r2_scores.append(fold_data[baseline]['val_r2'])
                    # Generalization gap
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
        
        print("   Out-of-sample (TEST SET) performance of the baselines:")
        for baseline, stats in all_test_scores.items():
            val_stats = all_val_scores[baseline]
            gap_stats = generalization_gaps[baseline]
            print(f"      {baseline:20} | Test: R²={stats['mean_r2']:.3f}±{stats['std_r2']:.3f} | Val: R²={val_stats['mean_r2']:.3f} | Gap: {gap_stats['mean_gap']:+.3f}")
        
        # Find the best baseline based on the TEST set (not validation)
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
                'architecture': 'sql_engine',
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

            if best_mean_test_r2 < 0:
                predictability_analysis['predictability_level'] = 'very_low'
                print(f"   Very low predictability: R²_test < 0")
                print(f"      Interpretation: model worse than a constant baseline")
            elif best_mean_test_r2 < 0.05:
                predictability_analysis['predictability_level'] = 'very_low'
                print(f"   Very low predictability: R²_test = {best_mean_test_r2:.3f}")
                print(f"      Interpretation: almost no predictive power")
            elif best_mean_test_r2 < 0.15:
                predictability_analysis['predictability_level'] = 'low'
                print(f"   Low predictability: R²_test = {best_mean_test_r2:.3f}")
                print(f"      Interpretation: limited predictive power")
            elif best_mean_test_r2 < 0.35:
                predictability_analysis['predictability_level'] = 'moderate'
                print(f"   Moderate predictability: R²_test = {best_mean_test_r2:.3f}")
                print(f"      Interpretation: reasonable predictive power")
            else:
                predictability_analysis['predictability_level'] = 'good'
                print(f"   Good predictability: R²_test = {best_mean_test_r2:.3f}")
                print(f"      Interpretation: good predictive power")
            
            avg_generalization_gap = np.mean([gap_data['mean_gap'] for gap_data in generalization_gaps.values()])
            abs_avg_gap = abs(avg_generalization_gap)

            if abs_avg_gap <= 0.05:
                print(f"   Excellent stability: very low mean gap ({avg_generalization_gap:+.3f})")
                stability_level = "excellent"
            elif abs_avg_gap <= 0.1:
                print(f"   Good stability: mean gap within expectation ({avg_generalization_gap:+.3f})")
                stability_level = "good"
            elif abs_avg_gap <= 0.15:
                print(f"   Moderate stability: acceptable temporal variation ({avg_generalization_gap:+.3f})")
                stability_level = "moderate"
            else:
                print(f"   Instability detected: high mean gap ({avg_generalization_gap:+.3f})")
                print(f"      Possible overfitting or strong temporal variation")
                stability_level = "low"
            
            predictability_analysis['stability_analysis'] = {
                'avg_generalization_gap': float(avg_generalization_gap),
                'stability_level': stability_level
            }
            
        else:
            predictability_analysis = {
                'architecture': 'sql_engine',
                'baseline_scores': {},
                'predictability_level': 'unknown'
            }
        
        return predictability_analysis
    
    def save_results(self, target_analysis: Dict, baseline_results: Dict, 
                    predictability_analysis: Dict):
        """
        Save the results of the ML Data Warehouse Consumer analysis.

        Args:
            target_analysis: Results of the target distribution analysis
            baseline_results: Results of the baseline models
            predictability_analysis: Predictability analysis

        Returns:
            Dict: Complete results saved
        """
        print(f"\nSaving DuckDB results...")
        
        full_results = {
            'architecture': 'sql_engine_consumer',
            'pattern': 'ml_sql_engine_consumer',
            'target_variable': self.target_col,
            'data_source': self.db_path,
            'data_access_method': 'direct_view_queries',
            'target_distribution_analysis': target_analysis,
            'baseline_model_results': baseline_results,
            'predictability_analysis': predictability_analysis,
            'summary': {
                'total_folds_analyzed': len(baseline_results),
                'best_baseline_model': predictability_analysis.get('best_baseline', 'unknown'),
                'best_baseline_r2': predictability_analysis.get('best_test_r2', 0),
                'predictability_level': predictability_analysis.get('predictability_level', 'unknown'),
                'r2_score_identical_tolerance': 0.001
            }
        }
        
        results_file = f"{self.results_path}/baseline_analysis_sql_engine_consumer_results.json"
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"   Results saved: {results_file}")
        
        return full_results
    
    def run_complete_analysis(self):
        """
        Run the complete baseline analysis via the ML Data Warehouse Consumer.

        Returns:
            Dict: Complete results of the Data Warehouse vs Data Lake comparative analysis
        """
        print(f"Complete analysis - DuckDB architecture")
        
        try:
            target_analysis = self.analyze_target_distribution()
            
            # 2. Test the baseline models via Feature Store views
            baseline_results = self.test_baseline_models()
            
            # 3. Analyze predictability
            predictability_analysis = self.analyze_predictability(baseline_results)
            
            # 4. Save results
            results = self.save_results(target_analysis, baseline_results, 
                                       predictability_analysis)
            
            # 5. Final summary
            print(f"\nSummary - DuckDB architecture:")
            print(f"   Target: {self.target_col}")
            print(f"   Predictability: {predictability_analysis.get('predictability_level', 'unknown').upper()}")
            print(f"   Best baseline: {predictability_analysis.get('best_baseline', 'unknown')}")
            print(f"   R² Test: {predictability_analysis.get('best_test_r2', 0):.3f}")
            
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
            
            # Check whether the temporal views were used
            views_used = 0
            fallbacks_used = 0
            if hasattr(self, '_fold_data_cache'):
                for cache_key in self._fold_data_cache.keys():
                    fold_id = int(cache_key.split('_')[1])
                    split = cache_key.split('_')[2]
                    view_name = f"vw_fold_{fold_id}_{split}"
                    view_exists = self.conn_manager.execute_scalar(f"""
                        SELECT COUNT(*) > 0 
                        FROM information_schema.views 
                        WHERE table_name = '{view_name}'
                    """)
                    if view_exists:
                        views_used += 1
                    else:
                        fallbacks_used += 1
            
            if views_used > 0 and fallbacks_used == 0:
                print(f"   Temporal Views: {views_used} views used")
            elif views_used > 0 and fallbacks_used > 0:
                print(f"   [WARN] Temporal Views: partial ({views_used} views, {fallbacks_used} fallbacks)")
            else:
                print(f"   [WARN] Temporal Views: none (fallback only)")
                print(f"   Run setup.py first to create the temporal views")

            return results
            
        except Exception as e:
            # Re-raised for the same reason as in the hierarchical model: a
            # dictionary with status 'failed' passes through as a successful run.
            print(f"\n[ERROR] DuckDB analysis: {e}")
            raise
        finally:
            self.cleanup()

if __name__ == "__main__":
    print("=" * 60)
    analyzer = None
    try:
        analyzer = BaselineModelAnalysisSqlEngine()
        results = analyzer.run_complete_analysis()
        print(f"\nDuckDB baseline analysis completed!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        if analyzer:
            analyzer.cleanup()
        raise
    print("=" * 60)
