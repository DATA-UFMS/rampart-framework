#!/usr/bin/env python3
"""Reproducible setup of the ML pipeline for the Data Warehouse architecture.

Runs the methodological protocol in SQL-first mode: opens the DuckDB database
produced in the collection phase, creates temporal folds with anti-leak gaps,
keeps the same feature engineering process used in the Data Lake architecture,
and exports the artifacts needed for comparison. The focus is on preserving
symmetry with the Data Lake so that observed differences reflect characteristics
of the schema-on-write paradigm."""

import os
import sys
from typing import List, Dict, Any

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.base_architecture import BaseArchitectureML
from core.config import get_absolute_output_path

from collection.sql_engine.connection_manager import (
    DuckDBConnectionManager, 
    SQLProcessingError
)


class SqlEngineArchitectureML(BaseArchitectureML):
    """ML pipeline implementation for the Data Warehouse architecture.

    Reproduces the same protocol applied to the Data Lake architecture,
    differing only by in-database SQL processing. Every artifact it produces
    (folds, target statistics, feature matrices) follows the framework's
    convention so that benchmarking and practical equivalence are possible."""

    PARADIGM_META = {
        'name': 'sql_engine',
        'label': 'SQL Engine (DuckDB)',
        'processor_module': 'collection.sql_engine.processor',
        'processor_class': 'SqlEngineProcessor',
        'processor_run_method': 'run_sql_engine_processing',
        'baseline_module': 'architectures_ml.sql_engine.models.baseline_analysis',
        'baseline_class': 'BaselineModelAnalysisSqlEngine',
        'hierarchical_module': 'architectures_ml.sql_engine.models.hierarchical_model',
        'hierarchical_class': 'HierarchicalModelSQLFirst',
        'setup_script': 'src/architectures_ml/sql_engine/setup.py',
        'processor_script': 'src/collection/sql_engine/processor.py',
        'baseline_script': 'src/architectures_ml/sql_engine/models/baseline_analysis.py',
        'hierarchical_script': 'src/architectures_ml/sql_engine/models/hierarchical_model.py',
        # Declared here because the three paradigms write to distinct
        # layouts; without it an analysis module has to know each
        # paradigm's layout in order to find its results.
        # The engine keeps the data inside the database: there is no master parquet.
        'master_artifact': {'kind': 'duckdb_table', 'table': 'analytics_wide',
                            'database': 'collection/sql_engine/{dataset}_data.duckdb'},
        'baseline_results_json': 'ml_pipeline/architectures/sql_engine/models/baseline_analysis_sql_engine_consumer_results.json',
    }

    def __init__(self):
        """Initializes paths, DuckDB connection and logger."""
        # Base architecture initialization
        output_base = get_absolute_output_path('ml_pipeline/architectures/sql_engine')
        super().__init__(architecture_name='sql_engine', output_base_path=output_base)
        
        print("Initializing DuckDB ML Pipeline")
        print("SQL-first with temporal validation")
        
        # Data Warehouse specific settings
        dataset_name = self.dataset_config.name
        self.db_path = get_absolute_output_path(f'collection/sql_engine/{dataset_name}_data.duckdb')
        self.conn_manager = None
        
        print(f"  Base directory: {self.output_base}")
        print(f"  DuckDB: {self.db_path}")
        print("  Zero file I/O, native SQL processing without cache")
    
    def release_resources(self) -> None:
        """Closes the DuckDB connection. See BaseArchitectureML.release_resources."""
        manager = getattr(self, 'conn_manager', None)
        if manager is not None:
            manager.close_connection()
            self.conn_manager = None

    def setup_environment(self) -> None:
        """Opens the DuckDB database produced in the collection phase and initializes the manager.

        Raises ``FileNotFoundError`` when the database is unavailable, signalling
        that the collection stage has to be re-run. `DuckDBConnectionManager`
        encapsulates simple retries (3 attempts, 1s pause) to avoid transient
        failures and guarantees that subsequent queries stay within the same
        transactional context."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"DuckDB database not found: {self.db_path}\n"
                f"Run 'sql_engine/processor.py' before this ML pipeline."
            )
        
        self.conn_manager = DuckDBConnectionManager(
            self.db_path,
            max_retries=3,      
            retry_delay=1.0
        )
        
        print("  Connection manager configured")
        print("  ACID compliance enabled")
    
    def load_data(self) -> None:
        """
        Runs data loading via native SQL.

        Returns:
            None: Data stays in-database, following the Data Warehouse paradigm

        Data stays in-database -- materialization on demand via SQL.
        """
        print("\nAnalyzing data via native SQL")
        
        stats_query = """
            SELECT
                COUNT(*) as total_records,
                (SELECT COUNT(*) FROM information_schema.columns
                 WHERE table_name = 'analytics_wide') as total_columns,
                MIN(year) as min_year,
                MAX(year) as max_year,
                COUNT(DISTINCT entity_id) as unique_countries,
                COUNT(DISTINCT year) as temporal_periods
            FROM analytics_wide
        """
        
        stats_result = self.conn_manager.execute_sql(stats_query).iloc[0]
        
        total_records = int(stats_result['total_records'])
        total_columns = int(stats_result['total_columns'])
        min_year = int(stats_result['min_year'])
        max_year = int(stats_result['max_year'])
        unique_countries = int(stats_result['unique_countries'])
        temporal_periods = int(stats_result['temporal_periods'])
        
        years_span = max_year - min_year + 1
        avg_obs_per_country = total_records / unique_countries if unique_countries > 0 else 0

        print(f"  {total_records:,} observations x {total_columns} variables")
        print(f"  {min_year}-{max_year} ({years_span} years, {temporal_periods} periods)")
        print(f"  {unique_countries} countries ({avg_obs_per_country:.1f} obs/country)")
        
        if years_span < 10:
            print(f"  [WARN] Short time series ({years_span} years) may limit walk-forward validation")

        if total_records < 100:
            print(f"  [WARN] Small dataset ({total_records} obs) may affect statistical power")
        
        # Data Warehouse paradigm: data never leaves the database
        return None
    
    def validate_data(self, data: Any) -> None:
        """
        Runs data integrity and adequacy validation.
        
        Args:
            data: Ignored - validation runs directly in the database
            
        Validations implemented:
            1. Schema validation: Check that essential columns exist
            2. Data coverage analysis: Completeness analysis for the target variable
            3. Temporal consistency: Check series continuity
            4. Geographic coverage: Representativeness analysis by country
        
        Adequacy criteria:
            - Minimum 50 valid observations for the target (statistical power)
            - entity_id, year required (unique identifiers)
            - Target coverage >20% to avoid extreme class imbalance
        
        Aborts when the configured target column is absent, rather than
        substituting a similarly named one.
        """
        print("Validating data integrity")
        
        target_source_col = self.source_column
        
        # 1. Schema validation
        print("  [1/4] Schema validation")
        column_exists = self.conn_manager.execute_scalar(f"""
            SELECT COUNT(*) > 0
            FROM information_schema.columns
            WHERE table_name = 'analytics_wide'
            AND column_name = '{target_source_col}'
        """)
        
        # The configured target must exist. Substituting a similarly named
        # column would silently move the experiment to a different target,
        # invalidating every downstream comparison.
        if not column_exists:
            available = self.conn_manager.execute_sql("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'analytics_wide'
                ORDER BY column_name
            """)
            raise ValueError(
                f"Target column '{target_source_col}' declared by "
                f"{type(self.dataset_config).__name__} is absent from "
                f"analytics_wide. Available columns: {available}"
            )
        
        # 2. Coverage analysis
        print("  [2/4] Coverage analysis")
        coverage_stats = self.conn_manager.execute_sql(f"""
            SELECT
                COUNT(*) as total_records,
                COUNT({self.source_column}) as valid_target,
                AVG({self.source_column}) as target_mean,
                STDDEV({self.source_column}) as target_std,
                MIN({self.source_column}) as target_min,
                MAX({self.source_column}) as target_max
            FROM analytics_wide
        """).iloc[0]
        
        total_records = int(coverage_stats['total_records'])
        valid_target = int(coverage_stats['valid_target'])
        target_coverage = (valid_target / total_records) * 100 if total_records > 0 else 0
        
        print(f"    Target coverage: {valid_target:,}/{total_records:,} ({target_coverage:.1f}%)")
        print(f"    Statistics: mean={coverage_stats['target_mean']:.1f}, std={coverage_stats['target_std']:.1f}")
        print(f"    Range: [{coverage_stats['target_min']:.1f}, {coverage_stats['target_max']:.1f}]")
        
        if valid_target < 50:
            print(f"    [WARN] Too few valid data points ({valid_target}<50)")
            print("      May compromise the statistical power of the ML models")

        if target_coverage < 20:
            print(f"    [WARN] Low coverage ({target_coverage:.1f}%<20%)")
            print("      Risk of selection bias in predictions")
        
        # 3. Temporal consistency
        print("  [3/4] Temporal consistency")
        temporal_stats = self.conn_manager.execute_sql("""
            SELECT
                COUNT(DISTINCT year) as unique_years,
                MIN(year) as min_year,
                MAX(year) as max_year,
                COUNT(DISTINCT entity_id) as unique_countries
            FROM analytics_wide
        """).iloc[0]
        
        years_span = int(temporal_stats['max_year']) - int(temporal_stats['min_year']) + 1
        actual_years = int(temporal_stats['unique_years'])
        temporal_completeness = (actual_years / years_span) * 100
        
        print(f"    Temporal span: {temporal_stats['min_year']}-{temporal_stats['max_year']} ({years_span} years)")
        print(f"    Temporal completeness: {actual_years}/{years_span} years ({temporal_completeness:.1f}%)")

        if temporal_completeness < 80:
            print("    [WARN] Significant temporal gaps may affect walk-forward validation")
        
        # 4. Geographic coverage
        print("  [4/4] Geographic representativeness")
        unique_countries = int(temporal_stats['unique_countries'])
        obs_per_country = total_records / unique_countries if unique_countries > 0 else 0
        
        print(f"    Unique countries: {unique_countries}")
        print(f"    Mean observations/country: {obs_per_country:.1f}")

        if unique_countries < 10:
            print("    [WARN] Few countries may limit geographic generalization")
        
        # 5. Required column validation
        required_cols = ['entity_id', 'year']
        missing_cols = []
        
        for col in required_cols:
            col_exists = self.conn_manager.execute_scalar(f"""
                SELECT COUNT(*) > 0
                FROM information_schema.columns
                WHERE table_name = 'analytics_wide'
                AND column_name = '{col}'
            """)
            
            if not col_exists:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(
                f"Required columns missing: {missing_cols}. "
                f"Schema incompatible with temporal analysis."
            )
        
        print("  Validation complete")

    def create_target_implementation(self, data: Any) -> None:
        """
        Builds the target variable via SQL transformation with educational grounding.
        
        Args:
            data: Ignored - transformation runs directly in the Data Warehouse
            
        Returns:
            None: Target added as a persistent column in the main table
            
        Transformation applied:
            Dropout Rate = 100 - Completion Rate
            
        Educational rationale:
            The dropout rate is metrically more interpretable than the
            completion rate for educational policy analysis:
            
            1. Problem orientation: High values indicate a need for intervention
            2. Linearity: Direct relation with adverse socioeconomic factors
            3. Comparability: International standard in the educational literature
               (UNESCO, 2018; World Bank Education Statistics)
        
        Robustness:
            - Preserves original NULLs (does not impute missing data)
            - Keeps the [0,100] range for interpretability
            - Uses a CASE statement for explicit edge-case handling
        
        """
        print(f"Building target: {self.source_column} -> {self.target_column}")
        print("  Dropout Rate = 100 - Completion Rate")
        self.conn_manager.execute_sql_no_return(f"""
            ALTER TABLE analytics_wide
            ADD COLUMN IF NOT EXISTS {self.target_column} DOUBLE
        """)
        
        transformation_query = f"""
            UPDATE analytics_wide
            SET {self.target_column} =
                CASE
                    WHEN {self.source_column} IS NULL THEN NULL
                    WHEN {self.source_column} < 0 THEN NULL     -- Invalid data
                    WHEN {self.source_column} > 100 THEN NULL   -- Invalid data
                    ELSE 100 - {self.source_column}
                END
        """
        
        self.conn_manager.execute_sql_no_return(transformation_query)
        
        # Post-transformation validation
        validation_stats = self.conn_manager.execute_sql(f"""
            SELECT
                COUNT(*) as total_records,
                COUNT({self.target_column}) as valid_targets,
                AVG({self.target_column}) as mean_dropout,
                STDDEV({self.target_column}) as std_dropout,
                MIN({self.target_column}) as min_dropout,
                MAX({self.target_column}) as max_dropout
            FROM analytics_wide
        """).iloc[0]
        
        valid_targets = int(validation_stats['valid_targets'])
        total_records = int(validation_stats['total_records'])
        success_rate = (valid_targets / total_records) * 100 if total_records > 0 else 0
        
        print(f"  Valid records: {valid_targets:,}/{total_records:,} ({success_rate:.1f}%)")
        print(f"  Mean dropout rate: {validation_stats['mean_dropout']:.1f}% +/- {validation_stats['std_dropout']:.1f}%")
        print(f"  Range: [{validation_stats['min_dropout']:.1f}%, {validation_stats['max_dropout']:.1f}%]")

        # Returns updated data from the database for use in the pipeline
        return self.conn_manager.execute_sql("SELECT * FROM analytics_wide")
    
    def _compute_target_statistics(self, data: Any) -> Dict[str, float]:
        """
        Computes complete descriptive statistics of the target variable via optimized SQL.
        
        Args:
            data: Ignored - analysis runs directly in the Data Warehouse
            
        Returns:
            Dict holding descriptive statistics: central moments,
            quartiles, skewness measures and adequacy for ML modelling
            
        Statistics computed:
            1. Moments: mean, standard deviation, skewness
            2. Range: minimum, maximum, amplitude
            3. Quartiles: Q1, median, Q3 for distribution analysis
            4. Completeness: valid vs missing counts for quality analysis
            
        Optimization:
            Single query with multiple aggregations to minimize roundtrips
            to the database (1 query vs 6+ individual queries).
        """
        comprehensive_stats_query = f"""
            WITH target_stats AS (
                SELECT
                    COUNT(*) as total_records,
                    COUNT({self.target_column}) as valid_count,
                    COUNT(*) - COUNT({self.target_column}) as missing_count,
                    AVG({self.target_column}) as mean_val,
                    STDDEV({self.target_column}) as std_val,
                    MIN({self.target_column}) as min_val,
                    MAX({self.target_column}) as max_val,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {self.target_column}) as q1,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {self.target_column}) as median,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {self.target_column}) as q3
                FROM analytics_wide
                WHERE {self.target_column} IS NOT NULL
            )
            SELECT *,
                   (max_val - min_val) as range_val,
                   (q3 - q1) as iqr_val
            FROM target_stats
        """
        
        stats_result = self.conn_manager.execute_sql(comprehensive_stats_query).iloc[0]
        
        # Result structuring
        statistics = {
            # Counts and completeness
            'total_records': int(stats_result.get('total_records', 0)),
            'valid_count': int(stats_result.get('valid_count', 0)),
            'missing_count': int(stats_result.get('missing_count', 0)),
            
            # Central moments
            'mean': self.reported_statistic(stats_result.get('mean_val')),
            'std': self.reported_statistic(stats_result.get('std_val')),
            'variance': self.reported_statistic(
                (stats_result.get('std_val') or float('nan')) ** 2),
            
            # Range and extremes
            'min': self.reported_statistic(stats_result.get('min_val')),
            'max': self.reported_statistic(stats_result.get('max_val')),
            'range': self.reported_statistic(stats_result.get('range_val')),
            
            # Quartiles and position measures
            'q1': self.reported_statistic(stats_result.get('q1')),
            'median': self.reported_statistic(stats_result.get('median')),
            'q3': self.reported_statistic(stats_result.get('q3')),
            'iqr': self.reported_statistic(stats_result.get('iqr_val')),
        }
        
        # Derived metrics for ML analysis
        if statistics['valid_count'] > 0:
            statistics['completeness_rate'] = statistics['valid_count'] / statistics['total_records']
            
            # Coefficient of variation (variability normalization)
            if statistics['mean'] != 0:
                statistics['coefficient_variation'] = statistics['std'] / abs(statistics['mean'])
            else:
                statistics['coefficient_variation'] = float('inf')
        else:
            statistics['completeness_rate'] = 0.0
            statistics['coefficient_variation'] = float('nan')
        
        return statistics
    
    def _validate_temporal_folds(self, data: Any, folds: List[Dict]) -> None:
        """
        Validates the temporal fold structure.
        
        Args:
            data: Ignored - analysis runs via SQL in the Data Warehouse
            folds: List of fold configurations for walk-forward validation
            
        Validations implemented:
            1. Statistical adequacy: Minimum 30 observations per fold (CLT)
            2. Geographic representativeness: Country coverage per fold
            3. Temporal consistency: Check of anti-leakage gaps
            4. Balance: Balanced distribution across train/validation/test
            
        Criteria:
            - Train: Minimum 30 obs (CLT rule of thumb)
            - Validation: Minimum 15 obs (basic statistical power)
            - Test: Minimum 10 obs (minimum out-of-sample evaluation)
            - Geographic coverage: >50% of countries in each fold for generalization
            
        Temporal cross-validation (Bergmeir & Benítez, 2012) requires
            a specific structure to avoid data leakage in time series.
        """
        print("Validating temporal folds")
        
        for i, fold in enumerate(folds):
            fold_id = fold['fold_id']
            print(f"\n  Fold {fold_id}:")
            
            # Query optimized for the fold's complete statistics
            fold_stats_query = f"""
                WITH fold_analysis AS (
                    SELECT
                        'train' as split_type,
                        COUNT(*) as obs_count,
                        COUNT(DISTINCT entity_id) as country_count,
                        COUNT({self.target_column}) as valid_targets,
                        AVG({self.target_column}) as target_mean
                    FROM analytics_wide
                    WHERE year >= {fold['train_start']} AND year <= {fold['train_end']}
                      AND NOT (year >= {fold.get('train_gap_start', fold['train_end'])}
                               AND year <= {fold.get('train_gap_end', fold['train_end'])})
                    
                    UNION ALL
                    
                    SELECT
                        'val' as split_type,
                        COUNT(*) as obs_count,
                        COUNT(DISTINCT entity_id) as country_count,
                        COUNT({self.target_column}) as valid_targets,
                        AVG({self.target_column}) as target_mean
                    FROM analytics_wide
                    WHERE year >= {fold['val_start']} AND year <= {fold['val_end']}
                    
                    UNION ALL
                    
                    SELECT
                        'test' as split_type,
                        COUNT(*) as obs_count,
                        COUNT(DISTINCT entity_id) as country_count,
                        COUNT({self.target_column}) as valid_targets,
                        AVG({self.target_column}) as target_mean
                    FROM analytics_wide
                    WHERE year >= {fold['test_start']} AND year <= {fold['test_end']}
                      AND NOT (year >= {fold.get('val_gap_start', fold['test_start'])}
                               AND year <= {fold.get('val_gap_end', fold['test_start'])})
                )
                SELECT * FROM fold_analysis ORDER BY
                    CASE split_type
                        WHEN 'train' THEN 1
                        WHEN 'val' THEN 2
                        WHEN 'test' THEN 3
                    END
            """
            
            fold_results = self.conn_manager.execute_sql(fold_stats_query)
            
            # Extraction and validation per split
            validation_warnings = []
            
            for _, row in fold_results.iterrows():
                split_type = row['split_type']
                obs_count = int(row['obs_count'])
                country_count = int(row['country_count'])
                valid_targets = int(row['valid_targets'] or 0)
                # Null means a split with no observed target, not a mean of
                # zero: the value goes into the fold artifact and would be
                # read as a measurement.
                target_mean = self.reported_statistic(row['target_mean'])
                
                # Store on the fold for later use
                fold[f'{split_type}_count'] = obs_count
                fold[f'{split_type}_countries'] = country_count
                fold[f'{split_type}_valid_targets'] = valid_targets
                fold[f'{split_type}_target_mean'] = target_mean
                
                min_obs_required = {'train': 30, 'val': 15, 'test': 10}[split_type]

                print(f"    {split_type.upper()}: {obs_count:,} obs, {country_count} countries, "
                      f"{valid_targets} valid targets (mean={target_mean:.1f}%)")
                
                if obs_count < min_obs_required:
                    warning = f"{split_type}: Few data points ({obs_count}<{min_obs_required}) - limited statistical power"
                    validation_warnings.append(warning)
                
                # Criterion 2: Geographic representativeness
                total_countries = self.conn_manager.execute_scalar(
                    "SELECT COUNT(DISTINCT entity_id) FROM analytics_wide"
                )
                geographic_coverage = (country_count / total_countries) * 100 if total_countries > 0 else 0
                
                if geographic_coverage < 50:
                    warning = f"{split_type}: Low geographic coverage ({geographic_coverage:.1f}%<50%)"
                    validation_warnings.append(warning)
                
                # Criterion 3: Target completeness
                target_completeness = (valid_targets / obs_count) * 100 if obs_count > 0 else 0
                if target_completeness < 70:
                    warning = f"{split_type}: Low target completeness ({target_completeness:.1f}%<70%)"
                    validation_warnings.append(warning)
            
            # Temporal consistency validation
            train_end = fold['train_end']
            val_start = fold['val_start']
            val_end = fold['val_end']
            test_start = fold['test_start']
            
            train_val_gap = val_start - train_end - 1
            val_test_gap = test_start - val_end - 1
            
            print(f"    Gaps: Train->Val: {train_val_gap} years, Val->Test: {val_test_gap} years")
            
            MIN_GAP = 2
            if train_val_gap < MIN_GAP:
                validation_warnings.append(
                    f"Insufficient train-validation gap ({train_val_gap}<{MIN_GAP} years skipped)"
                )
            
            if val_test_gap < MIN_GAP:
                validation_warnings.append(
                    f"Insufficient validation-test gap ({val_test_gap}<{MIN_GAP} years skipped)"
                )
            
            if validation_warnings:
                for warning in validation_warnings:
                    print(f"    [WARN] {warning}")
            else:
                print("    Fold ok")
    
    def save_folds(self, data: Any, folds: List[Dict]) -> None:
        """
        Creates temporal views instead of saving files.
        
        Args:
            data: Ignored (uses SQL directly)
            folds: List of folds
        """
        print("\nCreating DuckDB temporal views")
        
        for fold in folds:
            fold_id = fold['fold_id']
            fold_dir = f"{self.prep_dir}/folds/fold_{fold_id}"
            os.makedirs(fold_dir, exist_ok=True)
            
            print(f"  Creating temporal views for fold {fold_id}...")
            
            try:
                train_view_query = f"""
                    CREATE OR REPLACE VIEW vw_fold_{fold_id}_train AS
                    SELECT * FROM vw_selected_features 
                    WHERE year >= {fold['train_start']} AND year <= {fold['train_end']}
                      AND NOT (year >= {fold['train_gap_start']} AND year <= {fold['train_gap_end']})
                    ORDER BY entity_id, year
                """
                self.conn_manager.execute_sql_no_return(train_view_query)
                
                val_view_query = f"""
                    CREATE OR REPLACE VIEW vw_fold_{fold_id}_val AS
                    SELECT * FROM vw_selected_features 
                    WHERE year >= {fold['val_start']} AND year <= {fold['val_end']}
                    ORDER BY entity_id, year
                """
                self.conn_manager.execute_sql_no_return(val_view_query)
                
                test_view_query = f"""
                    CREATE OR REPLACE VIEW vw_fold_{fold_id}_test AS
                    SELECT * FROM vw_selected_features 
                    WHERE year >= {fold['test_start']} AND year <= {fold['test_end']}
                      AND NOT (year >= {fold['val_gap_start']} AND year <= {fold['val_gap_end']})
                    ORDER BY entity_id, year
                """
                self.conn_manager.execute_sql_no_return(test_view_query)
                
                print(f"    Views created: vw_fold_{fold_id}_{{train,val,test}}")
                
                train_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM vw_fold_{fold_id}_train")
                val_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM vw_fold_{fold_id}_val")
                test_count = self.conn_manager.execute_scalar(f"SELECT COUNT(*) FROM vw_fold_{fold_id}_test")
                
                print(f"    Check: Train={train_count}, Val={val_count}, Test={test_count}")
                
            except SQLProcessingError as e:
                print(f"    [ERROR] View creation failed for fold {fold_id}: {e}")
                raise
            
            fold_metadata = {
                **fold,
                'storage_method': 'duckdb_temporal_views',
                'view_names': {
                    'train': f'vw_fold_{fold_id}_train',
                    'val': f'vw_fold_{fold_id}_val',
                    'test': f'vw_fold_{fold_id}_test'
                }
            }
            self.save_fold_metadata(fold_metadata, fold_dir)
        
        try:
            master_view_query = """
                CREATE OR REPLACE VIEW vw_master_data AS
                SELECT * FROM analytics_wide 
                ORDER BY entity_id, year
            """
            self.conn_manager.execute_sql_no_return(master_view_query)
            print(f"  Master view created: vw_master_data")
            
        except SQLProcessingError as e:
            print(f"  [ERROR] Master view creation failed: {e}")
            raise
        
        total_obs = self.conn_manager.execute_scalar("SELECT COUNT(*) FROM analytics_wide")
        total_countries = self.conn_manager.execute_scalar("SELECT COUNT(DISTINCT entity_id) FROM analytics_wide")
        min_year = self.conn_manager.execute_scalar("SELECT MIN(year) FROM analytics_wide")
        max_year = self.conn_manager.execute_scalar("SELECT MAX(year) FROM analytics_wide")
        
        self.save_master_config(folds, total_obs, total_countries, (int(min_year), int(max_year)))
        
        print(f"  DuckDB: Temporal views created, zero file I/O")
    
    def discover_numeric_columns(self, data: Any) -> List[str]:
        """
        Identifies numeric columns by querying the engine's catalog.

        Args:
            data: Ignored - analysis runs via SQL metadata queries

        Returns:
            List of numeric column names

        Queries information_schema instead of scanning the data, which is the
        native way for a SQL engine to answer this question.

        Limitations:
            - Does not detect numeric categorical variables (e.g., country codes)
            - Ignores derived features not persisted in the schema
            - Assumes every numeric type is appropriate for ML
        """
        numeric_columns_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'analytics_wide'
            AND data_type IN ('DOUBLE', 'INTEGER', 'FLOAT', 'DECIMAL', 'NUMERIC')
            ORDER BY column_name
        """
        result = self.conn_manager.execute_sql(numeric_columns_query)
        return result['column_name'].tolist()
    
    def compute_feature_correlations(self, data: Any,
                                    features: List[str]) -> Dict[str, float]:
        """
        Computes Pearson feature-target correlations over training data.

        Args:
            data: pandas DataFrame filtered to the training period
            features: List of candidate features for correlation analysis

        Returns:
            Dictionary {feature_name: absolute_correlation} for ranking
        """
        print(f"Analyzing correlation of {len(features)} features with the target")

        target_col = self.target_column
        correlations = {}
        failed_features = []

        df = data[features + [target_col]].dropna(subset=[target_col])
        print(f"  {len(df):,} valid observations")

        for feat in features:
            try:
                if feat not in df.columns:
                    correlations[feat] = 0.0
                    continue
                corr = df[feat].corr(df[target_col])
                if pd.isna(corr):
                    correlations[feat] = 0.0
                else:
                    correlations[feat] = abs(float(corr))
            except Exception as e:
                print(f"  [ERROR] Correlation for {feat}: {e}")
                correlations[feat] = 0.0
                failed_features.append(feat)

        valid_correlations = [r for r in correlations.values() if r > 0]

        if valid_correlations:
            avg_correlation = sum(valid_correlations) / len(valid_correlations)
            max_correlation = max(valid_correlations)
            print(f"  Mean correlation: {avg_correlation:.3f}, maximum: {max_correlation:.3f}")

        if failed_features:
            print(f"  Features that failed: {len(failed_features)}")

        return correlations
    
    def apply_collinearity_filter(self, data: Any, features: List[str],
                                   threshold: float = 0.8) -> List[str]:
        """
        Removes multicollinearity via greedy pairwise correlation filtering.

        Args:
            data: pandas DataFrame filtered to the training period
            features: List of candidate features for analysis
            threshold: Pairwise correlation threshold (default 0.8)

        Returns:
            Filtered feature list with reduced multicollinearity
        """
        if len(features) < 2:
            print("  Fewer than 2 features - collinearity filtering unnecessary")
            return features

        print(f"Filtering collinearity: {len(features)} features, threshold={threshold}")

        try:
            corr_data = data[features].dropna()

            print(f"  {len(corr_data):,} valid observations after dropna")

            if len(corr_data) > 10:
                corr_matrix = corr_data.corr().abs()

                selected = []
                rejected_count = 0
                features = sorted(features)

                for feature in features:
                    if feature not in corr_matrix.columns:
                        continue

                    if not selected:
                        selected.append(feature)
                    else:
                        max_corr = 0.0
                        worst_pair = None

                        for sel_feat in selected:
                            if sel_feat in corr_matrix.columns:
                                corr_val = corr_matrix.loc[feature, sel_feat]
                                if corr_val > max_corr:
                                    max_corr = corr_val
                                    worst_pair = sel_feat

                        if max_corr < threshold:
                            selected.append(feature)
                        else:
                            rejected_count += 1
                            if rejected_count <= 3:
                                print(f"    Rejected {feature}: r={max_corr:.3f} with {worst_pair}")

                reduction_rate = ((len(features) - len(selected)) / len(features)) * 100
                print(f"  Original: {len(features)}, selected: {len(selected)}, "
                      f"removed: {len(features) - len(selected)} ({reduction_rate:.1f}%)")

                return selected

            else:
                raise ValueError(
                    f"Collinearity filtering needs more than 10 complete rows; "
                    f"got {len(corr_data)}. Returning an arbitrary subset would "
                    f"give this paradigm a different feature set from the "
                    f"others, and the comparison assumes they share one."
                )

        except Exception as e:
            print(f"[ERROR] Collinearity filtering failed: {e}")
            raise
    
    def prepare_features(self, data: Any, selected_features: List[str]) -> None:
        """
        Builds the final view with selected features and transformations.
        
        Args:
            data: Ignored - processing runs via SQL in the Data Warehouse
            selected_features: Features after collinearity filtering, for transformation
            
        Returns:
            None: View vw_selected_features created persistently in the database
            
        Scientific feature engineering:
            Applies a symmetric log transform to the top-5 most correlated
            features to normalize skewed distributions:

            T(x) = sign(x) * ln(|x| + 1)

            Preserves zeros and works with negative values, suitable for
            educational data that may include deficits/declines.

            Top-5 limit based on the curse of dimensionality (Bellman, 1961).

        Schema of the resulting view:
            - Metadata: entity_id, year, {target_column}
            - Original features: selected_features (after collinearity filtering)
            - Transformed features: {feature}_log_transform for the top-5
        """
        print(f"Preparing final view with {len(selected_features)} features")
        
        # Criterion: Limit to the top-5 most promising features
        features_to_transform = selected_features[:5]
        transformed_features_sql = []
        
        print(f"  Transforming {len(features_to_transform)} features (symmetric log):")
        
        for feat in features_to_transform:
            # Transformation, symmetric log: sign(x) * ln(|x| + 1)
            # The a. prefix is required because the query uses a self-join with an alias
            transformation_sql = f"""
                CASE
                    WHEN a.{feat} IS NULL THEN NULL
                    WHEN a.{feat} = 0 THEN 0.0
                    ELSE SIGN(a.{feat}) * LN(ABS(a.{feat}) + 1)
                END AS {feat}_log_transform
            """
            transformed_features_sql.append(transformation_sql)
            print(f"    {feat} -> {feat}_log_transform")

        # View query construction
        all_features_sql = selected_features.copy()

        if transformed_features_sql:
            print(f"  {len(transformed_features_sql)} log transforms applied")
        
        # SQL query for the structured final view
        # Temporal lag via self-join (the value from exactly N years ago),
        # not positional LAG(), which assumes data without annual gaps.
        feature_view_query = f"""
            CREATE OR REPLACE VIEW vw_selected_features AS
            SELECT
                -- Temporal and geographic metadata (essential for temporal ML)
                a.entity_id,
                a.year,
                a.{self.target_column},
                -- Target lags (2 and 3 years) via temporal join without leakage
                lag2.{self.target_column} AS dropout_rate_lag_2,
                lag3.{self.target_column} AS dropout_rate_lag_3,

                -- Original features after collinearity filtering
                {', '.join(['a.' + f for f in all_features_sql])}

                {', -- Transformed features (symmetric log)' if transformed_features_sql else ''}
                {', '.join(transformed_features_sql) if transformed_features_sql else ''}

            FROM analytics_wide a
            LEFT JOIN analytics_wide lag2
                ON a.entity_id = lag2.entity_id AND a.year = lag2.year + 2
            LEFT JOIN analytics_wide lag3
                ON a.entity_id = lag3.entity_id AND a.year = lag3.year + 3
            WHERE a.{self.target_column} IS NOT NULL  -- Essential filter for supervised ML
            ORDER BY a.entity_id, a.year           -- Preserves temporal order for walk-forward
        """
        
        try:
            self.conn_manager.execute_sql_no_return(feature_view_query)
            
            # View validation and report
            view_validation_query = f"""
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT entity_id) as unique_countries,
                    MIN(year) as min_year,
                    MAX(year) as max_year,
                    AVG({self.target_column}) as avg_target
                FROM vw_selected_features
            """
            
            validation_result = self.conn_manager.execute_sql(view_validation_query).iloc[0]
            
            total_records = int(validation_result['total_records'])
            unique_countries = int(validation_result['unique_countries'])
            min_year = int(validation_result['min_year'])
            max_year = int(validation_result['max_year'])
            avg_target = float(validation_result['avg_target'])
            
            # Final dimensionality computation
            original_features = len(selected_features)
            transformed_features = len(transformed_features_sql)
            total_features = original_features + transformed_features + 3  # +3 metadata
            
            print(f"  View vw_selected_features created:")
            print(f"    {total_records:,} obs, {total_features} variables ({original_features} original, "
                  f"{transformed_features} transformed)")
            print(f"    {unique_countries} countries, {min_year}-{max_year}, mean target: {avg_target:.1f}%")
            
            # ML adequacy analysis
            observations_per_feature = total_records / total_features if total_features > 0 else 0
            
            if observations_per_feature < 10:
                print("    [WARN] Few data points/feature - risk of overfitting")
            elif observations_per_feature > 50:
                print("    Good observations/features ratio for ML")
            
        except SQLProcessingError as e:
            print(f"  [ERROR] Feature view creation failed: {e}")
            raise RuntimeError(f"Unable to create feature view: {e}")
        
        print("  Features ready for ML modelling")
        # Data Warehouse paradigm: data stays in the database for efficiency
        return None


def main():
    """
    Main function for running and testing the Data Warehouse ML pipeline.
    
    Runs the full temporal ML pipeline following the methodology:
    1. Environment setup and validation
    2. Data loading and validation
    3. Target variable construction
    4. Feature selection with pairwise collinearity filtering
    5. Walk-forward temporal fold creation
    6. Final preparation for modelling
    
    Suitable for:
        - Pipeline development and debugging
        - Methodology validation
        - Architectural performance benchmarking
        - Integration tests before production
        
    Not suitable for:
        - Production execution (use the dedicated API)
        - Exploratory analysis (use notebooks)
        - Architectural comparison (use architectural_benchmark.py)
    """
    print("=" * 80)
    print("DuckDB ML Pipeline")
    print("=" * 80)

    setup = None
    try:
        setup = SqlEngineArchitectureML()
        results = setup.run_setup()
        
        success_flag = results.get('success', None)
        status_flag = results.get('status', None)
        is_success = (success_flag is True) or (isinstance(status_flag, str) and status_flag.lower() == 'success')
        if is_success:
            print("Pipeline ok")
            print(f"  Architecture: {results.get('architecture', 'N/A')}")
            print(f"  Selected features: {results.get('features_selected', results.get('selected_features_count', 'N/A'))}")
            print(f"  Temporal folds: {results.get('folds_created', results.get('total_folds', 'N/A'))}")
            if isinstance(results.get('total_observations', None), (int, float)):
                print(f"  Observations processed: {int(results.get('total_observations')):,}")

            if 'processing_time' in results and isinstance(results.get('processing_time'), (int, float)):
                print(f"  Processing time: {results['processing_time']:.2f}s")
        else:
            print("[ERROR] Pipeline failed")
            if 'error' in results:
                print(f"  Error: {results['error']}")

        print(f"\nResults:")
        for key, value in results.items():
            if key not in ['status', 'error']:
                print(f"  {key}: {value}")

        return results

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        print("  Check that DuckDB was processed correctly")
        print("  Run sql_engine/processor.py before this script")
        return {'status': 'failed', 'error': str(e)}

    finally:
        # On the failure path too: a run that died halfway through left the
        # connection open all the same.
        if setup is not None:
            setup.release_resources()
    


if __name__ == "__main__":
    results = main()
    # A failed setup must not report success to the pipeline, which runs each
    # stage as a subprocess and reads its exit status.
    sys.exit(0 if isinstance(results, dict)
             and results.get('status') == 'success' else 1)
