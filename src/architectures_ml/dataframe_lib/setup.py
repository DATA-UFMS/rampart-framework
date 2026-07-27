#!/usr/bin/env python3
"""Reproducible setup of the ML pipeline for the Polars DataFrame architecture.

The module runs the stages of the methodological protocol in the native Polars
paradigm: loading via pl.scan_parquet() (LazyFrame), creation of temporal folds with
anti-leak gaps, feature alignment with the Data Lake and Data Warehouse architectures,
and artifact generation in `outputs/ml_pipeline/architectures/dataframe_lib/`.

Keeps methodological symmetry with DL and DW for a controlled comparison,
differing only in the Polars-specific implementation using expressions and
lazy evaluation for memory optimization."""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from typing import Any, List, Dict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.base_architecture import BaseArchitectureML
from core.config import get_absolute_output_path
from core.validation import (DataIntegrityValidator, TemporalValidator,
                             assert_lag_columns)
from core.logging_config import get_logger, log_ml_pipeline


class DataFrameLibArchitectureML(BaseArchitectureML):
    """ML pipeline implementation for the Polars DataFrame architecture.

    The class keeps methodological symmetry with the Data Lake and Data Warehouse
    versions: it uses the same temporal folds (QP1), guarantees equivalence of features
    and validations (QP2) and records every artifact required by the benchmark (QP3).

    Processing uses Polars with lazy evaluation (LazyFrames) for memory optimization
    and idiomatic expressions for transformations, differing in the paradigm
    but keeping equivalence in the final results.
    """

    PARADIGM_META = {
        'name': 'dataframe_lib',
        'label': 'DataFrame Library (Polars)',
        'processor_module': 'collection.dataframe_lib.processor',
        'processor_class': 'DataFrameLibProcessor',
        'processor_run_method': 'run_dataframe_lib_processing',
        'baseline_module': 'architectures_ml.dataframe_lib.models.baseline_analysis',
        'baseline_class': 'BaselineModelAnalysisDataFrameLib',
        'hierarchical_module': 'architectures_ml.dataframe_lib.models.hierarchical_model',
        'hierarchical_class': 'HierarchicalModelDataFrameLib',
        'setup_script': 'src/architectures_ml/dataframe_lib/setup.py',
        'processor_script': 'src/collection/dataframe_lib/processor.py',
        'baseline_script': 'src/architectures_ml/dataframe_lib/models/baseline_analysis.py',
        'hierarchical_script': 'src/architectures_ml/dataframe_lib/models/hierarchical_model.py',
        # Declared here because the three paradigms write to distinct
        # layouts; without it an analysis module would need to know the
        # layout of every paradigm in order to find its results.
        'master_artifact': {'kind': 'parquet',
                            'path': 'ml_pipeline/architectures/dataframe_lib/prep/'
                                    'master_data_dataframe_lib.parquet'},
        'baseline_results_json': 'ml_pipeline/architectures/dataframe_lib/models/baseline_results/baseline_analysis_dataframe_lib_results.json',
    }

    def _safe_write_parquet_file(self, df: pl.DataFrame, file_path: str) -> None:
        """
        Write a Parquet file with defensive handling of conflicts.

        Args:
            df: Polars DataFrame to persist
            file_path: Destination path for the Parquet file

        Conflict handling:
            - Creation of parent directories if absent
            - Removal of pre-existing conflicting files/directories
            - Atomic write via polars.write_parquet
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
        df.write_parquet(file_path)

    def __init__(self):
        """Initialize paths, validators and logging for the Polars DataFrame pipeline."""
        # Base architecture initialization
        output_base = get_absolute_output_path('ml_pipeline/architectures/dataframe_lib')
        super().__init__(architecture_name='dataframe_lib', output_base_path=output_base)

        self.logger = get_logger(__name__, with_ml_context=True)
        self.logger.set_context(architecture='dataframe_lib', module='setup')

        print("Initializing Polars ML Pipeline")
        print("Lazy evaluation with Polars expressions")

        self.parquet_path = get_absolute_output_path('collection/dataframe_lib/processed/final_results.parquet')
        self.fallback_path = get_absolute_output_path('collection/dataframe_lib/raw')

        self.temporal_validator = TemporalValidator(min_gap_years=2)
        self.data_validator = DataIntegrityValidator()

        print(f"  Base directory: {self.output_base}")
        print(f"  Primary data: {self.parquet_path}")
        print(f"  Raw data (fallback): {self.fallback_path}")
        print("  Lazy evaluation with Polars expressions")

    def setup_environment(self) -> None:
        """
        Configure the Polars environment with optimizations for temporal ML.

        Settings applied:
            1. String cache: Enabled for memory optimization
            2. Streaming: Lazy mode for datasets >1GB
            3. Random seeds: Determinism in stochastic operations

        Rationale for the parameters:
            - String cache: Reduces string overhead on educational datasets
            - Lazy evaluation: Automatic optimization of operations
            - Seeds: Controls sampling and statistical transformations

        """
        print("Configuring Polars")

        pl.enable_string_cache()

        print("  String cache enabled")
        print("  Lazy evaluation enabled")

        # No seeding of the global RNG here. BaseArchitectureML.__init__ calls
        # setup_reproducibility, which already does it for all three -- this was a
        # repetition present in two paradigms and absent in the third, in a
        # comparison that assumes they differ only in how they move data.
        #
        # And it makes no difference to the result: nothing consumes numpy's global
        # RNG. Every estimator receives an explicit random_state and every draw uses
        # a local default_rng. That is why the shuffled order in which the benchmark
        # runs the paradigms changes nothing -- an invariant that now has a test,
        # instead of holding by accident.

    def load_data(self) -> pl.DataFrame:
        """
        Load educational data with lazy evaluation (LazyFrame) via Polars.

        Returns:
            pl.DataFrame: Polars DataFrame with the loaded data (after .collect())

        Raises:
            FileNotFoundError: When neither processed nor raw data are available

        Hierarchical loading strategy:
            1. Processed data: Post-Data-Lake-pipeline data (optimized Parquet)
            2. Raw partitioned: Fallback to partitioned raw data
            3. Error handling: Detailed logging for debugging

        Polars advantages:
            - Lazy evaluation via scan_parquet() for datasets >RAM
            - Native Apache Arrow engine for performance
            - Idiomatic expressions for efficient transformations

        load_data returns a collected DataFrame (required for compatibility
        with the base class). Lazy evaluation is used internally in transformations.
        """
        self.logger.info("Starting loading with Polars lazy evaluation")
        print("\nLoading data (lazy loading)")

        lf = None
        data_source = None

        # Strategy 1: Processed data (optimized)
        if os.path.exists(self.parquet_path):
            try:
                lf = pl.scan_parquet(self.parquet_path)
                data_source = "processed"
                print(f"  LazyFrame loaded")
            except (OSError, pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
                self.logger.warning(f"Error loading processed data: {e}")
                print(f"  [ERROR] Processed data: {e}")

        # Strategy 2: Partitioned raw data (fallback)
        if lf is None and os.path.exists(self.fallback_path):
            try:
                print("  Falling back to partitioned raw data...")
                # Use glob for partitioned parquet files
                import glob
                parquet_files = glob.glob(f"{self.fallback_path}/**/*.parquet", recursive=True)
                if parquet_files:
                    lf = pl.scan_parquet(f"{self.fallback_path}/*.parquet")
                    data_source = "raw_partitioned"
                    print("  Raw loading ok")
            except (OSError, pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
                self.logger.error(f"Error loading raw data: {e}")
                print(f"  [ERROR] Raw data: {e}")

        # Loading validation
        if lf is None:
            raise FileNotFoundError(
                "Polars DataFrame data not found in any source.\n"
                f"Check: {self.parquet_path} or {self.fallback_path}\n"
                "Run 'dataframe_lib/processor.py' to generate processed data."
            )

        # Adequacy analysis

        # Lazy operations, computed a single time
        stats_lf = lf.select([
            pl.col('year').min().alias('year_min'),
            pl.col('year').max().alias('year_max'),
            pl.col('country_code').n_unique().alias('n_countries'),
            pl.len().alias('total_rows')
        ])

        computed_stats = stats_lf.collect().row(0)
        year_min, year_max, n_countries, total_rows = computed_stats

        # Adequacy analysis for temporal ML
        years_span = year_max - year_min + 1
        avg_obs_per_country = total_rows / n_countries if n_countries > 0 else 0

        print(f"  {year_min}-{year_max} ({years_span} years)")
        print(f"  {n_countries} countries ({avg_obs_per_country:.1f} obs/country)")
        print(f"  {total_rows:,} total observations")
        print(f"  Source: {data_source}")

        if years_span < 10:
            print("  [WARN] Short time series may limit walk-forward validation")

        if n_countries < 15:
            print("  [WARN] Few countries may affect geographic generalization")

        self.logger.info(f"Data loaded successfully via {data_source}")

        # Return the collected DataFrame for compatibility with the base class
        return lf.collect()

    @log_ml_pipeline('validation')
    def validate_data(self, df: pl.DataFrame) -> None:
        """
        Run validation with strategic Polars sampling.

        Args:
            df: Polars DataFrame with the loaded educational data

        Validation methodology:
            1. Adaptive sampling: min(1000, total_rows) for efficiency
            2. DataIntegrityValidator: Centralized validator for consistency
            3. Schema validation: Check for mandatory columns
            4. Range validation: Detection of impossible values
            5. Smart fallback: Automatic search for alternative variables

        Criteria:
            - Target coverage >50%: Adequate statistical power for ML
            - Range [0,100]: Consistency with educational definitions
            - Schema compliance: Presence of temporal/geographic identifiers
        """
        print("Validating data")

        # Adaptive sampling for efficient validation
        total_rows = len(df)
        sample_size = min(1000, total_rows)

        print(f"  Sampling: {sample_size:,}/{total_rows:,} ({sample_size/total_rows:.1%})")

        # Sampling with a reproducible seed
        sample_df = df.sample(n=sample_size, seed=self.config['random_seed'])
        sample_pd = sample_df.to_pandas()

        # Centralized validation with DataIntegrityValidator
        is_valid, validation_report = self.data_validator.validate_dataframe(
            sample_pd,
            target_col=self.source_column,
            check_completeness=True
        )

        if not is_valid:
            warnings = validation_report.get('warnings', [])
            self.logger.warning(f"Integrity problems detected: {len(warnings)} warnings")
            for warning in warnings[:3]:
                print(f"  [WARN] {warning}")

        # The configured target must exist. Substituting a similarly named
        # column would silently move the experiment to a different target,
        # invalidating every downstream comparison.
        if self.source_column not in df.columns:
            raise ValueError(
                f"Target column '{self.source_column}' declared by "
                f"{type(self.dataset_config).__name__} is absent from the "
                f"processed data. Available columns: {sorted(df.columns)}"
            )

        # Quality analysis via Polars

        # Computations via Polars expressions
        stats_lf = df.lazy().select([
            pl.col(self.source_column).is_not_null().sum().alias('target_data'),
            pl.col(self.source_column).min().alias('target_min'),
            pl.col(self.source_column).max().alias('target_max'),
            pl.col(self.source_column).mean().alias('target_mean'),
            (pl.col(self.source_column) > 100).sum().alias('over_100_count'),
            (pl.col(self.source_column) < 0).sum().alias('under_0_count'),
            pl.len().alias('total_rows')
        ])

        computed = stats_lf.collect().row(0)
        target_data, target_min, target_max, target_mean, over_100_count, under_0_count, total_rows_check = computed

        target_coverage = (target_data / total_rows_check) * 100 if total_rows_check > 0 else 0

        print(f"  Coverage: {target_data:,}/{total_rows_check:,} valid ({target_coverage:.1f}%)")
        print(f"  Range: [{target_min:.1f}%, {target_max:.1f}%]")
        print(f"  Mean: {target_mean:.1f}%")

        if target_coverage < 50:
            print("  [WARN] Low target coverage (<50%) may compromise ML")

        if over_100_count > 0:
            print(f"  [WARN] {over_100_count} values >100% (invalid data)")

        if under_0_count > 0:
            print(f"  [WARN] {under_0_count} values <0% (invalid data)")

        # Mandatory schema validation
        required_cols = ['country_code', 'year']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Incomplete schema for temporal ML: missing columns {missing_cols}.\n"
                "Country-year identifiers are mandatory for walk-forward validation."
            )

        print("  Validation complete")

    def create_target_implementation(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Build the target variable via a Polars transformation with idiomatic expressions.

        Args:
            df: Polars DataFrame with educational data

        Returns:
            Polars DataFrame enriched with the target variable dropout_rate_dataframe_lib

        Transformation:
            Dropout Rate = 100 - Completion Rate

        Polars implementation:
            Uses .with_columns() with expressions for efficiency, creating
            temporal lags via a temporal join (year+k) by country_code.

        """
        print(f"Building target: {self.source_column} -> {self.target_column}")
        print("  Dropout Rate = 100 - Completion Rate")

        # Range validation [0, 100]
        df_with_target = df.with_columns([
            pl.when(
                (pl.col(self.source_column) >= 0) & (pl.col(self.source_column) <= 100)
            ).then(100 - pl.col(self.source_column)).otherwise(None).alias(self.target_column)
        ])

        # Temporal lag via join (value from exactly N years back),
        # not a positional shift that assumes data without yearly gaps.
        print("  Creating lag features (dropout_rate_lag_2, lag_3)")
        try:
            base_lag = df_with_target.select(['country_code', 'year', self.target_column])

            # Lag of 2 years: joining on year+2 brings the value from 2 years back
            lag2 = base_lag.with_columns(
                (pl.col('year') + 2).alias('year')
            ).rename({self.target_column: 'dropout_rate_lag_2'})
            df_with_target = df_with_target.join(
                lag2, on=['country_code', 'year'], how='left'
            )

            # Lag of 3 years: same for 3 years back
            lag3 = base_lag.with_columns(
                (pl.col('year') + 3).alias('year')
            ).rename({self.target_column: 'dropout_rate_lag_3'})
            df_with_target = df_with_target.join(
                lag3, on=['country_code', 'year'], how='left'
            )

            print("  dropout_rate_lag_2 and dropout_rate_lag_3 created (temporal join country/year-k)")
        except (pl.exceptions.ColumnNotFoundError, pl.exceptions.ComputeError,
                KeyError) as exc:
            raise ValueError(
                f"dataframe_lib: failed to create the target lags: {exc}"
            ) from exc

        print("  Target created via Polars expressions")

        assert_lag_columns(df_with_target.collect_schema().names(),
                           'dataframe_lib', self.TARGET_LAG_ORDERS,
                           target_stem=self.TARGET_STEM)
        return df_with_target

    def _compute_target_statistics(self, df: pl.DataFrame) -> Dict[str, float]:
        """
        Compute descriptive statistics of the target variable via Polars.

        Args:
            df: Polars DataFrame with the target variable created

        Returns:
            Dictionary with float64 statistics for analysis

        Statistics computed:
            - Moments: mean, standard deviation
            - Range: minimum, maximum for outlier detection
            - Completeness: valid vs missing count for quality analysis

        Polars optimization:
            Aggregations via lazy expressions, computed a single time.
        """
        # Computation via Polars expressions
        stats_lf = df.lazy().select([
            pl.col(self.target_column).mean().alias('mean'),
            pl.col(self.target_column).std().alias('std'),
            pl.col(self.target_column).min().alias('min'),
            pl.col(self.target_column).max().alias('max'),
            pl.col(self.target_column).is_null().sum().alias('missing_count'),
            pl.col(self.target_column).is_not_null().sum().alias('valid_count')
        ])

        computed = stats_lf.collect().row(0)
        mean_val, std_val, min_val, max_val, missing_count, valid_count = computed

        # Conversion to float64 for consistency
        return {
            'mean': self.reported_statistic(mean_val),
            'std': self.reported_statistic(std_val),
            'min': self.reported_statistic(min_val),
            'max': self.reported_statistic(max_val),
            'missing_count': int(missing_count) if missing_count is not None else 0,
            'valid_count': int(valid_count) if valid_count is not None else 0
        }

    def _validate_temporal_folds(self, df: pl.DataFrame, folds: List[Dict]) -> None:
        """Temporal validation with TemporalValidator via Polars."""
        print("Validating temporal folds")

        for fold in folds:
            # Validate temporal integrity using years
            train_years = (fold['train_start'], fold['train_end'])
            val_years = (fold['val_start'], fold['val_end'])
            test_years = (fold['test_start'], fold['test_end'])

            is_valid = self.validate_temporal_integrity_years(train_years, val_years, test_years)
            if not is_valid:
                self.logger.warning(f"Fold {fold['fold_id']}: Temporal integrity problem")

            # Validate gaps using the centralized validator
            is_valid, errors = self.temporal_validator.validate_fold_integrity(fold)
            if not is_valid:
                for error in errors:
                    self.logger.warning(f"Fold {fold['fold_id']}: {error}")

            # Filters for counting via Polars expressions
            train_filter = (
                (pl.col('year') >= fold['train_start']) &
                (pl.col('year') <= fold['train_end']) &
                ~((pl.col('year') >= fold['train_gap_start']) &
                  (pl.col('year') <= fold['train_gap_end']))
            )
            val_filter = (
                (pl.col('year') >= fold['val_start']) &
                (pl.col('year') <= fold['val_end'])
            )
            test_filter = (
                (pl.col('year') >= fold['test_start']) &
                (pl.col('year') <= fold['test_end']) &
                ~((pl.col('year') >= fold['val_gap_start']) &
                  (pl.col('year') <= fold['val_gap_end']))
            )

            # Count data per fold via Polars
            fold_stats = df.lazy().select([
                train_filter.sum().alias('train_count'),
                val_filter.sum().alias('val_count'),
                test_filter.sum().alias('test_count'),
                pl.when(train_filter).then(pl.col('country_code')).n_unique().alias('train_countries'),
                pl.when(val_filter).then(pl.col('country_code')).n_unique().alias('val_countries'),
                pl.when(test_filter).then(pl.col('country_code')).n_unique().alias('test_countries'),
            ]).collect()

            fold_row = fold_stats.row(0)
            fold['train_count'] = int(fold_row[0]) if fold_row[0] is not None else 0
            fold['val_count'] = int(fold_row[1]) if fold_row[1] is not None else 0
            fold['test_count'] = int(fold_row[2]) if fold_row[2] is not None else 0
            fold['train_countries'] = int(fold_row[3]) if fold_row[3] is not None else 0
            fold['val_countries'] = int(fold_row[4]) if fold_row[4] is not None else 0
            fold['test_countries'] = int(fold_row[5]) if fold_row[5] is not None else 0

            print(f"\n  Fold {fold['fold_id']}:")
            print(f"    Train: {fold['train_count']} obs, {fold['train_countries']} countries")
            print(f"    Val: {fold['val_count']} obs, {fold['val_countries']} countries")
            print(f"    Test: {fold['test_count']} obs, {fold['test_countries']} countries")

    def discover_numeric_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Identify numeric columns via the native Polars type schema.

        Args:
            df: Polars DataFrame with educational data

        Returns:
            List of numeric column names
        """
        return [
            col for col in df.columns
            if df[col].dtype in [
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            ]
        ]

    def compute_feature_correlations(self, df: pl.DataFrame,
                                     features: List[str]) -> Dict[str, float]:
        """
        Compute feature-target Pearson correlations using the complete data.

        Args:
            df: Polars DataFrame with the complete educational data
            features: List of candidate features for correlation analysis

        Returns:
            Dictionary {feature_name: absolute_correlation} for ranking

        Methodology:
            1. Materializes the complete training data (no sampling)
            2. Computes the Pearson correlation via pandas for each feature
            3. Returns the absolute value for ranking by relevance
        """
        print("Analyzing feature-target correlations")

        target_col = self.target_column
        correlations = {}

        sample_pd = df.select([target_col] + features).drop_nulls(subset=[target_col]).to_pandas()

        print(f"  Materialized data: {len(sample_pd):,} obs, {len(features)} features")

        successful_correlations = 0
        failed_features = []

        for feat in features:
            if feat not in sample_pd.columns:
                correlations[feat] = 0.0
                continue

            try:
                corr = sample_pd[feat].corr(sample_pd[target_col])

                if pd.isna(corr):
                    correlations[feat] = 0.0
                else:
                    correlations[feat] = abs(float(corr))
                    successful_correlations += 1

            except (ValueError, TypeError, pl.exceptions.ComputeError) as e:
                self.logger.warning(f"Correlation error {feat}: {e}")
                correlations[feat] = 0.0
                failed_features.append(feat)

        print(f"  {successful_correlations}/{len(features)} correlations computed")

        if failed_features:
            print(f"  [WARN] {len(failed_features)} features with errors: {failed_features[:3]}")

        valid_correlations = [r for r in correlations.values() if r > 0]
        if valid_correlations:
            avg_corr = sum(valid_correlations) / len(valid_correlations)
            max_corr = max(valid_correlations)
            print(f"  Mean correlation: {avg_corr:.3f}, maximum: {max_corr:.3f}")

        return correlations

    def apply_collinearity_filter(self, df: pl.DataFrame, features: List[str],
                                   threshold: float = 0.8) -> List[str]:
        """
        Remove multicollinearity via greedy pairwise-correlation filtering.

        For each candidate feature, computes the maximum absolute correlation with
        the features already selected and rejects it if max |r| >= threshold.

        Args:
            df: Polars DataFrame with candidate features
            features: List of features for multicollinearity analysis
            threshold: Pairwise correlation threshold (default 0.8)

        Returns:
            Filtered list of features with reduced multicollinearity

        Greedy algorithm:
            1. First feature always accepted (baseline)
            2. Subsequent features accepted if max |r| < threshold
            3. Order preserved for determinism

        Materialization:
            - Complete training data (no sampling)
            - Correlation matrix via pandas after conversion
        """
        if len(features) <= 1:
            print("  Fewer than 2 features - collinearity check unnecessary")
            return features

        print(f"Filtering collinearity: {len(features)} features")

        try:
            corr_data = df.select(features).to_pandas().dropna()

            valid_rows = len(corr_data)
            print(f"  {valid_rows:,} valid observations post-dropna")

            if valid_rows > 10:
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
                print(f"  Insufficient data ({valid_rows}<=10) - top-10 fallback")
                return features[:10]

        except (ValueError, TypeError, np.linalg.LinAlgError) as e:
            self.logger.error(f"Error in collinearity filtering: {e}")
            print(f"[ERROR] Collinearity filtering failed: {e}")
            print("  Fallback: returning top-10 features")
            return features[:10]

    def prepare_features(self, df: pl.DataFrame, selected_features: List[str]) -> pl.DataFrame:
        """
        Prepare the final ML features with idiomatic Polars transformations.

        Args:
            df: Polars DataFrame with features selected via collinearity filtering
            selected_features: Post-selection features to transform

        Returns:
            Polars DataFrame enriched with original + transformed features

        Scientific Feature Engineering:
            Applies a symmetric log transform: T(x) = sign(x) * ln(|x| + 1)
            to the top-5 features to normalize asymmetric distributions.

        Methodological justifications:
            1. Top-5 limit: Based on the curse of dimensionality (Bellman, 1961)
            2. Symmetric log: Handles zeros and negatives naturally
            3. Polars expressions: Efficient transformations via expressions

        Final structure:
            - Metadata: country_code, year, target (essential for temporal ML)
            - Original features: selected_features (post-filtering)
            - Transformed features: {feature}_log_transform (top-5)
        """
        print("\nFeature engineering")

        # Criterion: Limit scope due to the curse of dimensionality
        features_to_transform = selected_features[:5] if len(selected_features) > 5 else selected_features
        transformed_count = 0

        print(f"  Transforming {len(features_to_transform)} features (symmetric log):")

        # Application of the transformation via Polars expressions
        new_cols = []
        for feat in features_to_transform:
            if feat not in df.columns:
                print(f"    {feat}: ABSENT (ignored)")
                continue

            transform_col = f"{feat}_log_transform"

            print(f"    {feat} -> {transform_col}")

            # Symmetric log transform: sign(x) * ln(|x| + 1)
            new_cols.append(
                pl.when(pl.col(feat).is_null())
                .then(None)
                .otherwise(pl.col(feat).sign() * (pl.col(feat).abs() + 1).log())
                .alias(transform_col)
            )
            transformed_count += 1

        # Apply all transformations at once via with_columns
        if new_cols:
            df = df.with_columns(new_cols)

        print(f"  {transformed_count} log transforms applied")

        # Construction of the final ML dataset

        # Essential metadata for temporal ML
        ml_features = ['country_code', 'year', self.target_column]

        # Original features post collinearity filtering
        ml_features.extend(selected_features)

        # Transformed features
        transformed_cols = [f"{feat}_log_transform" for feat in features_to_transform
                          if f"{feat}_log_transform" in df.columns]
        ml_features.extend(transformed_cols)

        # Include the target lags in the saved dataset
        for lag_col in ['dropout_rate_lag_2', 'dropout_rate_lag_3']:
            if lag_col in df.columns and lag_col not in ml_features:
                ml_features.append(lag_col)

        # Remove duplicates preserving order
        ml_features = list(dict.fromkeys(ml_features))
        ml_features = [col for col in ml_features if col in df.columns]

        print(f"  Final ML dataset: {len(ml_features)} variables "
              f"({len(selected_features)} original, {len(transformed_cols)} transformed)")

        # Final selection
        result_df = df.select(ml_features)

        print("  Feature engineering complete")

        return result_df

    def save_folds(self, df: pl.DataFrame, folds: List[Dict]) -> None:
        """
        Save folds as Parquet files, keeping the Polars DataFrame paradigm.

        Args:
            df: Processed Polars DataFrame
            folds: List of fold configurations
        """
        print("\nSaving Polars folds")

        for fold in folds:
            fold_id = fold['fold_id']
            fold_dir = f"{self.prep_dir}/folds/fold_{fold_id}"
            os.makedirs(fold_dir, exist_ok=True)

            print(f"  Processing fold {fold_id}...")

            # Filters via Polars expressions
            train_filter = (
                (pl.col('year') >= fold['train_start']) &
                (pl.col('year') <= fold['train_end']) &
                ~((pl.col('year') >= fold['train_gap_start']) &
                  (pl.col('year') <= fold['train_gap_end']))
            )
            val_filter = (
                (pl.col('year') >= fold['val_start']) &
                (pl.col('year') <= fold['val_end'])
            )
            test_filter = (
                (pl.col('year') >= fold['test_start']) &
                (pl.col('year') <= fold['test_end']) &
                ~((pl.col('year') >= fold['val_gap_start']) &
                  (pl.col('year') <= fold['val_gap_end']))
            )

            # Filtering and saving
            try:
                train_df = df.filter(train_filter)
                val_df = df.filter(val_filter)
                test_df = df.filter(test_filter)

                self._safe_write_parquet_file(train_df, f'{fold_dir}/train_data_dataframe_lib.parquet')
                self._safe_write_parquet_file(val_df, f'{fold_dir}/val_data_dataframe_lib.parquet')
                self._safe_write_parquet_file(test_df, f'{fold_dir}/test_data_dataframe_lib.parquet')

                print(f"    Fold {fold_id}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

            except Exception as e:
                print(f"    [ERROR] Saving fold {fold_id}: {e}")
                raise

            fold_metadata = {
                **fold,
                'storage_method': 'parquet_files',
                'paradigm': 'dataframe_lib'
            }
            self.save_fold_metadata(fold_metadata, fold_dir)

        # Master data
        print("\n  Saving master data...")
        try:
            master_path = f"{self.prep_dir}/master_data_dataframe_lib.parquet"
            self._safe_write_parquet_file(df, master_path)
            print(f"    Master data: {len(df)} records")

        except Exception as e:
            print(f"    [ERROR] Saving master data: {e}")
            raise

        # Master configuration
        total_obs = len(df)
        total_countries = df.select(pl.col('country_code').n_unique()).item()
        year_min = int(df.select(pl.col('year').min()).item())
        year_max = int(df.select(pl.col('year').max()).item())

        self.save_master_config(folds, total_obs, total_countries, (year_min, year_max))

        print(f"  Polars: folds saved")


def main():
    """Run the Polars DataFrame pipeline end-to-end for local validation."""
    print("=" * 80)
    print("Polars ML Pipeline")
    print("=" * 80)

    setup = None
    try:
        setup = DataFrameLibArchitectureML()
        results = setup.run_setup()

        if results.get('status') == 'success':
            print("Pipeline ok")
            print(f"  Features selected: {results.get('features_selected', 'N/A')}")
            print(f"  Folds created: {results.get('folds_created', 'N/A')}")
            print(f"  Timestamp: {results.get('setup_timestamp', 'N/A')}")
        else:
            print(f"[ERROR] Pipeline failed: {results.get('error', 'Unknown error')}")
            return results

        return results

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'architecture': 'dataframe_lib',
            'status': 'failed',
            'error': str(e),
            'setup_timestamp': datetime.now().isoformat()
        }

    finally:
        # Symmetric across paradigms: the benchmark re-runs each phase
        # twelve times in the same process, and a resource that survives
        # one repetition is measured by the next one.
        if setup is not None:
            setup.release_resources()


if __name__ == '__main__':
    # sys.exit, not the builtin exit: the latter comes from the site module and
    # may not exist under python -O or in an embedded environment.
    results = main()
    sys.exit(0 if results.get('status') == 'success' else 1)
