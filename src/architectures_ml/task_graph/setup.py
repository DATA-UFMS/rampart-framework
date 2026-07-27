#!/usr/bin/env python3
"""Reproducible setup of the ML pipeline for the Data Lake architecture.

The module runs the stages of the methodological protocol (QP1–QP3) in the
schema-on-read paradigm: loading via Dask, creation of temporal folds with
anti-leak gaps, feature alignment with the Data Warehouse architecture, and
artifact generation in `outputs/ml_pipeline/`. We keep the minimum necessary set
of transformations to guarantee symmetry with the Data Warehouse and to allow
distributed exploratory analysis without hidden caches or optimizations."""

import os
import sys
import glob
import shutil
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
from typing import Any, List, Dict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.base_architecture import BaseArchitectureML
from core.config import get_absolute_output_path
from core.scientific_config import SCIENTIFIC_CONFIG
from core.validation import (DataIntegrityValidator, TemporalValidator,
                             assert_lag_columns)
from core.logging_config import get_logger, log_ml_pipeline


class TaskGraphArchitectureML(BaseArchitectureML):
    """ML pipeline implementation for the Data Lake architecture.

    The class keeps methodological symmetry with the Data Warehouse version: it
    uses the same temporal folds (QP1), guarantees equivalence of features and
    validations (QP2) and records every artifact required by the benchmark (QP3).
    Processing is done with Dask in lazy mode, without additional cache layers,
    so that the intrinsic characteristics of the schema-on-read paradigm show
    through."""

    PARADIGM_META = {
        'name': 'task_graph',
        'label': 'Task-Graph Scheduler (Dask)',
        'processor_module': 'collection.task_graph.processor',
        'processor_class': 'TaskGraphProcessor',
        'processor_run_method': 'run_task_graph_processing',
        'baseline_module': 'architectures_ml.task_graph.models.baseline_analysis',
        'baseline_class': 'BaselineModelAnalysisTaskGraph',
        'hierarchical_module': 'architectures_ml.task_graph.models.hierarchical_model',
        'hierarchical_class': 'HierarchicalModelTaskGraph',
        'setup_script': 'src/architectures_ml/task_graph/setup.py',
        'processor_script': 'src/collection/task_graph/processor.py',
        'baseline_script': 'src/architectures_ml/task_graph/models/baseline_analysis.py',
        'hierarchical_script': 'src/architectures_ml/task_graph/models/hierarchical_model.py',
        # Declared here because the three paradigms write to distinct
        # layouts; without it an analysis module would need to know the
        # layout of every paradigm in order to find its results.
        'master_artifact': {'kind': 'parquet',
                            'path': 'ml_pipeline/architectures/task_graph/prep/'
                                    'master_data_task_graph.parquet'},
        'baseline_results_json': 'ml_pipeline/architectures/task_graph/models/baseline_results/baseline_analysis_task_graph_results.json',
    }

    def _safe_write_parquet_file(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Write a Parquet file with defensive cleanup of conflicts.

        Args:
            df: pandas DataFrame to persist
            file_path: Destination path for the Parquet file

        Raises:
            Exception: Propagated from the underlying write operation

        Conflict handling:
            - Creation of parent directories if absent
            - Removal of pre-existing conflicting files/directories
            - Atomic write via pandas.to_parquet with index=False

        Data Lakes frequently have naming conflicts between files and
            directories because of the schema-on-read nature. This function
            guarantees a successful write regardless of the filesystem state.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
        df.to_parquet(file_path, index=False)
    
    def __init__(self):
        """Initialize paths, validators and logging for the Data Lake pipeline."""
        # Base architecture initialization
        output_base = get_absolute_output_path('ml_pipeline/architectures/task_graph')
        super().__init__(architecture_name='task_graph', output_base_path=output_base)
        
        self.logger = get_logger(__name__, with_ml_context=True)
        self.logger.set_context(architecture='task_graph', module='setup')
        
        print("Initializing Dask ML Pipeline")
        print("Schema-on-read with lazy distributed processing")

        # Data Lake path settings
        self.task_graph_path = get_absolute_output_path('collection/task_graph/processed/final_results.parquet')
        self.fallback_path = get_absolute_output_path('collection/task_graph/raw')
        
        self.temporal_validator = TemporalValidator(min_gap_years=2)
        self.data_validator = DataIntegrityValidator()
        
        print(f"  Base directory: {self.output_base}")
        print(f"  Primary data: {self.task_graph_path}")
        print(f"  Raw data (fallback): {self.fallback_path}")
        print("  Lazy evaluation without additional cache layers")
    
    def setup_environment(self) -> None:
        """
        Configure the Dask environment with optimizations for temporal ML.

        Settings applied:
            1. Query planning: Enabled for automatic optimization of operations
            2. Memory management: Conservative thresholds for stability
            3. Random seeds: Determinism in stochastic operations
            4. Worker limits: Prevention of OOM on large datasets

        Rationale for the parameters:
            - memory.target=0.8: 80% RAM before spill (Dask best practices)
            - memory.spill=0.9: 90% RAM before killing the worker (failsafe)
            - query-planning=True: Graph optimization for datasets >1GB

        Seeds configured:
            - NumPy: Controls sampling and statistical transformations
            - Dask: Guarantees determinism in distributed array operations

        """
        print("Configuring Dask")

        dask.config.set({'dataframe.query-planning': True})      # Query optimization
        # Explicit core budget, the same as the other paradigms. Without it
        # the scheduler sizes the pool from the machine, and the comparison
        # starts to depend on where it was run.
        dask.config.set(
            {'num_workers': int(SCIENTIFIC_CONFIG['engine_threads'])})
        dask.config.set({'distributed.worker.memory.target': 0.8})  # 80% RAM target
        dask.config.set({'distributed.worker.memory.spill': 0.9})   # 90% RAM spill
        
        print("  Memory management: conservative thresholds")
        print("  Query optimization: enabled for datasets >1GB")

        # Dask has no native global seed config.
        # Reproducibility is guaranteed by the numpy seed.
        # No seeding of the global RNG here. BaseArchitectureML.__init__ calls
        # setup_reproducibility, which already does it for all three -- this was
        # a repetition present in two paradigms and absent in the third, in a
        # comparison that assumes they differ only in how they move data.
        #
        # And it makes no difference to the result: nothing consumes numpy's
        # global RNG. Every estimator receives an explicit random_state and
        # every draw uses a local default_rng. That is why the shuffled order in
        # which the benchmark runs the paradigms changes nothing -- an invariant
        # that now has a test, instead of holding by accident.
    
    def load_data(self) -> dd.DataFrame:
        """
        Load educational data with the distributed Schema-on-Read paradigm.

        Returns:
            dd.DataFrame: Dask DataFrame with lazy evaluation preserved for
                         memory optimization on datasets >10GB

        Raises:
            FileNotFoundError: When neither processed nor raw data are available

        Hierarchical loading strategy:
            1. Processed data: Post-Data-Lake-pipeline data (optimized format)
            2. Raw partitioned: Fallback to partitioned raw data
            3. Error handling: Detailed logging for debugging

        Schema-on-Read advantages:
            - Flexibility: Schema inferred dynamically during loading
            - Performance: PyArrow engine optimized for columnar formats
            - Scalability: Dask partitions allow processing >available RAM

        Lazy reading -- materialization on demand.
        """
        self.logger.info("Starting Dask loading with schema-on-read")
        print("\nLoading data (schema-on-read)")

        ddf = None
        data_source = None

        # Strategy 1: Processed data (optimized)
        if os.path.exists(self.task_graph_path):
            try:
                ddf = dd.read_parquet(self.task_graph_path, engine='pyarrow').persist()
                data_source = "processed"
                ncols = len(ddf.columns)
                print(f"  Loaded: {ddf.npartitions} partitions x {ncols} variables")
            except Exception as e:
                self.logger.warning(f"Error loading processed data: {e}")
                print(f"  [ERROR] Processed data: {e}")

        # Strategy 2: Partitioned raw data (fallback)
        if ddf is None and os.path.exists(self.fallback_path):
            try:
                print("  Falling back to partitioned raw data...")
                ddf = self._load_from_partitioned_raw_distributed()
                data_source = "raw_partitioned"
                print("  Raw loading ok")
            except Exception as e:
                self.logger.error(f"Error loading raw data: {e}")
                print(f"  [ERROR] Raw data: {e}")

        # Loading validation
        if ddf is None:
            raise FileNotFoundError(
                "Data Lake data not found in any source.\n"
                f"Check: {self.task_graph_path} or {self.fallback_path}\n"
                "Run 'task_graph/processor.py' to generate processed data."
            )

        # Adequacy analysis

        # Batch computation for efficiency (a single Dask compute call)
        stats_to_compute = {
            'year_min': ddf['year'].min(),
            'year_max': ddf['year'].max(),
            'n_countries': ddf['entity_id'].nunique(),
            'total_rows': ddf.index.size
        }
        computed_stats = dask.compute(stats_to_compute)[0]
        
        years_span = computed_stats['year_max'] - computed_stats['year_min'] + 1
        avg_obs_per_country = computed_stats['total_rows'] / computed_stats['n_countries']

        print(f"  {computed_stats['year_min']}-{computed_stats['year_max']} ({years_span} years)")
        print(f"  {computed_stats['n_countries']} countries ({avg_obs_per_country:.1f} obs/country)")
        print(f"  {computed_stats['total_rows']:,} total observations")
        print(f"  Source: {data_source}")

        if years_span < 10:
            print("  [WARN] Short time series may limit walk-forward validation")

        if computed_stats['n_countries'] < 15:
            print("  [WARN] Few countries may affect geographic generalization")

        self.logger.info(f"Data loaded successfully via {data_source}")
        
        return ddf
    
    @log_ml_pipeline('validation')
    def validate_data(self, ddf: dd.DataFrame) -> None:
        """
        Run distributed validation with strategic sampling.

        Args:
            ddf: Dask DataFrame with the loaded educational data

        Validation methodology:
            1. Adaptive sampling: min(1000, total_rows) for efficiency
            2. DataIntegrityValidator: Centralized validator for consistency
            3. Schema validation: Check for mandatory columns
            4. Range validation: Detection of impossible values
            5. Smart fallback: Automatic search for alternative variables

        Schema-on-Read paradigm:
            Validation run after loading, allowing flexibility in the structure
            of the data while guaranteeing minimum quality for ML.

        Criteria:
            - Target coverage >50%: Adequate statistical power for ML
            - Range [0,100]: Consistency with educational definitions
            - Schema compliance: Presence of temporal/geographic identifiers

        Sampling:
            For efficiency, a sample of min(1000, total_rows) observations.
        """
        print("Validating data")

        # Adaptive sampling for efficient validation
        total_rows = int(ddf.index.size.compute())
        sample_size = min(1000, total_rows)  # Balances precision vs efficiency

        print(f"  Sampling: {sample_size:,}/{total_rows:,} ({sample_size/total_rows:.1%})")

        # Sample creation preserving the distribution
        sample_df = ddf.head(sample_size, npartitions=ddf.npartitions)
        if hasattr(sample_df, 'compute'):
            sample_df = sample_df.compute()
        
        # Centralized validation with DataIntegrityValidator
        is_valid, validation_report = self.data_validator.validate_dataframe(
            sample_df,
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
        if self.source_column not in ddf.columns:
            raise ValueError(
                f"Target column '{self.source_column}' declared by "
                f"{type(self.dataset_config).__name__} is absent from the "
                f"processed data. Available columns: {sorted(ddf.columns)}"
            )
        
        # Distributed quality analysis

        # Optimized batch computation (a single Dask call)
        validation_stats = {
            'target_data': (~ddf[self.source_column].isna()).sum(),
            'target_min': ddf[self.source_column].min(),
            'target_max': ddf[self.source_column].max(),
            'target_mean': ddf[self.source_column].mean(),
            'over_100_count': (ddf[self.source_column] > 100).sum(),
            'under_0_count': (ddf[self.source_column] < 0).sum(),
            'total_rows': ddf.index.size
        }
        computed = dask.compute(validation_stats)[0]
        
        total_rows = computed['total_rows']
        target_coverage = (computed['target_data'] / total_rows) * 100

        print(f"  Coverage: {computed['target_data']:,}/{total_rows:,} valid ({target_coverage:.1f}%)")
        print(f"  Range: [{computed['target_min']:.1f}%, {computed['target_max']:.1f}%]")
        print(f"  Mean: {computed['target_mean']:.1f}%")

        if target_coverage < 50:
            print("  [WARN] Low target coverage (<50%) may compromise ML")

        if computed['over_100_count'] > 0:
            print(f"  [WARN] {computed['over_100_count']} values >100% (invalid data)")

        if computed['under_0_count'] > 0:
            print(f"  [WARN] {computed['under_0_count']} values <0% (invalid data)")

        # Mandatory schema validation
        required_cols = ['entity_id', 'year']
        missing_cols = [col for col in required_cols if col not in ddf.columns]
        if missing_cols:
            raise ValueError(
                f"Incomplete schema for temporal ML: missing columns {missing_cols}.\n"
                "Country-year identifiers are mandatory for walk-forward validation."
            )

        print("  Validation complete")
    
    def create_target_implementation(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Build the target variable via a distributed Dask transformation.

        Args:
            ddf: Dask DataFrame with educational data

        Returns:
            Dask DataFrame enriched with the target variable dropout_rate_task_graph

        Transformation:
            Dropout Rate = 100 - Completion Rate

        Educational rationale:
            Following UNESCO (2018) and World Bank Education Statistics,
            dropout rate offers direct interpretability for policy:
            - High values = need for urgent intervention
            - Standardized international comparability
            - Standardized international comparability

        Dask paradigm:
            Transformation applied lazily via .apply() with a meta specification
            to preserve types and optimize the computational graph.
        """
        print("Building target variable")

        def create_dropout_rate(completion_rate):
            """
            Pure function for the completion -> dropout rate transformation.

            Args:
                completion_rate: Completion rate (0-100%)

            Returns:
                Dropout rate (0-100%), or NaN if outside the valid range.

            Preserves NaN for missing values (does not impute artificially).
            Validates the range [0,100].
            """
            if pd.isna(completion_rate) or completion_rate < 0 or completion_rate > 100:
                return float('nan')
            return 100 - completion_rate
        
        print(f"  {self.source_column} -> {self.target_column}")
        print("  Dropout Rate = 100 - Completion Rate")
        
        # Distributed Dask transformation with a meta specification
        ddf_with_target = ddf.assign(
            **{self.target_column: ddf[self.source_column].apply(
                create_dropout_rate,
                meta=(self.target_column, 'f8')  # Float64 type specification
            )}
        )

        print("  Target created via Dask lazy evaluation")
        try:
            base = ddf_with_target[['entity_id', 'year', self.target_column]].rename(
                columns={self.target_column: 'dropout_rate_t'}
            )
            prev = base.assign(year=base['year'] + 2).rename(columns={'dropout_rate_t': 'dropout_rate_lag_2'})
            merged = dd.merge(ddf_with_target, prev[['entity_id', 'year', 'dropout_rate_lag_2']],
                              on=['entity_id', 'year'], how='left')
            # Lag of 3 years
            prev3 = base.assign(year=base['year'] + 3).rename(columns={'dropout_rate_t': 'dropout_rate_lag_3'})
            merged = dd.merge(merged, prev3[['entity_id', 'year', 'dropout_rate_lag_3']],
                              on=['entity_id', 'year'], how='left')
            ddf_with_target = merged
            print("  dropout_rate_lag_2 and dropout_rate_lag_3 created (join country/year-k)")
        except Exception as exc:
            raise ValueError(
                f"task_graph: failed to create the target lags: {exc}"
            ) from exc

        assert_lag_columns(ddf_with_target.columns, 'task_graph',
                           self.TARGET_LAG_ORDERS,
                           target_stem=self.TARGET_STEM)
        return ddf_with_target
    
    def _compute_target_statistics(self, ddf: dd.DataFrame) -> Dict[str, float]:
        """
        Compute descriptive statistics of the target variable via distributed Dask.

        Args:
            ddf: Dask DataFrame with the target variable created

        Returns:
            Dictionary with float64 statistics for analysis

        Statistics computed:
            - Moments: mean, standard deviation (unbiased)
            - Range: minimum, maximum for outlier detection
            - Completeness: valid vs missing count for quality analysis

        Distributed optimization:
            A single dask.compute() call to minimize materialization of the
            computational graph.
        """
        # Optimized batch computation for distributed efficiency
        stats_batch = {
            'mean': ddf[self.target_column].mean(),
            'std': ddf[self.target_column].std(),
            'min': ddf[self.target_column].min(),
            'max': ddf[self.target_column].max(),
            'missing_count': ddf[self.target_column].isna().sum(),
            'valid_count': (~ddf[self.target_column].isna()).sum()
        }
        
        # A single compute call for maximum efficiency
        computed = dask.compute(stats_batch)[0]

        # Conversion to float64 for consistency
        return {key: self.reported_statistic(value)
                for key, value in computed.items()}
    
    def _validate_temporal_folds(self, ddf: dd.DataFrame, folds: List[Dict]) -> None:
        """Temporal validation with TemporalValidator."""
        print("Validating temporal folds")

        # Validation via the centralized TemporalValidator
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
            
            train_filter = (
                (ddf['year'] >= fold['train_start']) &
                (ddf['year'] <= fold['train_end']) &
                ~((ddf['year'] >= fold['train_gap_start']) &
                  (ddf['year'] <= fold['train_gap_end']))
            )
            val_filter = (ddf['year'] >= fold['val_start']) & (ddf['year'] <= fold['val_end'])
            test_filter = (
                (ddf['year'] >= fold['test_start']) &
                (ddf['year'] <= fold['test_end']) &
                ~((ddf['year'] >= fold['val_gap_start']) &
                  (ddf['year'] <= fold['val_gap_end']))
            )
            
            # Count data per fold
            fold_stats = {
                'train_count': train_filter.sum(),
                'val_count': val_filter.sum(),
                'test_count': test_filter.sum(),
                'train_countries': ddf[train_filter]['entity_id'].nunique(),
                'val_countries': ddf[val_filter]['entity_id'].nunique(),
                'test_countries': ddf[test_filter]['entity_id'].nunique()
            }
            computed_fold = dask.compute(fold_stats)[0]
            fold.update(computed_fold)
            
            print(f"\n  Fold {fold['fold_id']}:")
            print(f"    Train: {fold['train_count']} obs, {fold['train_countries']} countries")
            print(f"    Val: {fold['val_count']} obs, {fold['val_countries']} countries")
            print(f"    Test: {fold['test_count']} obs, {fold['test_countries']} countries")
    
    def discover_numeric_columns(self, ddf: dd.DataFrame) -> List[str]:
        """
        Identify numeric columns via type inference over the inferred schema.

        Args:
            ddf: Dask DataFrame with educational data

        Returns:
            List of numeric column names

        Schema-on-Read methodology:
            Uses Pandas.select_dtypes() over a dynamically inferred schema,
            allowing flexibility in the input structure typical of Data Lakes.

        Limitations:
            - Does not detect numeric categorical variables (codes, IDs)
            - Ignores derived features not materialized in the DataFrame
            - Schema inference can be costly for very wide DataFrames
        """
        return ddf.select_dtypes(include=[np.number]).columns.tolist()
    
    def compute_feature_correlations(self, ddf: dd.DataFrame,
                                    features: List[str]) -> Dict[str, float]:
        """
        Compute feature-target Pearson correlations over the complete data.

        Args:
            ddf: Dask DataFrame with the complete educational data
            features: List of candidate features for correlation analysis

        Returns:
            Dictionary {feature_name: absolute_correlation} for ranking

        Methodology:
            Materializes the complete training set and computes the pairwise
            Pearson correlation between each feature and the target. For
            datasets at the scale of this benchmark (~22K rows), complete
            materialization is feasible and removes sampling variance.
        """
        print("Analyzing feature-target correlations")

        target_col = self.target_column
        correlations = {}

        sample_df = ddf[features + [target_col]].compute().dropna(subset=[target_col])

        print(f"  Materialized data: {len(sample_df):,} obs, {len(features)} features")
        
        successful_correlations = 0
        failed_features = []
        
        for feat in features:
            if feat not in sample_df.columns:
                correlations[feat] = 0.0
                continue
                
            try:
                corr = sample_df[feat].corr(sample_df[target_col])
                
                if pd.isna(corr):
                    correlations[feat] = 0.0
                else:
                    correlations[feat] = abs(float(corr))
                    successful_correlations += 1
                    
            except Exception as e:
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
    

    def apply_collinearity_filter(self, ddf: dd.DataFrame, features: List[str],
                                   threshold: float = 0.8) -> List[str]:
        """
        Remove multicollinearity via greedy pairwise correlation filtering.

        For each candidate feature, computes the maximum absolute correlation
        with the features already selected and rejects it if max |r| >= threshold.

        Args:
            ddf: Dask DataFrame with candidate features
            features: List of features for multicollinearity analysis
            threshold: Pairwise correlation threshold (default 0.8)

        Returns:
            Filtered list of features with reduced multicollinearity

        Greedy algorithm:
            1. The first feature is always accepted (baseline)
            2. Subsequent features are accepted if max |r| < threshold
            3. Deterministic order (features sorted)
        """
        if len(features) <= 1:
            print("  Fewer than 2 features - collinearity check unnecessary")
            return features

        print(f"Filtering collinearity: {len(features)} features")

        try:
            corr_data = ddf[features].compute().dropna()

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
                print(f"  Insufficient data ({len(corr_data)}<=10) - top-10 fallback")
                return features[:10]

        except Exception as e:
            self.logger.error(f"Error in collinearity filtering: {e}")
            print(f"[ERROR] Collinearity filtering failed: {e}")
            print("  Fallback: returning top-10 features")
            return features[:10]
    
    @log_ml_pipeline('feature_engineering')
    def prepare_features(self, ddf: dd.DataFrame, selected_features: List[str]) -> dd.DataFrame:
        """
        Run feature engineering with a distributed symmetric log transform.

        Args:
            ddf: Dask DataFrame with features selected via collinearity filtering
            selected_features: Post-selection features for transformation

        Returns:
            Dask DataFrame enriched with original + transformed features

        Scientific feature engineering:
            Applies the symmetric log transform: T(x) = sign(x) * ln(|x| + 1)
            to the top-5 features to normalize the skewed distributions common
            in socioeconomic data.

        Methodological rationale:
            1. Top-5 limit: Based on the curse of dimensionality (Bellman, 1961)
               and overfitting on small educational samples
            2. Symmetric log: Handles zeros and negatives naturally, suitable for
               educational indicators with deficits/declines
            3. Lazy evaluation: Transformations applied via Dask .apply()
               for memory optimization

        Final structure:
            - Metadata: entity_id, year, target (essential for temporal ML)
            - Original features: selected_features (post collinearity filtering)
            - Transformed features: {feature}_log_transform (top-5)

        Architectural equivalence:
            Implements the same transformations as the Data Warehouse via SQL,
            guaranteeing comparability for benchmarking.

        Logging:
            Captures quality metrics (missing%, dimensionality) for auditing and
            reproducibility of the ML pipeline.

        """
        print("\nFeature engineering")

        # Copy to preserve the original DataFrame
        ddf_work = ddf.copy()

        # Symmetric log transform: T(x) = sign(x) * ln(|x| + 1)

        # Criterion: Limit scope because of the curse of dimensionality
        features_to_transform = selected_features[:5] if len(selected_features) > 5 else selected_features
        transformed_count = 0

        print(f"  Transforming {len(features_to_transform)} features (symmetric log):")

        # Transformation applied feature by feature
        for feat in features_to_transform:
            if feat not in ddf_work.columns:
                print(f"    {feat}: ABSENT (skipped)")
                continue

            transform_col = f"{feat}_log_transform"

            print(f"    {feat} -> {transform_col}")
            
            ddf_work[transform_col] = ddf_work[feat].apply(
                lambda x: np.sign(x) * np.log(np.abs(x) + 1) if pd.notna(x) else np.nan,
                meta=(transform_col, 'f8')  # Float64 metadata for Dask
            )
            transformed_count += 1

        print(f"  {transformed_count} log transforms applied")

        # Construction of the final ML dataset

        # Metadata essential for temporal ML
        ml_features = ['entity_id', 'year', self.target_column]

        # Original features after collinearity filtering
        ml_features.extend(selected_features)

        # Transformed features (only the ones that were created)
        transformed_cols = [f"{feat}_log_transform" for feat in features_to_transform
                          if f"{feat}_log_transform" in ddf_work.columns]
        ml_features.extend(transformed_cols)

        # Include the target lags in the saved dataset, even if not selected
        for lag_col in ['dropout_rate_lag_2', 'dropout_rate_lag_3']:
            if lag_col in ddf_work.columns and lag_col not in ml_features:
                ml_features.append(lag_col)

        # Remove duplicates preserving order
        ml_features = list(dict.fromkeys(ml_features))

        # Keep only columns that exist in the DataFrame
        ml_features = [col for col in ml_features if col in ddf_work.columns]

        print(f"  Final ML dataset: {len(ml_features)} variables "
              f"({len(selected_features)} original, {len(transformed_cols)} transformed)")

        # Final selection
        result_ddf = ddf_work[ml_features]

        # Logging for auditing
        try:
            total_rows = int(ddf.index.size.compute())
            sample_size = min(100, total_rows)

            # Computation of quality statistics
            if hasattr(result_ddf, 'compute'):
                sample_stats = result_ddf.head(sample_size, npartitions=result_ddf.npartitions)
            else:
                sample_stats = result_ddf.head(sample_size)

            # Proportion of missing values
            missing_pct = float(sample_stats.isna().mean().mean() * 100)

            # Structured log for reproducibility
            self.logger.log_data_info(
                "ml_ready_data",
                shape=(total_rows, len(ml_features)),
                missing_pct=missing_pct
            )
            
            print(f"  {missing_pct:.1f}% missing values (sample n={sample_size})")

        except Exception as e:
            self.logger.warning(f"Error computing quality statistics: {e}")
            print(f"  [WARN] Quality statistics unavailable: {e}")

        print("  Feature engineering complete")
        
        return result_ddf
    
    def save_folds(self, ddf: dd.DataFrame, folds: List[Dict]) -> None:
        """Save folds"""
        print("\nSaving folds")

        for fold in folds:
            fold_id = fold['fold_id']
            fold_dir = f"{self.prep_dir}/folds/fold_{fold_id}"
            os.makedirs(fold_dir, exist_ok=True)
            
            print(f"  Processing fold {fold_id}...")
            
            train_filter = (
                (ddf['year'] >= fold['train_start']) &
                (ddf['year'] <= fold['train_end']) &
                ~((ddf['year'] >= fold['train_gap_start']) &
                  (ddf['year'] <= fold['train_gap_end']))
            )
            val_filter = (
                (ddf['year'] >= fold['val_start']) &
                (ddf['year'] <= fold['val_end'])
            )
            test_filter = (
                (ddf['year'] >= fold['test_start']) &
                (ddf['year'] <= fold['test_end']) &
                ~((ddf['year'] >= fold['val_gap_start']) &
                  (ddf['year'] <= fold['val_gap_end']))
            )
            
            train_ddf = ddf[train_filter]
            val_ddf = ddf[val_filter]
            test_ddf = ddf[test_filter]
            
            # Convert to Pandas and save
            try:
                train_df = train_ddf.compute()
                val_df = val_ddf.compute()
                test_df = test_ddf.compute()
                
                train_df = train_df.reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)
                self._safe_write_parquet_file(train_df, f'{fold_dir}/train_task_graph.parquet')
                self._safe_write_parquet_file(val_df, f'{fold_dir}/val_task_graph.parquet')
                self._safe_write_parquet_file(test_df, f'{fold_dir}/test_task_graph.parquet')
                
                print(f"    Fold {fold_id}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
                
            except Exception as e:
                print(f"    [ERROR] Saving fold {fold_id}: {e}")
                raise
            
            fold_metadata = {
                **fold,
                'storage_method': 'parquet_files',
                'paradigm': 'schema_on_read'
            }
            self.save_fold_metadata(fold_metadata, fold_dir)
        
        # Master data
        print("\n  Saving master data...")
        try:
            master_path = f"{self.prep_dir}/master_data_task_graph.parquet"
            master_df = ddf.compute().reset_index(drop=True)
            self._safe_write_parquet_file(master_df, master_path)
            print(f"    Master data: {len(master_df)} records")

        except Exception as e:
            print(f"    [ERROR] Saving master data: {e}")
            raise

        # Master configuration
        total_obs = len(master_df)
        total_countries = master_df['entity_id'].nunique()
        year_min = int(master_df['year'].min())
        year_max = int(master_df['year'].max())
        
        self.save_master_config(folds, total_obs, total_countries, (year_min, year_max))
        
        print(f"  Dask: folds saved")
    
    
    def _load_from_partitioned_raw_distributed(self) -> dd.DataFrame:
        """Load partitioned data with Schema-on-Read."""
        parquet_files = glob.glob(f"{self.fallback_path}/**/*.parquet", recursive=True)

        if not parquet_files:
            raise FileNotFoundError("No parquet file found")

        ddf = dd.read_parquet(self.fallback_path, engine='pyarrow').persist()

        # Conversion to wide format if necessary
        if 'indicator_name' in ddf.columns:
            ddf = self._convert_to_wide_format_distributed(ddf)
        
        return ddf
    
    def _convert_to_wide_format_distributed(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """Wide format conversion - uses Pandas when necessary."""
        try:
            print("    Applying pivot to wide format...")
            
            df = ddf.compute()
            
            index_cols = ['entity_id', 'entity_name', 'year']
            if 'entity_stratum' in df.columns:
                index_cols.append('entity_stratum')
            
            # Pivot the data
            df_wide = df.pivot_table(
                index=index_cols,
                columns='indicator_name',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            df_wide.columns.name = None
            
            ddf_wide = dd.from_pandas(df_wide, npartitions=max(1, len(df_wide) // 10000))
            
            print(f"    Conversion complete: {len(ddf_wide.columns)} columns")
            return ddf_wide

        except Exception as e:
            print(f"    [ERROR] Wide conversion: {e}")
            return ddf

    def run_setup_with_monitoring(self) -> Dict[str, Any]:
        """Run setup with monitoring."""
        with self.logger.timer("complete_setup_pipeline"):
            results = self.run_setup()
            
            results['paradigm'] = 'schema_on_read_dask'
            results['standardized'] = True
            results['version'] = 'no_cache'

            self.logger.info(
                "Data Lake setup complete",
                total_time=results.get('processing_time'),
            )

            return results


def main():
    """Run the Data Lake pipeline end-to-end for local validation."""
    print("=" * 80)
    print("Dask ML Pipeline")
    print("=" * 80)

    setup = None
    try:
        setup = TaskGraphArchitectureML()
        results = setup.run_setup_with_monitoring()
        
        if results.get('status') == 'success':
            print("Pipeline ok")
            print(f"  Paradigm: {results.get('paradigm', 'N/A')}")
            print(f"  Features selected: {results.get('features_selected', 'N/A')}")
            print(f"  Temporal folds: {results.get('folds_created', 'N/A')}")
            print(f"  Processing: {results.get('processing_time', 'N/A')}s")
        else:
            print("[ERROR] Dask pipeline failed")
            if 'error' in results:
                print(f"  Error: {results['error']}")

        print(f"\nResults:")
        for key, value in results.items():
            if key not in ['status', 'error']:
                print(f"  {key}: {value}")

        return results

    except Exception as e:
        print(f"\n[ERROR] Dask pipeline failed: {e}")
        print("  Check whether the data was processed by task_graph/processor.py")
        return {'status': 'failed', 'error': str(e)}

    finally:
        # Symmetric across paradigms: the benchmark re-runs each phase twelve
        # times in the same process, and a resource that survives one
        # repetition is measured by the next.
        if setup is not None:
            setup.release_resources()
    


if __name__ == "__main__":
    results = main()
    # A failed setup must not report success to the pipeline, which runs each
    # stage as a subprocess and reads its exit status.
    sys.exit(0 if isinstance(results, dict)
             and results.get('status') == 'success' else 1)
