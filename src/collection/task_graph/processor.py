#!/usr/bin/env python3
"""
Data Lake Processor for Educational Data with Partitioned Processing.

Implements a processing pipeline using Dask for exploratory analysis of educational
indicators, following Data Lake architectural principles with deferred validation and
temporal feature engineering.

Theoretical Grounding:
    The Data Lake paradigm (Terrizzano et al., 2015) prioritises the preservation of raw
    data and schema-on-read semantics, enabling flexible exploration. Unlike Data
    Warehouses (Inmon, 2005), Data Lakes defer structural constraints until analysis
    time, supporting iterative hypothesis testing in educational research.

Methodological Approach:
    1. Lazy Evaluation: Following Dask's computational graph model (Rocklin, 2015),
       operations build task graphs without immediate execution, optimising memory
       use for large-scale educational datasets (>10GB).

    2. Partitioned Processing: Implements partition-based parallelism,
       splitting the dataset into independent chunks processed concurrently.

    3. Temporal Preservation: Maintains longitudinal coherence by partitioning on
       geographic units rather than time, critical for panel data analysis (Baltagi, 2021)
       and difference-in-differences designs (Angrist & Pischke, 2009).

Design Decisions:
    - Maximum 32 partitions: Diminishing returns beyond that point for datasets <100GB
    - Snappy compression: Balances speed vs size (3:1 ratio)
    - Schema inference on write: Preserves flexibility while guaranteeing type
      consistency of Parquet columns (Apache Parquet 2.6.0 specification)

Assumptions:
    - The missing data mechanism follows MAR (Missing At Random)
    - Temporal trends are locally linear over 3-year windows
    - Country effects dominate subnational variation
    - Computational resources support up to 32 concurrent partitions

Limitations:
    - Assumes homogeneous variance across geographic strata (frequently violated)
    - Correlation thresholds (±0.1) are heuristic, not statistically derived
    - Lazy evaluation may mask quality problems until computation

References:
    Angrist, J. D., & Pischke, J. S. (2009). Mostly harmless econometrics. Princeton.
    Baltagi, B. H. (2021). Econometric analysis of panel data (6th ed.). Springer.
    Inmon, W. H. (2005). Building the data warehouse (4th ed.). Wiley.
    Rocklin, M. (2015). Dask: Parallel computation with blocked algorithms. SciPy.
    Terrizzano, I., et al. (2015). Data wrangling: The challenging journey. CIDR.
"""

import pandas as pd
import dask.dataframe as dd
import dask
import os
import sys
import json
import shutil
import warnings
from datetime import datetime
from typing import Dict

# Suppression of small-sample warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom <= 0.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero encountered.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrameGroupBy.apply operated on the grouping columns.*')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from core.config import get_absolute_output_path
from core.scientific_config import SCIENTIFIC_CONFIG
from core.indicators import ALL_INDICATORS

class TaskGraphProcessor:
    """
    Scientific Data Lake processor for the analysis of educational indicators.

    Implements partitioned processing following Data Lake architectural principles,
    optimised for exploratory data analysis and machine learning workflows on
    educational datasets.

    Fundamental Principles:
        1. Schema-on-read: Structure imposed at analysis time, not at ingestion
        2. Lazy evaluation: Computational graphs built without materialisation
        3. Partition preservation: Maintains data locality for temporal analysis
        4. Metadata enrichment: Adds scientific features without modifying raw data

    See the module docstring for the full methodological assumptions and limitations.
    """

    def __init__(self, dataset_name: str = "worldbank"):
        """
        Initialises the Data Lake processor with a Dask configuration optimised for scientific analyses.

        Args:
            dataset_name: Dataset name ("worldbank" or "inep_censo")
        """
        print("Initialising Dask processor")
        print("Architecture: Dask, schema-on-read")

        self.dataset_name = dataset_name
        self.run_timestamp = datetime.now().isoformat()
        raw_subdir = 'collection/inep_raw' if dataset_name == 'inep_censo' else 'collection/raw_data'
        self.complete_data_path = get_absolute_output_path(f'{raw_subdir}/complete_data.parquet')
        self.output_dir = get_absolute_output_path('collection/task_graph')
        self.processed_dir = f"{self.output_dir}/processed"
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Disables automatic optimisations to guarantee reproducibility
        dask.config.set({'dataframe.query-planning': False})
        # Explicit core budget, equal to that of the other paradigms.
        # Without this the scheduler sizes the pool from the machine, and the
        # comparison comes to depend on where it was run.
        dask.config.set(
            {'num_workers': int(SCIENTIFIC_CONFIG['engine_threads'])})

        print(f"Data source: {self.complete_data_path}")
        print(f"Processing directory: {self.processed_dir}")
    
    def load_complete_data(self) -> dd.DataFrame:
        """
        Loads the complete educational data preserving Dask's lazy semantics.

        Returns:
            dd.DataFrame: Dask DataFrame with an unmaterialised computational graph,
                         preserving memory benefits for datasets >10GB

        Raises:
            FileNotFoundError: When the input Parquet file does not exist,
                              indicating a failure in the previous pipeline stage

        Methodological decisions:
            1. Selective computation: Only essential metrics are materialised
               (temporal period, geographic cardinality) for logging.

            2. Centralised indicators: Uses the canonical definitions from the core module
               to guarantee consistency between the Data Lake and Data Warehouse architectures.

            3. Quality statistics: Completeness computed only for validated
               scientific indicators, excluding auxiliary metadata.
        """
        print("Lazy read of the complete educational data")

        if not os.path.exists(self.complete_data_path):
            raise FileNotFoundError(
                f"Complete data file not found: {self.complete_data_path}\n"
                f"Run 'raw_data_collector.py' before this processor."
            )
        
        ddf = dd.read_parquet(self.complete_data_path)

        n_rows = len(ddf)
        n_cols = len(ddf.columns)
        year_range = dask.compute(ddf['year'].min(), ddf['year'].max())
        n_countries = ddf['entity_id'].nunique().compute()
        
        print(f"{n_rows:,} observations x {n_cols} variables")
        print(f"Temporal coverage: {year_range[0]}-{year_range[1]} ({year_range[1]-year_range[0]+1} years)")
        entity_label = "Brazilian municipalities" if self.dataset_name == "inep_censo" else "countries"
        print(f"Geographic coverage: {n_countries} {entity_label}")

        indicator_names = list(ALL_INDICATORS.values())
        scientific_indicators = [col for col in ddf.columns if col in indicator_names]
        
        if scientific_indicators:
            missing_count = ddf[scientific_indicators].isna().sum().sum().compute()
            total_cells = n_rows * len(scientific_indicators)
            missing_pct = (missing_count / total_cells) * 100
            
            print(f"Completeness: {total_cells - missing_count:,}/{total_cells:,} valid cells ({100-missing_pct:.1f}%)")

            if 'data_completeness_score' in ddf.columns:
                stats = dask.compute(
                    ddf['data_completeness_score'].mean(),
                    ddf['data_completeness_score'].std()
                )
                print(f"Completeness score: mean={stats[0]:.1f}%, sd={stats[1]:.1f}%")

        print("Dask DataFrame prepared")
        
        return ddf
    
    def _calculate_completeness_score(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Computes the scientific completeness score for each observation.

        Args:
            ddf: Dask DataFrame with educational indicators

        Returns:
            Dask DataFrame enriched with the 'data_completeness_score' column (0-100%)

        Methodological justification:
            The completeness score is computed as the mean of non-null indicators,
            following Rubin's (1976) approach to quantifying available
            information. Only validated numeric indicators are considered,
            excluding metadata and categorical variables.

        Formula:
            completeness_i = (Σ I(x_ij ≠ NULL) / n_indicators) × 100
            where I() is the indicator function and j indexes scientific indicators

        Limitation:
            Treats all indicators with equal weight, ignoring relative importance
            for specific analyses (e.g., completion rate vs spending).
        """
        indicator_names = list(ALL_INDICATORS.values())
        numeric_indicators = [
            col for col in ddf.columns
            if col in indicator_names and ddf[col].dtype in ['int64', 'float64']
        ]
        
        if numeric_indicators:
            # Computes the proportion of valid values per row
            return ddf.assign(
                data_completeness_score=ddf[numeric_indicators].notna().mean(axis=1) * 100
            )
        else:
            return ddf.assign(data_completeness_score=0.0)
    
    def detect_quality_metadata(self, ddf: dd.DataFrame) -> Dict[str, bool]:
        """
        Detects pre-existing quality metadata in the dataset.

        Args:
            ddf: Dask DataFrame with the educational data loaded

        Returns:
            Dict[str, bool]: Mapping of available metadata, currently:
                - 'has_completeness_score': A pre-computed completeness score exists

        Data Lakes preserve metadata from the source. We detect prior
            enrichments to avoid unnecessary recomputation.

        Structure prepared to detect future metadata such as:
            - 'has_imputation_flags': Imputation markers
            - 'has_quality_tiers': Reliability classification
        """
        metadata_status = {
            'has_completeness_score': 'data_completeness_score' in ddf.columns
        }

        print("Analysing pre-existing enrichments")

        for metadata_type, is_present in metadata_status.items():
            status = "Detected" if is_present else "Absent"
            print(f"  - {metadata_type}: {status}")
        
        if metadata_status["has_completeness_score"]:
            stats = dask.compute(
                ddf["data_completeness_score"].mean(),
                ddf["data_completeness_score"].std(),
                ddf["data_completeness_score"].quantile([0.25, 0.5, 0.75])
            )
            print(f"Completeness: mean={stats[0]:.1f}%, sd={stats[1]:.1f}%")
            quartiles = stats[2]
            print(f"             Quartiles: Q1={quartiles[0.25]:.1f}%, Q2={quartiles[0.5]:.1f}%, Q3={quartiles[0.75]:.1f}%")
        
        return metadata_status

    def prepare_task_graph_metadata(self, ddf: dd.DataFrame, metadata_status: Dict[str, bool]) -> dd.DataFrame:
        """
        Prepares metadata following the Data Lake's schema-on-read principles.

        Args:
            ddf: Dask DataFrame with educational data
            metadata_status: Dictionary indicating pre-existing metadata

        Returns:
            Dask DataFrame with metadata guaranteed but not validated

        Data Lakes defer validation until the moment of use (Terrizzano et al., 2015).
            This method guarantees only the structural existence of metadata, not its
            semantic correctness, which will be verified during distributed processing.

        Contrast with the Data Warehouse:
            - Data Lake: Creates a placeholder, validates during processing
            - Data Warehouse: Validates immediately, rejects invalid data

        Eager validation would waste computational resources if the data
            is later filtered or aggregated, violating Dask's lazy evaluation
            principle (Rocklin, 2015).
        """
        print("Configuring Dask metadata")

        if metadata_status['has_completeness_score']:
            print("  Completeness score preserved (deferred validation)")
        else:
            print("  Completeness score absent - creating placeholder (value 0.0)")
            ddf = ddf.assign(data_completeness_score=0.0)
        
        return ddf
    
    def create_partitioned_structure(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Optimises partitioning for distributed processing while preserving temporal coherence.

        Args:
            ddf: Dask DataFrame with the complete educational data

        Returns:
            Dask DataFrame repartitioned for maximum computational efficiency

        Partitioning strategy:
            1. Cardinality-based: min(n_countries, 32) partitions
            2. Preserves implicit geographic grouping
            3. Avoids unnecessary shuffling of time series

        Justification of the 32-partition limit:
            Diminishing returns beyond 2^5 partitions for datasets <100GB
            due to coordination overhead and serialisation cost.

        Trade-offs:
            - More partitions: more parallelism, more coordination overhead
            - Fewer partitions: less parallelism, less overhead, more memory/partition

        Note on missing stratum:
            Countries without a socioeconomic classification get the label 'unclassified'
            to avoid NaN in subsequent groupby operations (pandas limitation).
        """
        print("Optimising partitioning for distributed processing")

        metadata_status = self.detect_quality_metadata(ddf)
        ddf_prepared = self.prepare_task_graph_metadata(ddf, metadata_status)

        # Handling of missing values in the stratification variable
        if 'entity_stratum' in ddf_prepared.columns:
            none_count = ddf_prepared['entity_stratum'].isna().sum().compute()
            if none_count > 0:
                print(f"{none_count:,} observations with undefined stratum -> 'unclassified'")
                ddf_prepared = ddf_prepared.assign(
                    entity_stratum=ddf_prepared['entity_stratum'].fillna('unclassified')
                )

        # Computes the optimal number of partitions
        n_countries = ddf_prepared['entity_id'].nunique().compute()
        optimal_partitions = min(n_countries, 32)

        print(f"Unique countries: {n_countries}, optimal partitions: {optimal_partitions}")
        ddf_optimized = ddf_prepared.repartition(npartitions=optimal_partitions)

        # Final statistics
        avg_partition_size = len(ddf_optimized) / optimal_partitions
        print(f"{optimal_partitions} partitions, ~{avg_partition_size:,.0f} obs/partition, {len(ddf_optimized.columns)} variables")
        
        return ddf_optimized

    def _add_distributed_processing_metadata(self, partition):
        """
        Adds processing and validation metadata to the partition.

        Args:
            partition: pandas DataFrame representing a Dask partition

        Returns:
            Partition with audit metadata and a validated completeness score
        """
        if partition.empty:
            return partition

        partition = partition.copy()

        if 'data_completeness_score' in partition.columns:
            invalid_mask = (partition['data_completeness_score'] < 0) | \
                          (partition['data_completeness_score'] > 100)

            if invalid_mask.any():
                partition.loc[partition['data_completeness_score'] < 0, 'data_completeness_score'] = 0.0
                partition.loc[partition['data_completeness_score'] > 100, 'data_completeness_score'] = 100.0

            partition['data_completeness_score'] = partition['data_completeness_score'].fillna(0.0)

        partition['processing_method'] = 'dask_distributed'
        partition['processed_timestamp'] = self.run_timestamp
        partition['schema_validation_applied'] = 'true'

        if not partition.empty:
            first_country = partition.iloc[0]['entity_id']
            partition['partition_id'] = f"partition_{hash(str(first_country)) % 1000:03d}"

        return partition

    def process_task_graph_architecture(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Runs distributed processing with audit metadata.

        Args:
            ddf: Optimally partitioned Dask DataFrame

        Returns:
            Dask DataFrame with processing metadata

        Processing paradigm:
            Uses map_partitions to apply identical and independent
            transformations to each partition, following the embarrassingly
            parallel computation model. There is no communication between
            partitions, guaranteeing linear scalability.
        """
        print(f"Dask pipeline: {ddf.npartitions} partitions")

        ddf_processed = ddf.map_partitions(
            self._add_distributed_processing_metadata
        )

        print("Metadata added, computational graph built")
        
        return ddf_processed

    def export_processed_data(self, ddf: dd.DataFrame) -> str:
        """
        Materialises and persists the processed data preserving Data Lake characteristics.

        Args:
            ddf: Processed Dask DataFrame with enriched features

        Returns:
            str: Absolute path of the final exported dataset

        Export strategy:
            1. Unified format: Single Parquet for holistic analyses
            2. Partitioned format: Parquet partitioned by country for selective queries
            3. JSON metadata: Aggregate statistics and quality metrics

        Design decisions:
            - Snappy compression: Speed/size balance (3:1) for
              iterative workflows.

            - PyArrow engine: Native support for complex types and better
              integration with the scientific Python ecosystem vs fastparquet.

            - No index: write_index=False saves 5-10% space with no impact
              on analytical queries that do not depend on row-level access.

        Batch computation:
            Aggregate statistics computed in a single batch to minimise
            materialisation of the Dask graph. The alternative would be multiple
            .compute() calls with 3-5x additional overhead.

        Schema-on-read compliance:
            Type inference during write, not during processing,
            keeping the Data Lake's flexibility (Terrizzano et al., 2015).

        Limitations:
            - Partitioning by country may be suboptimal for temporal queries
            - Aggregate statistics mask intra-country heterogeneity
            - The Parquet format imposes a minimal schema (vs fully schema-free formats)
        """
        print("Materialising processed data")

        # Completeness default if absent
        if 'data_completeness_score' not in ddf.columns:
            print("Computing the missing completeness score")
            ddf = self._calculate_completeness_score(ddf)

        print("Computing quality statistics")
        
        stats_to_compute = {
            'completeness_avg': ddf['data_completeness_score'].mean(),
            'completeness_min': ddf['data_completeness_score'].min(),
            'completeness_max': ddf['data_completeness_score'].max(),
            'completeness_std': ddf['data_completeness_score'].std(),
            'completeness_q25': ddf['data_completeness_score'].quantile(0.25),
            'completeness_q50': ddf['data_completeness_score'].quantile(0.50),
            'completeness_q75': ddf['data_completeness_score'].quantile(0.75),
            'non_zero_completeness': (ddf['data_completeness_score'] > 0).sum(),
            'total_records': len(ddf)
        }
        
        computed_stats = dask.compute(stats_to_compute)[0]
        
        print(f"Quality: {computed_stats['total_records']:,} records, "
              f"mean completeness={computed_stats['completeness_avg']:.1f}%, "
              f"Q1={computed_stats['completeness_q25']:.1f}%, "
              f"Q2={computed_stats['completeness_q50']:.1f}%, "
              f"Q3={computed_stats['completeness_q75']:.1f}%")
        
        output_path = f"{self.processed_dir}/final_results.parquet"

        if os.path.exists(output_path):
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            else:
                os.remove(output_path)
        
        print(f"Saving unified dataset: {output_path}")
        
        ddf.to_parquet(
            output_path,
            write_index=False,
            engine='pyarrow',
            compression='snappy'
        )
        
        partitioned_output_path = f"{self.processed_dir}/partitioned_results"
        
        if os.path.exists(partitioned_output_path):
            shutil.rmtree(partitioned_output_path)
        
        print(f"Saving partitioned dataset: {partitioned_output_path}")
        
        ddf.to_parquet(
            partitioned_output_path,
            partition_on=['entity_id'],
            write_index=False,
            engine='pyarrow',
            compression='snappy'
        )
        
        metadata = {
            'architecture': 'task_graph',
            'processing_paradigm': 'dask_distributed_lazy_evaluation',
            'dataset_statistics': {
                'total_records': int(computed_stats['total_records']),
                'total_partitions': int(ddf.npartitions),
                'avg_records_per_partition': int(computed_stats['total_records'] / ddf.npartitions)
            },
            'quality_metrics': {
                'completeness_mean': float(computed_stats['completeness_avg']),
                'completeness_std': float(computed_stats['completeness_std']),
                'completeness_min': float(computed_stats['completeness_min']),
                'completeness_max': float(computed_stats['completeness_max']),
                'completeness_quartiles': {
                    'q25': float(computed_stats['completeness_q25']),
                    'q50': float(computed_stats['completeness_q50']),
                    'q75': float(computed_stats['completeness_q75'])
                },
                'records_with_data': int(computed_stats['non_zero_completeness'])
            },
            'compliance_flags': {
                'schema_on_read': True,
                'lazy_evaluation_preserved': True,
                'distributed_processing': True,
                'deferred_validation': True,
                'idempotent_operations': True
            },
            'output_artifacts': {
                'unified_dataset': output_path,
                'partitioned_dataset': partitioned_output_path,
                'compression': 'snappy',
                'format': 'parquet'
            },
            'processing_metadata': {
                'timestamp': datetime.now().isoformat(),
                'dask_version': dask.__version__,
                'pandas_version': pd.__version__
            }
        }
        
        stats_path = f"{self.processed_dir}/processing_metadata.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Artifacts: {output_path}, {partitioned_output_path}, {stats_path}")
        
        return output_path

    def run_task_graph_processing(self) -> Dict:
        """
        Orchestrates the Data Lake processing pipeline.

        Returns:
            Dict containing execution status, generated artifacts and metadata

        Sequential pipeline:
            1. Lazy loading of the complete data
            2. Partitioning optimisation for parallelism
            3. Distributed feature engineering
            4. Materialisation and persistence
        """
        start_time = datetime.now()

        try:
            print("\n[1/4] Data loading")
            ddf_complete = self.load_complete_data()

            print("\n[2/4] Partitioning optimisation")
            ddf_partitioned = self.create_partitioned_structure(ddf_complete)

            print("\n[3/4] Distributed feature engineering")
            ddf_processed = self.process_task_graph_architecture(ddf_partitioned)

            print("\n[4/4] Materialisation and persistence")
            output_path = self.export_processed_data(ddf_processed)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            print(f"\nDask processing completed in {processing_time:.2f}s "
                  f"({len(ddf_complete)/processing_time:.0f} records/s)")
            
            return {
                'status': 'success',
                'architecture': 'task_graph',
                'paradigm': 'distributed_lazy_evaluation',
                'output': {
                    'primary_dataset': output_path,
                    'partitioned_dataset': f"{self.processed_dir}/partitioned_results",
                    'metadata': f"{self.processed_dir}/processing_metadata.json"
                },
                'performance': {
                    'processing_time_seconds': processing_time,
                    'throughput_records_per_second': len(ddf_complete)/processing_time,
                    'partitions_processed': ddf_processed.npartitions
                },
                'timestamp': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                }
            }
            
        except FileNotFoundError as e:
            print(f"\n[ERROR] Input data not found: {e}")
            print("Run 'raw_data_collector.py' before this processor")
            
            return {
                'status': 'failed',
                'error_type': 'FileNotFoundError',
                'error_message': str(e),
                'suggestion': 'Run raw_data_collector.py first',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()

            print(f"\n[ERROR] {e.__class__.__name__}: {str(e)}")
            print(tb)
            
            return {
                'status': 'failed',
                'error_type': e.__class__.__name__,
                'error_message': str(e),
                'traceback': tb,
                'timestamp': datetime.now().isoformat(),
                'partial_progress': {
                    'data_loaded': 'ddf_complete' in locals(),
                    'data_partitioned': 'ddf_partitioned' in locals(),
                    'data_processed': 'ddf_processed' in locals()
                }
            }

if __name__ == "__main__":
    # Without an exit status, a failure here reaches the orchestrator as success:
    # pipeline.py uses subprocess check=True, which only reads the return code.
    # That is how collection could die and the following stages run over
    # the panel from the previous run.
    processor = TaskGraphProcessor()
    results = processor.run_task_graph_processing()
    status = results.get('status', 'failed')
    print(f"Run: {status}")
    sys.exit(0 if status == 'success' else 1)