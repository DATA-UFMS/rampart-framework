#!/usr/bin/env python3
"""
Polars DataFrame Processor for Latin American Educational Data.

Implements a processing pipeline with Polars while keeping Data Lake
architectural principles: lazy reading, memory-efficient transformations and
schema-on-read.

Theoretical Grounding:
    The Data Lake paradigm (Terrizzano et al., 2015) prioritises the preservation
    of raw data and schema-on-read semantics. Polars offers native lazy evaluation,
    unlike eager Pandas, enabling automatic query plan optimisations.

    Architectural differences vs Dask:
    - Polars: Single-machine lazy evaluation with query plan optimisations
    - Dask: Multi-machine distributed with coordination overhead
    - Polars is suitable for datasets <100GB on modern machines (≥32GB RAM)

Design Decisions:
    - Polars lazy scanning: schema-on-read with automatic optimisation
    - In-memory pivoting: trade-off between speed and RAM use
    - No partitioning: Polars optimises automatically via the query plan
    - Parquet as the default format: ecosystem compatibility
"""

import polars as pl
import os
import sys
import json
import shutil
from datetime import datetime
from typing import Dict


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from core.config import get_absolute_output_path
from core.indicators import ALL_INDICATORS


class DataFrameLibProcessor:
    """
    Scientific Polars DataFrame processor for educational indicator analysis.

    Implements lazy processing with Polars following Data Lake architectural
    principles, optimised for exploratory data analysis with a smaller memory
    footprint than Dask for datasets <100GB.

    Fundamental Principles:
        1. Native lazy evaluation: Polars builds optimised query plans automatically
        2. Schema-on-read: Structure imposed at analysis time, not at ingestion
        3. Efficient transformations: Vectorised operations in compiled Rust
        4. Automatic pivoting: long->wide conversion optimised for Parquet
    """

    def __init__(self, dataset_name: str = "worldbank"):
        """
        Initialize the Polars DataFrame processor.

        Args:
            dataset_name: Dataset name ("worldbank" or "inep_censo")
        """
        print("Initializing Polars processor")
        print("Architecture: Polars, schema-on-read")

        self.dataset_name = dataset_name
        raw_subdir = 'collection/inep_raw' if dataset_name == 'inep_censo' else 'collection/raw_data'
        self.complete_data_path = get_absolute_output_path(f'{raw_subdir}/complete_data.parquet')
        self.output_dir = get_absolute_output_path('collection/dataframe_lib')
        self.processed_dir = f"{self.output_dir}/processed"

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        print(f"Data source: {self.complete_data_path}")
        print(f"Processing directory: {self.processed_dir}")

    def load_complete_data(self) -> pl.LazyFrame:
        """
        Load the complete educational data with lazy evaluation.

        Returns:
            pl.LazyFrame: Lazy Polars DataFrame without materialization

        Raises:
            FileNotFoundError: When the input Parquet file does not exist

        Methodological decisions:
            1. Lazy scanning: Only the schema is read initially
            2. Selective computation: Critical statistics computed selectively
            3. Centralized indicators: Guarantees consistency across architectures
        """
        print("Lazy reading of the complete educational data")

        if not os.path.exists(self.complete_data_path):
            raise FileNotFoundError(
                f"Complete data file not found: {self.complete_data_path}\n"
                f"Run 'raw_data_collector.py' before this processor."
            )

        df_lazy = pl.scan_parquet(self.complete_data_path)

        n_rows = df_lazy.select(pl.lit(1)).collect().shape[0]
        n_cols = len(df_lazy.collect_schema().names())

        # Compute only the critical statistics
        stats = df_lazy.select([
            pl.col('year').min().alias('year_min'),
            pl.col('year').max().alias('year_max'),
            pl.col('country_code').n_unique().alias('n_countries')
        ]).collect()

        year_min = stats['year_min'][0]
        year_max = stats['year_max'][0]
        n_countries = stats['n_countries'][0]

        print(f"{n_rows:,} observations x {n_cols} variables")
        print(f"Temporal coverage: {year_min}-{year_max} ({year_max-year_min+1} years)")
        entity_label = "Brazilian municipalities" if self.dataset_name == "inep_censo" else "countries"
        print(f"Geographic coverage: {n_countries} {entity_label}")

        indicator_names = list(ALL_INDICATORS.values())
        scientific_indicators = [col for col in df_lazy.collect_schema().names()
                                if col in indicator_names]

        if scientific_indicators:
            missing_stats = df_lazy.select([
                pl.concat_list([pl.col(col).is_null() for col in scientific_indicators])
                  .list.sum().sum().alias('total_missing'),
                pl.lit(len(scientific_indicators)).alias('n_indicators')
            ]).collect()

            total_cells = n_rows * len(scientific_indicators)
            missing_count = missing_stats['total_missing'][0] * len(scientific_indicators)
            missing_pct = (missing_count / total_cells) * 100 if total_cells > 0 else 0

            print(f"Completeness: {total_cells - missing_count:,}/{total_cells:,} valid cells ({100-missing_pct:.1f}%)")

        print("Polars LazyFrame prepared")

        return df_lazy

    def pivot_long_to_wide(self, df_lazy: pl.LazyFrame) -> pl.LazyFrame:
        """
        Transform data from long to wide format.

        Args:
            df_lazy: Lazy Polars DataFrame in long format

        Returns:
            Lazy Polars DataFrame in wide format

        Transformation logic:
            1. Original data: (country, year, indicator, value)
            2. Final result: (country, year, indicator1, indicator2, ...)
            3. Uses unpivot (Polars), equivalent to a reverse melt
        """
        print("Pivot long->wide")

        # Check whether it is already in wide format
        schema = df_lazy.collect_schema()

        # If it has 'indicator_name' or 'indicator', a pivot is needed
        if 'indicator_name' in schema or 'indicator' in schema:
            print("Long format detected - converting to wide")

            # Identify the indicator column
            indicator_col = 'indicator_name' if 'indicator_name' in schema else 'indicator'
            value_col = 'value' if 'value' in schema else 'indicator_value'

            # Keep the dimension columns (country, year, etc.)
            id_cols = ['country_code', 'year']

            # Pivot: turn indicators into columns
            df_wide = df_lazy.pivot(
                on=indicator_col,
                index=id_cols,
                values=value_col,
                aggregate_function='first'  # There should be no duplicates
            )

            print(f"Pivot complete - columns: {len(df_wide.collect_schema().names())}")

        else:
            print("Data already in wide format - preserving structure")
            df_wide = df_lazy

        return df_wide

    def export_processed_data(self, df_lazy: pl.LazyFrame) -> str:
        """
        Materialize and persist the processed data.

        Args:
            df_lazy: Processed lazy Polars DataFrame

        Returns:
            str: Absolute path of the exported final dataset

        Export strategy:
            1. Lazy materialization: Polars optimises automatically before writing
            2. Parquet format: Ecosystem compatibility
            3. JSON metadata: Quality and audit statistics
        """
        print("Materializing processed data")

        output_path = f"{self.processed_dir}/final_results.parquet"

        if os.path.exists(output_path):
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            else:
                os.remove(output_path)

        print(f"Saving dataset: {output_path}")

        df_collected = df_lazy.collect()
        df_collected.write_parquet(output_path, compression='snappy')

        # Final statistics
        n_rows = len(df_collected)
        n_cols = len(df_collected.columns)

        # JSON metadata
        metadata = {
            'architecture': 'dataframe_lib',
            'processing_paradigm': 'polars_lazy_evaluation',
            'dataset_statistics': {
                'total_records': int(n_rows),
                'total_columns': int(n_cols)
            },
            'data_quality': {
                'format': 'parquet',
                'compression': 'snappy',
                'row_count': int(n_rows),
                'column_count': int(n_cols)
            },
            'compliance_flags': {
                'schema_on_read': True,
                'lazy_evaluation_preserved': True,
                'task_graph_pattern': True
            },
            'output_artifacts': {
                'dataset': output_path,
                'compression': 'snappy',
                'format': 'parquet'
            },
            'processing_metadata': {
                'timestamp': datetime.now().isoformat(),
                'polars_version': pl.__version__
            }
        }

        stats_path = f"{self.processed_dir}/processing_metadata.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Artifacts: {output_path}, {stats_path}")
        print(f"{n_rows:,} records x {n_cols} columns")

        return output_path

    def run_dataframe_lib_processing(self) -> Dict:
        """
        Orchestrate the complete Polars DataFrame processing pipeline.

        Returns:
            Dict containing execution status, generated artifacts and metadata

        Sequential pipeline:
            1. Lazy loading of the complete data
            2. long->wide transformation
            3. Materialization and persistence

        """
        start_time = datetime.now()

        try:
            print("\n[1/3] Data loading")
            df_lazy = self.load_complete_data()

            print("\n[2/3] long->wide transformation")
            df_wide = self.pivot_long_to_wide(df_lazy)

            print("\n[3/3] Materialization and persistence")
            output_path = self.export_processed_data(df_wide)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            print(f"\nPolars processing complete in {processing_time:.2f}s")

            return {
                'status': 'success',
                'architecture': 'dataframe_lib',
                'paradigm': 'lazy_evaluation',
                'output': {
                    'dataset': output_path,
                    'metadata': f"{self.processed_dir}/processing_metadata.json"
                },
                'performance': {
                    'processing_time_seconds': processing_time
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
                'timestamp': datetime.now().isoformat()
            }


if __name__ == "__main__":
    # Without an exit status, a failure here reaches the orchestrator as
    # success: pipeline.py uses subprocess check=True, which only reads the
    # return code. That is how collection could die and the following stages
    # still run over the panel of the previous run.
    processor = DataFrameLibProcessor()
    results = processor.run_dataframe_lib_processing()
    status = results.get('status', 'failed')
    print(f"Execution: {status}")
    sys.exit(0 if status == 'success' else 1)
