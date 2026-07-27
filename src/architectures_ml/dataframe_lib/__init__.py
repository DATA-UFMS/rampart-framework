"""
Polars DataFrame ML architecture, for comparison with Data Lake (Dask) and Data Warehouse (DuckDB).

This module implements the ML pipeline for the Polars DataFrame architecture,
keeping methodological equivalence with the Data Lake and Data Warehouse
implementations so that the benchmarking is fair.

Paradigm:
    Lazy evaluation with idiomatic Polars expressions, focusing on memory
    optimization via pl.scan_parquet() (LazyFrame) and on transformations via
    expressions instead of eager evaluation or SQL.

Anti-leakage protocol (P1-P5):
    Implements the same protections against data leakage as DL and DW:
    - P1: Strict temporal ordering (train < val < test)
    - P2: Minimum gaps between splits (2 years)
    - P3: Exclusion of target-derived features
    - P4: Feature selection restricted to the training period
    - P5: Preprocessing fitted exclusively on the training data

Exported classes:
    - DataFrameLibArchitectureML: ML pipeline implementation
"""

__all__ = ['DataFrameLibArchitectureML', 'main']
__version__ = '1.0.0'


def __getattr__(name):
    """Lazy import, to keep backward compatibility without loading eagerly."""
    if name in ('DataFrameLibArchitectureML', 'main'):
        from .setup import DataFrameLibArchitectureML, main
        return DataFrameLibArchitectureML if name == 'DataFrameLibArchitectureML' else main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
