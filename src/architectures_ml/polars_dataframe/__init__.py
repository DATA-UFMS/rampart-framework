"""
Arquitetura ML Polars DataFrame para comparação com Data Lake (Dask) e Data Warehouse (DuckDB).

Este módulo implementa o pipeline de ML scientificamente rigoroso para a arquitetura
Polars DataFrame, mantendo equivalência metodológica com as implementações Data Lake
e Data Warehouse para benchmarking justo em contexto de pesquisa SBBD 2026.

Paradigma:
    Lazy evaluation com expressions idiomáticas Polars, focando em otimização de
    memória via pl.scan_parquet() (LazyFrame) e transformações via expressions
    em vez de eager evaluation ou SQL.

Protocolo anti-leakage (P1-P5):
    Implementa as mesmas proteções contra data leakage que DL e DW:
    - P1: Ordenação temporal estrita (train < val < test)
    - P2: Gaps mínimos entre splits (2 anos)
    - P3: Exclusão de features derivadas do target
    - P4: Feature selection restrita ao período de treino
    - P5: Preprocessing ajustado exclusivamente no treino

Classes exportadas:
    - PolarsDataFrameArchitectureML: Implementação do pipeline ML
"""

from .setup import PolarsDataFrameArchitectureML, main

__all__ = ['PolarsDataFrameArchitectureML', 'main']
__version__ = '1.0.0'
__author__ = 'SBBD 2026 Research Team'
