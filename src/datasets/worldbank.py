"""
Configuração do dataset World Bank para o framework de benchmarking.

Encapsula indicadores, range temporal, estratificação geográfica e
parâmetros de walk-forward para os dados do Banco Mundial (LATAM,
2000-2023, nível país×ano).
"""

from core.dataset_config import register_dataset
from core.indicators import (
    ALL_INDICATORS,
)
from core.config import COUNTRY_STRATA


class WorldBankDatasetConfig:
    """Configuração do dataset World Bank (país × ano, LATAM)."""

    # Identificação
    name = "worldbank"
    label = "World Bank - LATAM Education Indicators"

    # Temporal
    temporal_range = (2000, 2023)
    year_column = "year"

    # Entidade geográfica
    entity_column = "country_code"
    entity_name_column = "country_name"
    stratification_column = "country_stratum"
    strata = COUNTRY_STRATA

    # Target
    target_source_column = "lower_secondary_completion_rate"
    target_expected_range = (0.0, 80.0)
    min_valid_count = 500

    # Features
    # O catálogo coletado, não o pool de candidatas. excluded_columns o
    # estreita: duas das declaradas aqui saem por decisão L2 -- a coluna-fonte
    # do alvo, e a taxa de matrícula, que é mecanicamente ligada à evasão
    # (evasão reduz matrícula, então prever uma pela outra é medir o mesmo
    # fenômeno duas vezes). O pool efetivo é a diferença, e sai no artefato de
    # seleção como total_features_analyzed.
    feature_columns = list(ALL_INDICATORS.values())
    excluded_columns = [
        "country_code", "country_name", "year", "country_stratum",
        "synthetic_flag", "data_source", "etl_batch_id",
        "collection_timestamp", "data_completeness_score",
        "processing_method", "processed_timestamp", "partition_id",
        "lower_secondary_completion_rate",
        "enrollment_rate_secondary_net",
    ]

    # Walk-forward
    walk_forward_config = {
        "min_train": 8,
        "val_len": 2,
        "test_len": 2,
        "gap": 2,
        "step": 1,
    }

    # Paths
    raw_data_subdir = "collection/raw_data"
    collector_module = "collection.raw_data_collector"


register_dataset(WorldBankDatasetConfig())
