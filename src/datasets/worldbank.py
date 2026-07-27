"""
World Bank dataset configuration for the benchmarking framework.

Encapsulates indicators, temporal range, geographic stratification and
walk-forward parameters for the World Bank data (LATAM,
2000-2023, country×year level).
"""

from core.dataset_config import register_dataset
from core.indicators import (
    ALL_INDICATORS,
)
from core.config import COUNTRY_STRATA


class WorldBankDatasetConfig:
    """World Bank dataset configuration (country × year, LATAM)."""

    # Identification
    name = "worldbank"
    label = "World Bank - LATAM Education Indicators"

    # Temporal
    temporal_range = (2000, 2023)
    year_column = "year"

    # Geographic entity
    entity_column = "entity_id"
    entity_name_column = "entity_name"
    stratification_column = "entity_stratum"
    strata = COUNTRY_STRATA

    # Target
    target_source_column = "target_source_rate"
    target_expected_range = (0.0, 80.0)
    min_valid_count = 500

    # Features
    # The collected catalog, not the candidate pool. excluded_columns narrows it:
    # two of those declared here are dropped by an L2 decision -- the target's
    # source column, and the enrollment rate, which is mechanically tied to
    # dropout (dropout reduces enrollment, so predicting one from the other
    # measures the same phenomenon twice). The effective pool is the difference,
    # and it appears in the selection artifact as total_features_analyzed.
    feature_columns = list(ALL_INDICATORS.values())
    excluded_columns = [
        "entity_id", "entity_name", "year", "entity_stratum",
        "synthetic_flag", "data_source", "etl_batch_id",
        "collection_timestamp", "data_completeness_score",
        "processing_method", "processed_timestamp", "partition_id",
        "target_source_rate",
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
