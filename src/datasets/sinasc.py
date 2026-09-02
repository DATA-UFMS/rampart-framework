"""
SINASC live-birth dataset configuration for the benchmarking framework.

Declarações de Nascidos Vivos (DATASUS/SINASC), final vintages, aggregated
at the municipality-of-residence × year level. The third panel (F2.1): a
replication testbed for the interference audit, on the same 5,564
municipalities as INEP but with an outcome from another domain.

Source: ftp://ftp.datasus.gov.br/dissemin/publicos/SINASC/1996_/Dados/DNRES/
License: Lei de Acesso à Informação, Brazil's freedom of information law (Lei 12.527/2011)
"""

from core.dataset_config import register_dataset


class SinascDatasetConfig:
    """SINASC live births dataset configuration (municipality × year, Brazil)."""

    # Identification
    name = "sinasc"
    label = "DATASUS SINASC - Cesarean share of deliveries (municipality × year)"

    # Temporal (DNBR{year}.dbc final vintages, 2001-2024)
    temporal_range = (2001, 2024)
    year_column = "year"

    # Geographic entity.
    #
    # The collector writes the framework's internal schema directly
    # (entity_id / entity_name / entity_stratum) and maps residence codes onto
    # the INEP-derived municipality table, so the entity universe and the
    # stratum vocabulary (state abbreviation) are the same as inep_censo.
    entity_column = "entity_id"
    entity_name_column = "entity_name"
    stratification_column = "entity_stratum"
    strata = {
        "norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
        "nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
        "sudeste": ["ES", "MG", "RJ", "SP"],
        "sul": ["PR", "RS", "SC"],
        "centro_oeste": ["DF", "GO", "MS", "MT"],
    }

    # Target. The collector stores the vaginal share of deliveries with a known
    # route (PARTO in {1, 2}); the framework derives the cesarean share as
    # 100 - target_source_rate, the same identity the other two panels use.
    target_source_column = "target_source_rate"
    target_expected_range = (0.0, 100.0)
    min_valid_count = 5000

    # Candidate features: maternal, prenatal and newborn composition shares,
    # mirroring FEATURE_COLS in collection/sinasc_collector.py. None of them is
    # a function of PARTO, so none partitions the target.
    feature_columns = [
        "share_mother_lt20", "share_mother_ge35",
        "share_escmae_low", "share_escmae_high",
        "share_prenatal_7plus", "share_prenatal_none",
        "share_preterm", "share_multiple", "share_firstbirth",
        "share_male", "share_lbw", "share_hospital",
    ]

    excluded_columns = [
        "entity_id", "entity_name", "year", "entity_stratum",
        "target_source_rate",  # target source
        # Provenance and a candidate weight for an n >= 20 sensitivity, never
        # a feature: it is the denominator every share above was built from.
        "births_total",
    ]

    # Walk-forward: 2001-2024 (24 years), identical geometry to inep_censo.
    # With gap=2 (P2), min_train=5, val=1, test=1:
    # Minimum: 5 + 2 + 1 + 2 + 1 = 11 years -> (24-11)/1 + 1 = 14 folds,
    # evaluation years 2011-2024. Target lags are the probe harness's (2, 3),
    # as on INEP: the dataset config declares none.
    walk_forward_config = {
        "min_train": 5,
        "val_len": 1,
        "test_len": 1,
        "gap": 2,
        "step": 1,
    }

    # Paths
    raw_data_subdir = "collection/raw_data"
    collector_module = "collection.sinasc_collector"


register_dataset(SinascDatasetConfig())
