"""
INEP Censo Escolar dataset configuration for the benchmarking framework.

Microdata from the Censo Escolar da Educação Básica (INEP/MEC), filtered
to Ensino Médio (upper secondary), aggregated at the municipality × year level.

Source: https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar
License: Lei de Acesso à Informação, Brazil's freedom of information law (Lei 12.527/2011)
"""

from core.dataset_config import register_dataset


class InepCensoDatasetConfig:
    """INEP Censo Escolar dataset configuration (municipality × year, Brazil)."""

    # Identification
    name = "inep_censo"
    label = "INEP Censo Escolar - Ensino Médio (municipality × year)"

    # Temporal (Indicadores Educacionais: XLS 2007-2011, XLSX 2012-2024)
    temporal_range = (2007, 2024)
    year_column = "year"

    # Geographic entity.
    #
    # The collector maps each municipality onto the framework's internal schema
    # (country_code / country_name / country_stratum), so the pipeline needs no
    # dataset-specific handling. The stratum carries the state abbreviation.
    entity_column = "country_code"
    entity_name_column = "country_name"
    stratification_column = "country_stratum"
    strata = {
        "norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
        "nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
        "sudeste": ["ES", "MG", "RJ", "SP"],
        "sul": ["PR", "RS", "SC"],
        "centro_oeste": ["DF", "GO", "MS", "MT"],
    }

    # Target. The collector inverts the upper-secondary abandonment rate into a
    # completion rate, from which the framework derives dropout_rate.
    target_source_column = "lower_secondary_completion_rate"
    target_expected_range = (0.0, 100.0)
    min_valid_count = 5000

    # Candidate features: INEP's lower-secondary rendimento rates, mirroring
    # FEATURE_COLS in collection/inep_collector.py. Upper-secondary rates are
    # absent by design -- they partition the target exactly.
    feature_columns = [
        "aprov_ef", "aprov_ef_ai", "aprov_ef_af",
        "reprov_ef", "reprov_ef_ai", "reprov_ef_af",
        "abandono_ef", "abandono_ef_ai", "abandono_ef_af",
    ]

    excluded_columns = [
        "country_code", "country_name", "year", "country_stratum",
        "lower_secondary_completion_rate",  # target source
        "data_completeness_score",
    ]

    # Walk-forward: 2007-2024 (18 years)
    # With gap=2 (P2), min_train=5, val=1, test=1:
    # Minimum: 5 + 2 + 1 + 2 + 1 = 11 years → (18-11)/1 + 1 = 8 folds
    walk_forward_config = {
        "min_train": 5,
        "val_len": 1,
        "test_len": 1,
        "gap": 2,
        "step": 1,
    }

    # Paths
    raw_data_subdir = "collection/inep_raw"
    collector_module = "collection.inep_collector"

    # ---------------------------------------------------------------
    # INEP-specific configuration (not in the Protocol, but useful)
    # ---------------------------------------------------------------

    # Microdata download URL
    download_url_template = (
        "https://download.inep.gov.br/dados_abertos/"
        "microdados_censo_escolar_{year}.zip"
    )

    # Format of the raw CSVs
    csv_separator = "|"
    csv_encoding = "iso-8859-1"

    # Filter: Ensino Médio only (TP_ETAPA_ENSINO codes 25-38)
    etapa_ensino_filter = list(range(25, 39))

    # Columns to extract from the MATRICULA table
    matricula_columns = [
        "NU_ANO_CENSO", "CO_MUNICIPIO", "NO_MUNICIPIO", "CO_UF",
        "TP_SEXO", "TP_COR_RACA", "NU_IDADE", "TP_ZONA_RESIDENCIAL",
        "TP_ETAPA_ENSINO", "TP_MEDIACAO_DIDATICO_PEDAGO",
        "IN_TRANSPORTE_PUBLICO",
    ]

    # Columns to extract from the ESCOLA table
    escola_columns = [
        "CO_ENTIDADE", "CO_MUNICIPIO", "NU_ANO_CENSO",
        "IN_INTERNET", "IN_LABORATORIO_INFORMATICA",
        "IN_LABORATORIO_CIENCIAS", "IN_BIBLIOTECA",
        "IN_QUADRA_ESPORTES_COBERTA", "IN_QUADRA_ESPORTES_DESCOBERTA",
        "IN_AGUA_POTAVEL", "IN_ESGOTO_REDE_PUBLICA",
        "IN_ENERGIA_REDE_PUBLICA",
    ]

    # Situação do Aluno columns (2nd collection round, rendimento/movimento)
    situacao_columns = [
        "NU_ANO_CENSO", "CO_MUNICIPIO", "TP_SITUACAO_ALUNO",
    ]

    # Status codes (TP_SITUACAO_ALUNO or an equivalent variable)
    # 1=Passed, 2=Failed, 3=Completed,
    # 4=Transferred, 5=Stopped attending (DROPOUT), 6=Deceased
    situacao_abandono_codes = [5]
    situacao_ativa_codes = [1, 2, 3, 5]  # excludes transferred and deceased

    # Crossover experiment: municipality subsets for scale
    crossover_municipality_sizes = [100, 500, 1000, 5570]


register_dataset(InepCensoDatasetConfig())
