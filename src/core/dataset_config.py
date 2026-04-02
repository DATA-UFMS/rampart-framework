"""
Protocolo e registro de configurações de dataset.

Define o contrato DatasetConfig que encapsula tudo que varia entre
datasets (World Bank, INEP Censo Escolar, etc.), permitindo que o
framework opere de forma dataset-agnostica.
"""

from typing import Any, Dict, List, Tuple, runtime_checkable, Protocol


@runtime_checkable
class DatasetConfig(Protocol):
    """Protocolo que todo dataset deve satisfazer."""

    # Identificação
    name: str
    label: str

    # Temporal
    temporal_range: Tuple[int, int]
    year_column: str

    # Entidade geográfica
    entity_column: str
    entity_name_column: str
    stratification_column: str
    strata: Dict[str, List[str]]

    # Target
    target_source_column: str
    target_expected_range: Tuple[float, float]
    min_valid_count: int

    # Features
    feature_columns: List[str]
    excluded_columns: List[str]

    # Walk-forward
    walk_forward_config: Dict[str, int]

    # Paths
    raw_data_subdir: str
    collector_module: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_DATASET_REGISTRY: Dict[str, Any] = {}


def register_dataset(config: Any) -> None:
    """Registra um DatasetConfig no registry global."""
    if not isinstance(config, DatasetConfig):
        raise TypeError(
            f"{type(config).__name__} não satisfaz o protocolo DatasetConfig"
        )
    _DATASET_REGISTRY[config.name] = config


def get_dataset(name: str) -> Any:
    """Retorna DatasetConfig pelo nome. Levanta KeyError se não encontrado."""
    if name not in _DATASET_REGISTRY:
        available = list(_DATASET_REGISTRY.keys())
        raise KeyError(
            f"Dataset '{name}' não registrado. Disponíveis: {available}"
        )
    return _DATASET_REGISTRY[name]


def list_datasets() -> List[str]:
    """Retorna lista de nomes de datasets registrados."""
    return list(_DATASET_REGISTRY.keys())
