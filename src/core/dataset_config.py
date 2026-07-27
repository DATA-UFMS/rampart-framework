"""
Protocol and registry of dataset configurations.

Defines the DatasetConfig contract that encapsulates everything that varies
between datasets (World Bank, INEP Censo Escolar, etc.), allowing the
framework to operate in a dataset-agnostic way.
"""

from typing import Any, Dict, List, Tuple, runtime_checkable, Protocol


@runtime_checkable
class DatasetConfig(Protocol):
    """Protocol that every dataset must satisfy."""

    # Identification
    name: str
    label: str

    # Temporal
    temporal_range: Tuple[int, int]
    year_column: str

    # Geographic entity
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
    """Register a DatasetConfig in the global registry."""
    if not isinstance(config, DatasetConfig):
        raise TypeError(
            f"{type(config).__name__} does not satisfy the DatasetConfig protocol"
        )
    _DATASET_REGISTRY[config.name] = config


def get_dataset(name: str) -> Any:
    """Return a DatasetConfig by name. Raises KeyError if not found."""
    if name not in _DATASET_REGISTRY:
        available = list(_DATASET_REGISTRY.keys())
        raise KeyError(
            f"Dataset '{name}' not registered. Available: {available}"
        )
    return _DATASET_REGISTRY[name]


def list_datasets() -> List[str]:
    """Return a list of registered dataset names."""
    return list(_DATASET_REGISTRY.keys())
