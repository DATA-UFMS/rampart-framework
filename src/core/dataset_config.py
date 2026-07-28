"""
Protocol and registry of dataset configurations.

Defines the DatasetConfig contract that encapsulates everything that varies
between datasets (World Bank, INEP Censo Escolar, etc.), allowing the
framework to operate in a dataset-agnostic way.
"""

from typing import (Any, Dict, Iterable, List, Protocol, Tuple,
                    runtime_checkable)


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


def modelling_features(config: Any, available: Iterable[str]) -> List[str]:
    """The columns a model may fit on: declared features, minus the exclusions.

    Exists because `feature_columns` is not the answer on its own, and reading
    it as though it were has already cost one wrong result. For World Bank its
    first entry is `target_source_rate`, and the target is `100 - that`; a probe
    that skipped `excluded_columns` handed every model the answer, every model
    scored R^2 = 1, and the probe reported them all as immune to contamination.
    They were not immune -- there was simply no headroom left to inflate.

    The paradigms have always applied both lists. Validation scripts stand
    outside the paradigms and were each re-deriving the policy by hand, which is
    the arrangement where one copy quietly disagrees with the others. This is
    the one copy.
    """
    excluded = set(config.excluded_columns)
    available = list(available)
    features = [column for column in config.feature_columns
                if column in available and column not in excluded]

    source = getattr(config, 'target_source_column', None)
    if source is not None and source in features:
        raise ValueError(
            f"{source!r} is the column the target is derived from and it "
            f"survived the exclusions for dataset {config.name!r}. Any model "
            f"fitted on this list would be reading its own answer.")
    return features
