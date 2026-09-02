"""
Dataset auto-registration.

Importing this package automatically registers every available dataset
in the global registry (core.dataset_config).
"""

# Importing each module registers the dataset via register_dataset()
from datasets import worldbank  # noqa: F401
from datasets import inep_censo  # noqa: F401
from datasets import sinasc  # noqa: F401
