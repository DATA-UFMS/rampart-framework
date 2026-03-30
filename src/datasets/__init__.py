"""
Auto-registro de datasets.

Importar este pacote registra automaticamente todos os datasets
disponíveis no registry global (core.dataset_config).
"""

# Importar cada módulo registra o dataset via register_dataset()
from datasets import worldbank  # noqa: F401
from datasets import inep_censo  # noqa: F401
