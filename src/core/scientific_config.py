#!/usr/bin/env python3
"""
Módulo de Configuração Científica Centralizada para o Benchmark.

Este arquivo define constantes e configurações que devem ser IDÊNTICAS
entre as arquiteturas Data Warehouse e Data Lake para garantir um
benchmark cientificamente válido.

Parâmetros definidos aqui governam:
- Reprodutibilidade (seeds)
- Lógica de seleção de features (correlações pairwise)
- Validação temporal
- Transformações de features
"""

import numpy as np

# Seed global para garantir reprodutibilidade em todas as operações
# estocásticas (amostragem, inicialização de modelos, etc.).
RANDOM_SEED = 42

# Dicionário de configuração científica unificado.
# Usado por ambos os pipelines para garantir consistência.
SCIENTIFIC_CONFIG = {
    # Reprodutibilidade
    'random_seed': RANDOM_SEED,

    # Seleção de Features
    'collinearity_threshold': 0.8,
    'correlation_precision': 1e-3, # MAE máximo permitido entre correlações
    'min_correlation_sample_size': 5000,
    'correlation_sample_fraction': 0.1,
    # Amostragem de correlação (para alinhar DL e DW)
    'correlation_sampling': True,
    'correlation_min_sample_size': 5000,

    # Validação Temporal
    'temporal_gap_years': 2,
    # Parâmetros do gerador automático de folds
    'temporal_range_start': 2000,
    'temporal_range_end': 2023,
    'folds_min_train_years': 8,
    'folds_val_len_years': 2,
    'folds_test_len_years': 2,
    'folds_step_years': 1,
    # Opcional: limitar número de folds (None para todos)
    'folds_max': None,

    # Transformação de Features
    # Symmetric log transform: T(x) = sign(x) * ln(|x| + 1)
    # Implementações equivalentes:
    #   SQL:    SIGN(x) * LN(ABS(x) + 1)
    #   Python: np.sign(x) * np.log(np.abs(x) + 1)
    'feature_transform': 'symmetric_log',

    # Validação de Equivalência
    'target_stats_max_diff': 0.01,      # 1%
    'features_overlap_min_pct': 0.85,   # 85%
    'correlations_max_mae': 0.001,
    'fold_sizes_max_diff_pct': 0.05,    # 5%
    'float_precision_tolerance': 1e-9
    ,
    # Parâmetros estatísticos
    # Número padrão de iterações de bootstrap para ICs
    'bootstrap_iters': 3000,
    'sesoi_r2': 0.01,
    'sesoi_nrmse': 0.05,
    'sesoi_mase': 0.05
}

def setup_reproducibility():
    """
    Função auxiliar para configurar a seed em bibliotecas relevantes.
    Deve ser chamada no início de cada script de pipeline.
    """
    np.random.seed(RANDOM_SEED)
    
    try:
        import dask
        dask.config.set({'array.random.seed': RANDOM_SEED})
    except ImportError:
        pass 

    print(f"🌱 Reprodutibilidade configurada com RANDOM_SEED={RANDOM_SEED}")
