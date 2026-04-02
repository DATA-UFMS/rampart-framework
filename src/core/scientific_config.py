#!/usr/bin/env python3
"""
Módulo de Configuração Científica Centralizada para o Benchmark.

Este arquivo define constantes e configurações que devem ser IDÊNTICAS
entre as arquiteturas Data Warehouse e Data Lake para garantir um
benchmark válido.

Parâmetros definidos aqui governam:
- Reprodutibilidade (seeds)
- Lógica de seleção de features (correlações pairwise)
- Validação temporal (P1-P2)
- Detecção de proxy (P3)
- Escopo temporal de seleção (P4)
- Transformações de features

Parâmetros de P5 (escopo de preprocessing) e HPO são enforced
diretamente no código dos modelos (ver BaseArchitectureML e
hierarchical_model.py) e validados por testes unitários.
"""

import random

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
    'correlation_sample_fraction': 0.1,
    # Amostragem de correlação (reduz custo computacional mantendo equivalência)
    'correlation_sampling': True,
    'correlation_min_sample_size': 5000,
    # Limiar para detecção de proxy features (Kapoor & Narayanan, 2023)
    # Alinhado com max_corr da seleção de features (defense-in-depth)
    'proxy_correlation_threshold': 0.80,

    # Validação Temporal
    'temporal_gap_years': 2,
    'embargo_years': 0,  # Embargo adicional (López de Prado 2018); 0 = desativado
    # Parâmetros do gerador automático de folds
    #
    # Estes parâmetros produzem n=9 folds walk-forward. Esse n é o
    # máximo alcançável sem violar as restrições temporais (P1-P2):
    #   n = floor((end - start - min_train - val - test - 2*gap + 1) / step)
    #   n = floor((2023 - 2000 - 8 - 2 - 2 - 4 + 1) / 1) = 8... +1 = 9
    #
    # Aumentar n exigiria reduzir gap (comprometendo P2), reduzir
    # min_train (comprometendo estabilidade do treino) ou usar folds
    # sobrepostos (comprometendo independência). A decisão de manter
    # n=9 prioriza integridade anti-leakage sobre poder estatístico,
    # conforme recomendado para dados temporais (Cerqueira et al. 2020;
    # Roberts et al. 2017).
    #
    # Implicação: o Wilcoxon pareado com n=9 tem poder ~30% para
    # efeitos médios (d~0.5). Por isso o método primário de decisão
    # é bootstrap CI (que não depende de premissas assintóticas), e
    # Wilcoxon + Hodges-Lehmann são complementos de robustez. Um
    # resultado "inconclusivo" é o desfecho esperado quando o efeito
    # real é pequeno e n é limitado — não indica falha metodológica,
    # mas reflete a precisão disponível (Lakens et al. 2018).
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
    'float_precision_tolerance': 1e-9,
    # Parâmetros estatísticos
    # Número padrão de iterações de bootstrap para ICs
    'bootstrap_iters': 3000,

    # SESOI (Smallest Effect Size Of Interest) — Lakens et al. (2018)
    #
    # Definidos a priori usando abordagem híbrida:
    #   - distribution-based para R² (referência em Cohen 1988)
    #   - anchor-based para MASE/WAPE (resolução prática de decisão)
    #
    # sesoi_r2 = 0.01: metade do efeito pequeno de Cohen (1988, f²=0.02,
    #   equivalente a R²~0.02). Deliberadamente conservador — exigimos
    #   equivalência dentro de uma margem menor que o convencionalmente
    #   considerado "pequeno". Se |delta_R²| < 0.01, a diferença preditiva
    #   entre arquiteturas é irrelevante para qualquer aplicação prática.
    #
    # sesoi_mase = 0.05: MASE é relativo ao forecast naïve (Hyndman &
    #   Koehler 2006); delta_MASE de 0.05 significa que ambas arquiteturas
    #   estão dentro de 5% uma da outra em relação ao baseline naïve.
    #   Abaixo da resolução em que um pesquisador alteraria sua escolha
    #   de paradigma de dados.
    #
    # sesoi_wape = 0.05: 5 pontos percentuais de erro ponderado. Margem
    #   dentro da qual a diferença não alteraria uma decisão prática
    #   de adoção de arquitetura em contexto educacional.
    #
    # Referências:
    #   Lakens, D., Scheel, A. M., & Isager, P. M. (2018). Equivalence
    #     Testing for Psychological Research: A Tutorial. Advances in
    #     Methods and Practices in Psychological Science, 1(2), 259-269.
    #   Cohen, J. (1988). Statistical Power Analysis for the Behavioral
    #     Sciences (2nd ed.). Lawrence Erlbaum Associates.
    #   Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures
    #     of forecast accuracy. International Journal of Forecasting,
    #     22(4), 679-688.
    'sesoi_r2': 0.01,
    'sesoi_mase': 0.05,
    'sesoi_wape': 0.05
}

def setup_reproducibility():
    """
    Função auxiliar para configurar a seed em bibliotecas relevantes.
    Deve ser chamada no início de cada script de pipeline.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    try:
        import dask
        # Dask não possui config nativa de seed global.
        # A reprodutibilidade é garantida pela seed do numpy.
    except ImportError:
        pass

    print(f"Seed={RANDOM_SEED}")
