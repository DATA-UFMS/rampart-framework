#!/usr/bin/env python3
"""
Módulo de Configuração Científica Centralizada para o Benchmark.

Este arquivo define constantes e configurações que devem ser IDÊNTICAS
entre os três paradigmas (sql_engine, task_graph, dataframe_lib) para
que o benchmark seja válido.

Parâmetros definidos aqui governam:
- Reprodutibilidade (seeds)
- Lógica de seleção de features (correlações pairwise)
- Validação temporal (P1-P2)
- Detecção de proxy (P3)
- Escopo temporal de seleção (P4)
- Transformações de features
- Espaço de busca dos modelos hierárquicos
- Parâmetros estatísticos (bootstrap, SESOI)

Este dicionário é serializado no snapshot de reprodutibilidade, então um
parâmetro definido fora daqui é um parâmetro ausente do snapshot.

P5 (escopo de preprocessing) é enforced no código dos modelos, por ser
uma propriedade de onde as estatísticas são ajustadas, e não um valor;
os testes unitários verificam esse enforcement.
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
    # Read by run_feature_selection and handed to each paradigm's pairwise
    # filter. It used to be declared here while the three filters kept their
    # own default, so changing this value did nothing.
    'collinearity_threshold': 0.8,
    # Limiar para detecção de proxy features (Kapoor & Narayanan, 2023)
    # Alinhado com max_corr da seleção de features (defense-in-depth)
    'proxy_correlation_threshold': 0.80,
    # Ceiling on how much of the target the selected features may jointly
    # explain. Catches additive identities that pairwise correlation misses.
    'identity_r2_threshold': 0.95,
    # Applies to the whole feature set, lags included, and asks a different
    # question: whether the target is reproduced to numerical precision. A
    # genuine lag never does that, so anything above 1 - this value means a
    # column labelled as lagged carries the contemporaneous value.
    'target_reproduction_tolerance': 1e-9,
    # Autoregressive features: lagged values of the target itself. Exempt from
    # the pairwise proxy check, since predicting a series from its own past is
    # the task rather than a leak, and a lag correlates with the target by
    # construction. The exemption is recorded with the measured correlation, and
    # does not extend to the joint reconstruction check.
    'autoregressive_feature_marker': '_lag_',

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
    # Recorded, not dispatched on. The transform is written out in each
    # paradigm's own idiom -- a CASE expression in SQL, a Polars expression, a
    # Dask apply -- so there is nothing here to switch. The three are checked
    # against this declaration and against each other in the test suite.
    'feature_transform': 'symmetric_log',

    # Cores made available to each engine's own execution.
    #
    # The paradigms are parallel systems: a SQL engine vectorises across threads,
    # a DataFrame library schedules work-stealing over Arrow, a task-graph
    # scheduler runs workers. Pinning these to one would not remove a confound --
    # it would measure a configuration nobody deploys and dissolve the premise of
    # the comparison, since a scheduler with a single worker is not a scheduler.
    #
    # The criterion is an equal hardware budget: every paradigm gets the same
    # number of cores and is free to exploit them as its design dictates. How
    # well each does so is a property of the paradigm, and a finding rather than
    # noise. Left unset -- as it was -- each engine sized itself from the host's
    # core count, so the comparison silently depended on the machine and no
    # artifact recorded how many cores any engine had.
    #
    # Declared as an integer rather than derived from the host, so the
    # configuration is reproducible elsewhere. Validated against the available
    # cores: oversubscription would make latency reflect scheduling contention.
    #
    # Every latency result, including the scale crossover, is conditional on this
    # value.
    'engine_threads': 8,

    # Threads made available to the numerical libraries beneath scikit-learn.
    #
    # Pinned to one, and this is a measurement decision rather than a
    # performance one. Left unset, OpenBLAS sizes its pool from the available
    # cores -- twelve on the development machine -- so a stage's latency depends
    # on how many cores it happens to get. That is not merely irreproducible
    # across machines: the paradigms do not contend for cores equally, since the
    # task-graph scheduler runs workers alongside the fit, so part of a measured
    # difference would be thread contention rather than the paradigm.
    #
    # Raising this reintroduces that confound. It must be set before NumPy is
    # imported, which is why the pipeline exports it to each subprocess.
    'blas_threads': 1,

    # Search space of the hierarchical stage.
    #
    # Defined here rather than inside each paradigm for two reasons. Three copies
    # can drift apart, and paradigms searching different spaces are not fitting
    # the same model -- which is the premise the equivalence check rests on. And
    # the reproducibility snapshot records this dictionary, so a search space
    # living in the paradigms is a search space absent from the snapshot.
    'hierarchical_model': {
        # RidgeCV alphas as logspace(start, stop, count).
        'ridge_alpha_log10_start': -1,
        'ridge_alpha_log10_stop': 3,
        'ridge_alpha_count': 20,
        # Inner folds for alpha selection. RidgeCV rejects fewer than two,
        # so a panel with a single residual row falls back to its
        # leave-one-out generalised cross-validation.
        'ridge_cv_folds': 3,
        # Shrinkage applied to the residual component.
        'residual_shrinkage_grid': (0.6, 0.8, 1.0),
        # Random forest over entity effects, tuned on the validation window.
        'rf_max_depth_grid': (5, 6, 7),
        'rf_min_samples_leaf_grid': (5, 8, 12),
        'rf_n_estimators': 200,
        'rf_min_samples_split': 15,
        'rf_max_features': 'sqrt',
        # Single-threaded: parallel tree building would make latency depend on
        # core availability rather than on the paradigm under measurement.
        'rf_n_jobs': 1,
    },

    # Cross-paradigm equivalence is verified as bitwise identity of the
    # predicted vectors, not as agreement within a tolerance. Four tolerances
    # once lived here -- 85% feature overlap, 1% on target statistics, MAE
    # 0.001 on correlations, 5% on fold sizes -- and nothing read any of them.
    # They described a weaker claim than the one the framework makes and
    # enforces, and they were recorded in the published config snapshot, where
    # a reader would reasonably take them for the operative criterion.
    'float_precision_tolerance': 1e-9,
    # Parâmetros estatísticos
    #
    # Bootstrap resamples. The latency and effect-size intervals are percentile
    # intervals; the equivalence estimate uses BCa and falls back to percentile.
    # All of them read quantiles of the bootstrap distribution, so all inherit
    # the sensitivity Hesterberg (2015) quantifies: r >= 15000 for Monte Carlo
    # variability in percentile endpoints to stay within 10% of the exhaustive
    # value, with 10^4 as his figure for routine use. At n=10 folds the whole
    # family of comparisons runs in a few seconds, so the stricter requirement
    # costs nothing.
    #
    # Single source of truth: the statistical modules read this value and must
    # not carry a default of their own, or the reported resample count and the
    # executed one can drift apart.
    #
    #   Hesterberg, T. C. (2015). What Teachers Should Know About the
    #     Bootstrap: Resampling in the Undergraduate Statistics Curriculum.
    #     The American Statistician, 69(4), 371-386.
    'bootstrap_iters': 15000,

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
