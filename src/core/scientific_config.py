#!/usr/bin/env python3
"""
Centralised Scientific Configuration Module for the Benchmark.

This file defines constants and settings that must be IDENTICAL
across the three paradigms (sql_engine, task_graph, dataframe_lib) for
the benchmark to be valid.

Parameters defined here govern:
- Reproducibility (seeds)
- Feature selection logic (pairwise correlations)
- Temporal validation (P1-P2)
- Proxy detection (P3)
- Temporal scope of selection (P4)
- Feature transforms
- Search space of the hierarchical models
- Statistical parameters (bootstrap, SESOI)

This dictionary is serialised into the reproducibility snapshot, so a
parameter defined outside it is a parameter missing from the snapshot.

P5 (preprocessing scope) is enforced in the model code, since it is
a property of where the statistics are fitted, and not a value;
the unit tests verify that enforcement.
"""

import random

import numpy as np

# Global seed to guarantee reproducibility across all stochastic
# operations (sampling, model initialisation, etc.).
RANDOM_SEED = 42

# Unified scientific configuration dictionary.
# Used by both pipelines to guarantee consistency.
SCIENTIFIC_CONFIG = {
    # Reproducibility
    'random_seed': RANDOM_SEED,

    # Feature Selection
    # Read by run_feature_selection and handed to each paradigm's pairwise
    # filter. It used to be declared here while the three filters kept their
    # own default, so changing this value did nothing.
    'collinearity_threshold': 0.8,

    # Floor on marginal association for a candidate to enter selection,
    # in absolute value. The sign does not enter: neither RidgeCV nor
    # RandomForest cares about the direction of a marginal association, and in
    # this domain most protective factors (GDP, completion, enrolment) associate
    # negatively with dropout. The signed comparison discarded all of them.
    #
    # 0.15 sits above the conventional limit of negligible association
    # (Cohen, 1988: 0.10 small, 0.30 medium, 0.50 large) and below medium.
    'feature_selection_min_abs_correlation': 0.15,

    # Alternative floor, used only when the strict one does not gather the
    # minimum number of features below. It is exactly Cohen's boundary for
    # "small": below that the association is conventionally negligible, and
    # admitting the feature would be trading noise for count. It replaces a
    # multiplier of 0.67 that had no derivation at all.
    'feature_selection_relaxed_min_abs_correlation': 0.10,

    # How many features make the model non-degenerate. It is a pragmatic floor,
    # not a statistical result, and it is written as such: failing to reach it
    # does not halt the run -- what halts it is reaching zero.
    'feature_selection_min_features': 5,
    # |r| above which a feature is suspected of being the target under another
    # name (Kapoor & Narayanan, 2023). One question, one number: selection uses
    # it as a ceiling over the training window and the audit applies it over the
    # whole panel. They were two values equal by coincidence, and the comment
    # said they were aligned without anything requiring it.
    'proxy_correlation_threshold': 0.80,
    # Ceiling on how much of the target the selected features may jointly
    # explain. Catches additive identities that pairwise correlation misses.
    'identity_r2_threshold': 0.95,
    # Applies to the whole feature set, lags included, and asks a different
    # question: whether the target is reproduced to numerical precision. A
    # genuine lag never does that, so anything above 1 - this value means a
    # column labelled as lagged carries the contemporaneous value.
    'target_reproduction_tolerance': 1e-9,
    # Autoregressive features -- lagged values of the target itself -- are
    # exempt from the pairwise proxy check, since predicting a series from its
    # own past is the task rather than a leak, and a lag correlates with the
    # target by construction. The exemption is recorded with the measured
    # correlation and does not extend to the joint reconstruction check.
    #
    # No marker is configured for them: they are passed to the audit by name,
    # derived from BaseArchitectureML.TARGET_STEM and TARGET_LAG_ORDERS. A
    # substring rule here would have silently excused any feature whose name
    # happened to contain it.

    # Temporal Validation
    'temporal_gap_years': 2,
    'embargo_years': 0,  # Additional embargo (López de Prado 2018); 0 = disabled
    # Parameters of the automatic fold generator
    #
    # These parameters produce n=9 walk-forward folds. That n is the
    # maximum reachable without violating the temporal constraints (P1-P2):
    # The count is of test start points, not of intervals, so it is
    # the size of the closed range [test_start_min, test_start_max]:
    #
    #   test_start_min = start + min_train + val + 2*gap = 2000+8+2+4 = 2014
    #   test_start_max = end - test + 1                  = 2023-2+1  = 2022
    #   n = floor((test_start_max - test_start_min) / step) + 1 = 9
    #
    # The "+1" is not an ad hoc adjustment: a closed range with equal endpoints
    # has one element, not zero. The previous form subtracted the two endpoints
    # and added one afterwards, which gave the same number by accident of
    # arrangement.
    #
    # Raising n would require reducing the gap (compromising P2), reducing
    # min_train (compromising training stability) or using overlapping
    # folds (compromising independence). The decision to keep
    # n=9 prioritises anti-leakage integrity over statistical power,
    # as recommended for temporal data (Cerqueira et al. 2020;
    # Roberts et al. 2017).
    #
    # Implication: the paired Wilcoxon with n=9 has ~30% power for
    # medium effects (d~0.5). That is why the primary decision method
    # is the bootstrap CI (which does not depend on asymptotic premises), and
    # Wilcoxon + Hodges-Lehmann are robustness complements. An
    # "inconclusive" result is the expected outcome when the real
    # effect is small and n is limited — it does not indicate a methodological
    # failure, but reflects the available precision (Lakens et al. 2018).
    'temporal_range_start': 2000,
    'temporal_range_end': 2023,
    'folds_min_train_years': 8,
    'folds_val_len_years': 2,
    'folds_test_len_years': 2,
    'folds_step_years': 1,
    # Optional: cap the number of folds (None for all)
    'folds_max': None,

    # Feature Transform
    # Symmetric log transform: T(x) = sign(x) * ln(|x| + 1)
    # Equivalent implementations:
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
    # Statistical parameters
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
    # Defined a priori using a hybrid approach:
    #   - distribution-based for R² (reference in Cohen 1988)
    #   - anchor-based for MASE/WAPE (practical decision resolution)
    #
    # sesoi_r2 = 0.01: half of Cohen's small effect (1988, f²=0.02,
    #   equivalent to R²~0.02). Deliberately conservative — we require
    #   equivalence within a margin smaller than what is conventionally
    #   considered "small". If |delta_R²| < 0.01, the predictive difference
    #   between architectures is irrelevant for any practical application.
    #
    # sesoi_mase = 0.05: MASE is relative to the naïve forecast (Hyndman &
    #   Koehler 2006); a delta_MASE of 0.05 means both architectures
    #   are within 5% of each other relative to the naïve baseline.
    #   Below the resolution at which a researcher would change their choice
    #   of data paradigm.
    #
    # sesoi_wape = 0.05: 5 percentage points of weighted error. A margin
    #   within which the difference would not change a practical decision
    #   to adopt an architecture in an educational context.
    #
    # References:
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
    Helper function to configure the seed in the relevant libraries.
    Must be called at the start of each pipeline script.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    try:
        import dask
        # Dask has no native global seed config.
        # Reproducibility is guaranteed by the numpy seed.
    except ImportError:
        pass

    print(f"Seed={RANDOM_SEED}")
