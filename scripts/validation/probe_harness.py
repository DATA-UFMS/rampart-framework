#!/usr/bin/env python3
"""Shared scaffolding for the World Bank probes.

Loading the panel, deriving the folds, imputing from the training window and
injecting class III contamination are the same four steps in every probe that
asks a question about this panel outside the pipeline. They were copied into
three scripts, and the copy in the first one is where the wrong feature list
came from -- it read `feature_columns` directly and handed every model the
column the target is derived from.

One copy, so the next probe inherits the fix instead of the bug.

These are probes, not the pipeline: they run a plain sklearn fit on a design
matrix rather than the paradigms' engines, and the injection happens here rather
than through `core.injection`. What they can conclude is bounded by that, and
each script says so for itself.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from datasets.worldbank import *  # noqa: F401,F403,E402  -- registers the dataset
from core.dataset_config import get_dataset, modelling_features
from core.scientific_config import RANDOM_SEED  # noqa: E402

#: Where the raw panels live. Absolute paths under a home directory were fine
#: while everything ran on one laptop and are the first thing that breaks
#: anywhere else: the Dockerfile copies src, tests and scripts, and the panels
#: are in neither. `RAMPART_PANEL_DIR` overrides the default, and a missing file
#: says which file and how to point at it rather than raising a bare
#: FileNotFoundError from inside pandas.
#: parents[3] is the workspace holding both the repository and the collections
#: side by side; parents[2] is the repository itself and is where a first attempt
#: pointed, which resolved to a directory that has never existed.
PANEL_ROOT = Path(os.environ.get(
    'RAMPART_PANEL_DIR',
    Path(__file__).resolve().parents[3] / 'dw-vs-dl-dropout-prediction-latam'))

PANEL = str(PANEL_ROOT / 'azure_results_v7_wb' / 'collection' / 'raw_data'
            / 'complete_data.parquet')

#: Per-dataset loading, because the two raw collections differ in exactly three
#: ways and nothing else: where they live, what their entity column was called
#: before the schema was neutralised, and how the target is derived.
#:
#: The INEP target comes from an algebraic identity the collector documents --
#: approval, failure and dropout sum to 100 for each stage -- so dropout in upper
#: secondary is 100 minus the other two. Those two columns are consequently
#: absent from the registry's feature list, which `modelling_features` enforces:
#: they determine the target exactly, the same trap `target_source_rate` is on the
#: World Bank side.
PANELS = {
    'worldbank': {
        'path': PANEL,
        'rename': {'country_code': 'entity_id',
                   'lower_secondary_completion_rate': 'target_source_rate'},
        'target': lambda df: 100.0 - df['target_source_rate'],
    },
    'inep_censo': {
        'path': str(PANEL_ROOT / 'azure_results_v7_inep' / 'collection'
                    / 'inep_raw' / 'complete_data.parquet'),
        'rename': {'country_code': 'entity_id', 'country_name': 'entity_name',
                   'country_stratum': 'entity_stratum'},
        'target': lambda df: 100.0 - df['aprov_em'] - df['reprov_em'],
    },
    # The same source, recollected after the cross-sectional imputation tiers were
    # removed, and it exists because of what the axis reads on each panel: the kNN
    # closed form is matched to 0.0108 on the World Bank and to about 0.10 on INEP
    # at either sample size. The panel that calibrates well is the one whose target
    # was 33.5% imputed -- 193 of 768 cells from the mean of other countries in the
    # same year -- and a group mean pulls neighbours' targets together, which is the
    # condition under which kNN absorption should match (2k-1)/k^2. So the good
    # calibration may be a property of synthesised targets rather than of the
    # instrument, and this panel is how that gets decided instead of argued.
    #
    # It lives beside the other collections rather than under outputs/, both
    # because that is what it is and because a parquet under
    # outputs/worldbank/collection/ is input to the processors: leaving it there
    # made three exit-status tests pass for the wrong reason, since they assert a
    # processor fails with no input and were finding this panel instead.
    #
    # Already neutralised at the source, so only the entity column is renamed.
    # 768 rows become 511: every row whose target came from imputation is gone,
    # which is exactly the 64 temporal plus 193 geographic the old log records.
    # Verified against the old panel -- the indicators it never imputed agree to
    # 99.6-100%, so the API is stable and the disagreement elsewhere is the
    # imputation. One exception to declare: gdp_per_capita_constant_2015 agrees on
    # 80.2% while never having been imputed, which is the World Bank rebasing a
    # constant-price series between April and July 2026.
    'worldbank_clean': {
        'path': str(PANEL_ROOT / 'worldbank_clean' / 'collection' / 'raw_data'
                    / 'complete_data.parquet'),
        'rename': {'country_code': 'entity_id'},
        'target': lambda df: 100.0 - df['target_source_rate'],
        # Same indicators, same target identity, same exclusions -- the panel
        # differs in which cells are observed, not in what the dataset means, so
        # borrowing the registered config is the accurate thing rather than a
        # shortcut. Registering a second dataset would also make it appear as a
        # pipeline target, which it is not.
        'config': 'worldbank',
    },
    # The clean test of the imputation hypothesis, and the reason the first one was
    # not a test. Comparing the original panel against the recollected one changes
    # three things at once: the recollected panel loses Brazil and Haiti entirely,
    # stops being balanced (5 to 24 years per entity against 24 for all), and only
    # then drops the imputation. So the contrast could not be attributed.
    #
    # This is the original panel restricted to the rows the recollected one keeps --
    # the 511 with an observed target, on which the two panels agree on the target
    # exactly. Same rows, same target, same folds. The one thing that differs is
    # where the FEATURE values came from: upstream geographic and global means with
    # calibrated noise here, against missing-and-filled-from-the-training-window in
    # `worldbank_clean`. One variable.
    'worldbank_imputed_features': {
        'path': PANEL,
        'rename': {'country_code': 'entity_id',
                   'lower_secondary_completion_rate': 'target_source_rate'},
        'target': lambda df: 100.0 - df['target_source_rate'],
        'config': 'worldbank',
        'filter': lambda df: _rows_shared_with(df, 'worldbank_clean'),
    },
    # The 52 clipped targets, removed. Range validation runs after the row drop and
    # clips observed values as high as 124.80 down to exactly 100.0, so the target --
    # 100 minus that rate -- carries a 52-row point mass at exactly 0, on the boundary
    # of the very quantity leakage severity is read against. That is 10.2% of the
    # panel, it is present in the original panel too, and it is declared nowhere. This
    # arm asks whether any finding depends on it.
    'worldbank_clean_unclipped': {
        'path': str(PANEL_ROOT / 'worldbank_clean' / 'collection' / 'raw_data'
                    / 'complete_data.parquet'),
        'rename': {'country_code': 'entity_id'},
        'target': lambda df: 100.0 - df['target_source_rate'],
        'config': 'worldbank',
        'filter': lambda df: df[df['target_source_rate'] != 100.0],
    },
}


def _rows_shared_with(df, other):
    """Rows of `df` whose (entity, year) key appears in another registered panel.

    Reads the other panel's parquet directly rather than calling `panel()`, because
    `panel()` would build its target and lags for a frame that is only being used as
    a key set, and because a filter that recursed into the loader would recurse
    through this function again.
    """
    spec = PANELS[other]
    keys = (pd.read_parquet(spec['path'])
            .rename(columns=spec['rename'])[['entity_id', 'year']])
    return df.merge(keys, on=['entity_id', 'year'], how='inner')

#: Roth's own duplication rates, so the doses are not ours to choose.
DOSES = (0.05, 0.10, 0.30)

def ladder_roster():
    """The models a probe runs: the LADDER, unless RAMPART_MODELS names a
    subset (comma list of rung names; 'ladder_mlp' adds the neural rung,
    'ladder_xgboost'/'ladder_lightgbm' the F1.3 boosting rungs).
    Default unchanged, so every existing run and test keeps its roster."""
    import os
    from core.models.ladder import (LADDER, lightgbm_rung, neural_rung,
                                    xgboost_rung)
    wanted = os.environ.get('RAMPART_MODELS', '').strip()
    rungs = list(LADDER) + [neural_rung(), xgboost_rung(), lightgbm_rung()]
    if not wanted:
        return list(LADDER)
    names = {w.strip() for w in wanted.split(',')}
    picked = [r for r in rungs if r.name in names]
    missing = names - {r.name for r in picked}
    if missing:
        raise SystemExit(f'RAMPART_MODELS names unknown rungs: {sorted(missing)}')
    return picked


#: Lags of the target, which the pipeline also builds. Two and three years
#: because the gap is two: a one-year lag would be inside it.
LAGS = (2, 3)


def entity_subsample(df, cap, *, seed=RANDOM_SEED):
    """A subsample of entities that is a subsample, not a region.

    `sorted(df['entity_id'].unique())[:cap]` was written in five probes, and on INEP
    it does not do what its callers believed. The entity id there is the IBGE
    municipality code, whose leading digits are the federative unit, so the first 400
    sorted are prefixes 11 to 17 -- the whole North region, seven of twenty-seven
    states, target mean 11.26 against 8.29 for the full panel. Section 4.2o then
    attributed the difference between that subsample and the full run to sample size,
    which is why this function exists: n and region had moved together and only one of
    them was named.

    Seeded, so a subsampled replication is still reproducible, and in one place, which
    is the rule this repository has now broken six times.
    """
    entities = np.sort(df['entity_id'].unique())
    if cap is None or int(cap) >= len(entities):
        return df
    keep = np.random.default_rng(seed).choice(entities, size=int(cap), replace=False)
    return df[df['entity_id'].isin(set(keep))].reset_index(drop=True)


def spillover_degree(df, cfg) -> float:
    """How many *other* evaluation rows share an entity with a contaminated one.

    The moderator the regime prediction rests on, and it is computable from the
    fold configuration before anything runs. An evaluation window covers
    `test_len` years and a panel has some number of rows per entity-year, so a
    contaminated row has that many minus itself to spill onto. World Bank gives
    exactly one; INEP gives zero, because each municipality contributes a single
    row to a one-year evaluation window and that row is the contaminated one.
    """
    rows_per_entity_year = (len(df)
                            / (df['entity_id'].nunique() * df['year'].nunique()))
    return cfg.walk_forward_config['test_len'] * rows_per_entity_year - 1.0


def panel(dataset: str = 'worldbank'):
    """A panel with target and lags, plus the columns a model may fit on.

    The feature list comes from `modelling_features`, never from
    `feature_columns` directly -- on both panels that list contains columns the
    target is computed from.
    """
    if dataset not in PANELS:
        raise KeyError(f"unknown panel {dataset!r}; known: {list(PANELS)}")
    spec = PANELS[dataset]
    cfg = get_dataset(spec.get('config', dataset))
    source = Path(spec['path'])
    if not source.exists():
        raise FileNotFoundError(
            f"panel {dataset!r} is not at {source}.\n"
            f"The raw collections live outside this repository and the container "
            f"does not carry them. Point RAMPART_PANEL_DIR at the directory "
            f"holding azure_results_v7_wb/ and azure_results_v7_inep/, or copy "
            f"them in. Note that outputs/{dataset}/collection/raw_data/ holds an "
            f"eight-byte fixture, not the panel.")
    df = pd.read_parquet(source).rename(columns=spec['rename'])
    df['target'] = spec['target'](df)
    # Applied before the lags, so a lag never reaches across a row the filter removed
    # and quietly reports a two-year lag that spans four. Takes the frame and returns
    # the frame, because the two filters that exist need different things: one selects
    # rows by a key set read from another panel, the other drops rows by a value.
    if spec.get('filter') is not None:
        before = len(df)
        df = spec['filter'](df).reset_index(drop=True)
        print(f"panel {dataset}: {len(df)} of {before} rows after the declared filter")
    for k in LAGS:
        lag = df[['entity_id', 'year', 'target']].copy()
        lag['year'] += k
        df = df.merge(lag.rename(columns={'target': f'lag_{k}'}),
                      on=['entity_id', 'year'], how='left')
    columns = modelling_features(cfg, df.columns) + [f'lag_{k}' for k in LAGS]
    return df, columns, cfg


def folds(cfg):
    """Walk-forward windows, with both gaps between training and evaluation.

    The evaluation window starts after the validation window and two gaps, which
    is where the pipeline puts it: training to validation is one gap, validation
    to test is another.
    """
    w = cfg.walk_forward_config
    start, end = cfg.temporal_range
    out, train_end = [], start + w['min_train'] - 1
    while True:
        test_start = train_end + 2 * w['gap'] + w['val_len'] + 1
        test_end = test_start + w['test_len'] - 1
        if test_end > end:
            return out
        out.append((start, train_end, test_start, test_end))
        train_end += w['step']


def prepared(df, columns, train_start, train_end, test_start, test_end,
             *, min_train_rows=20, min_test_rows=3):
    """Training and evaluation frames, imputed from the training window only.

    P5 holds even in a probe. A baseline that imputed from the whole panel would
    inflate both arms and hide the contrast the probe exists to measure.

    Returns seven vectors -- design matrix, target, entity and year for the
    training window, then design matrix, target, entity and year for the
    evaluation one -- or None when a window is too thin to fit or to score, so a
    caller can skip the fold rather than average a meaningless number into it.

    The evaluation year is returned because the injector needs it: a leaked row
    keeps its own year, which is what lets the disjointness gate see it.
    """
    # Sorted by year, stably, and this is a contract rather than a convenience.
    # The in-context estimators cap their context to the most recent rows, and
    # they do it by taking the tail of whatever frame they are handed -- which is
    # only the recency rule if the frame is in chronological order. Sorting here,
    # once, is what lets the cap live in the estimator instead of being an
    # obligation on every caller. `kind='stable'` keeps rows of the same year in
    # the panel's own order, so the choice does not depend on the sort.
    train = (df[(df['year'] >= train_start) & (df['year'] <= train_end)]
             .sort_values('year', kind='stable'))
    test = (df[(df['year'] >= test_start) & (df['year'] <= test_end)]
            .sort_values('year', kind='stable'))
    fill = train[columns].median()
    X_train, X_test = train[columns].fillna(fill), test[columns].fillna(fill)
    keep = X_train.notna().all(axis=1) & train['target'].notna()
    valid = X_test.notna().all(axis=1) & test['target'].notna()
    if keep.sum() < min_train_rows or valid.sum() < min_test_rows:
        return None
    return (X_train[keep].reset_index(drop=True),
            train['target'][keep].reset_index(drop=True),
            train['entity_id'][keep].reset_index(drop=True),
            train['year'][keep].reset_index(drop=True),
            X_test[valid].reset_index(drop=True),
            test['target'][valid].reset_index(drop=True),
            test['entity_id'][valid].reset_index(drop=True),
            test['year'][valid].reset_index(drop=True))


def contaminate(X_train, y_train, entities_train, years_train,
                X_test, y_test, entities_test, years_test,
                *, dose, rng):
    """Class III: a fraction of the evaluation rows, with their labels, added.

    The added rows keep their own years, as `core.injection` keeps them. An
    earlier version stamped them as the last training year, reasoning that a
    leaked row is one the pipeline believes belongs in the window -- but the
    pipeline's own injector does not do that, and a probe that differs from the
    thing it is probing is measuring something else. The true years are also what
    lets the disjointness gate see the violation at all.
    """
    count = max(1, int(round(dose * len(X_test))))
    picked = np.sort(rng.choice(len(X_test), size=count, replace=False))
    return (
        pd.concat([X_train, X_test.iloc[picked]], ignore_index=True),
        pd.concat([y_train, y_test.iloc[picked]], ignore_index=True),
        pd.concat([entities_train, entities_test.iloc[picked]],
                  ignore_index=True),
        pd.concat([years_train, years_test.iloc[picked]], ignore_index=True),
    )


def fold_rng(*parts):
    """A generator keyed on the fold and whatever else identifies the cell.

    So that rerunning one fold reproduces it, which a single run-level generator
    would not: the draw for fold three would depend on how many folds ran first.
    """
    from core.scientific_config import RANDOM_SEED
    return np.random.default_rng(
        abs(hash((RANDOM_SEED, *parts))) % (2 ** 32))


def declare_provenance(**extra) -> None:
    """Print the settings that govern the numbers this run is about to print.

    A log that carries the readings but not the settings that produced them can
    only be audited from outside itself. Proving that a published interval used
    the configured resample count once meant reading the git history of
    `scientific_config` and dating the commit against the run -- which answers
    what the default was, not what ran.
    """
    from core.scientific_config import RANDOM_SEED, SCIENTIFIC_CONFIG
    icl = SCIENTIFIC_CONFIG['in_context_models']
    print("\n--- provenance: the settings these numbers were produced under ---")
    print(f"  bootstrap_iters (configured)  {SCIENTIFIC_CONFIG['bootstrap_iters']}")
    print(f"  absorption_probes             {icl['absorption_probes']}")
    print(f"  absorption_replicates         {icl['absorption_replicates']}")
    print(f"  random_seed                   {RANDOM_SEED}")
    for key, value in extra.items():
        print(f"  {key:<29} {value}")


def audit_resamples() -> int:
    """Print the resample counts that actually ran, against the configured one.

    Returns the number of distinct counts observed. The configured value is what
    the protocol declares; this is what executed, and the two part company as
    soon as one call site passes `iters=` of its own. Printed at the end of a
    probe so that the log answers "how many resamples produced this interval"
    without leaving the log.
    """
    from core.scientific_config import SCIENTIFIC_CONFIG
    from statistical_validation.dependent_bootstrap import observed_resample_counts

    configured = int(SCIENTIFIC_CONFIG['bootstrap_iters'])
    observed = observed_resample_counts()
    print("\n--- resample audit ---")
    if not observed:
        print("  no intervals were produced in this run")
        return 0
    for count in sorted(observed):
        mark = '' if count == configured else '   <-- NOT the configured count'
        print(f"  {observed[count]:>5} intervals at {count:>6} resamples{mark}")
    if set(observed) != {configured}:
        print(f"  the configured count is {configured}; this run did not use it "
              f"everywhere, so no single resample count describes these tables")
    return len(observed)
