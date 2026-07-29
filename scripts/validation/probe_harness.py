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
from core.dataset_config import get_dataset, modelling_features  # noqa: E402

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
}

#: Roth's own duplication rates, so the doses are not ours to choose.
DOSES = (0.05, 0.10, 0.30)

#: Lags of the target, which the pipeline also builds. Two and three years
#: because the gap is two: a one-year lag would be inside it.
LAGS = (2, 3)


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
