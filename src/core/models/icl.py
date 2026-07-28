#!/usr/bin/env python3
"""In-context learners, behind the same contract as the ladder rungs.

Two families, so that a claim about in-context learning is not a claim about one
model: TabPFN and TabICL. Both take the training rows as context and produce
predictions in a single forward pass; neither fits parameters to them.

**Optional, and absent by default.** Neither package is in the lockfile. The
published artifact reproduces without torch, and a reviewer who only wants the
paradigm-equivalence result does not download a foundation model to get it. The
import happens inside the adapter, so an environment without the package fails
at the point of use with a message that says what to install, rather than at
import time in a run that was never going to use it.

**The weights are pinned to v2, and this is not a default we inherited.** As of
tabpfn 8.1.0 the default model is v3, whose weights sit behind a browser license
acceptance and a personal token; v2.5 and v2.6 are gated the same way. Version
v2 is not gated. Three reasons converge on it: it is the one an artifact can
reproduce without a credential, it is the one described in the peer-reviewed
account, and it is the one whose architecture this study audited. Pinning it in
code rather than in the environment matters -- a machine with a different
TABPFN_MODEL_VERSION would otherwise silently produce numbers from a different
model, and the receipt would not show it.

The pin is provisional, and the plan is to carry v3 alongside it rather than
instead of it. What settles it against the three reasons above: v3 is what the
package hands a practitioner by default, so it is the version a contamination
audit is actually about. Carrying both makes the version an axis -- whether
newer weights amplify more is a result, not an inconvenience -- and leaves one
arm that still reproduces without an account. Note that the version is not a
constant to swap: the inductive behaviour, the cost curve and the batch
tolerance recorded here were all measured on v2, and v3 is a different
architecture (architectures/tabpfn_v3.py). Each has to be re-measured, starting
with the inductive check, because the mechanism this study argues for rests on
it.

**Determinism is a tolerance, not an identity.** Predicting the same rows in one
batch, singly, and in halves moves the answer by about 1e-5 of the target's
standard deviation, with identical weights and seed. The cross-paradigm claim
this framework makes elsewhere is bitwise; for these models bitwise is not
available, and the honest form is a characterised tolerance.

**Context is capped, and the cap is recorded.** The forward pass is quadratic in
context length: measured here, 2,500 rows cost 32s and 10,000 cost 325s, so the
INEP training window of roughly 60,000 rows does not fit in any budget and would
exhaust memory first. The cap is the documented pretraining limit rather than a
number chosen to fit the schedule, and the rows kept are the most recent ones,
because dropping the oldest is the choice a practitioner facing the same limit
would make.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.models.ladder import entity_effect_frames
from core.scientific_config import RANDOM_SEED, SCIENTIFIC_CONFIG


class ICLUnavailable(ImportError):
    """Raised when an in-context model is asked for and its package is absent."""


def _tabpfn_regressor():
    """A TabPFN regressor pinned to the ungated v2 weights.

    The pin is applied to the settings object rather than to the environment
    because that object is built once at import, so exporting the variable
    afterwards has no effect -- which is exactly the silent-wrong-model failure
    this guards against. The assertion below is the guard: if a future release
    resolves the version some other way, the run stops instead of proceeding
    with weights nobody chose.
    """
    try:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import resolve_model_version
        from tabpfn.settings import settings
    except ImportError as exc:
        raise ICLUnavailable(
            "tabpfn is not installed. It is an optional dependency, kept out "
            "of the lockfile so the published artifact reproduces without it: "
            "pip install 'rampart[icl]'") from exc

    settings.tabpfn.model_version = ModelVersion.V2
    resolved = resolve_model_version(None)
    if resolved != ModelVersion.V2:
        raise RuntimeError(
            f"TabPFN resolved to {resolved} rather than v2. The other versions "
            f"require a license acceptance and a personal token, so an artifact "
            f"built on them does not reproduce; refusing to continue.")

    _ic = SCIENTIFIC_CONFIG['in_context_models']
    return TabPFNRegressor(n_estimators=_ic['tabpfn_n_estimators'],
                           random_state=RANDOM_SEED, device=_ic['device'])


def _tabicl_regressor():
    try:
        from tabicl import TabICLRegressor
    except ImportError as exc:
        raise ICLUnavailable(
            "tabicl is not installed. It is an optional dependency: "
            "pip install 'rampart[icl]'") from exc

    _ic = SCIENTIFIC_CONFIG['in_context_models']
    return TabICLRegressor(random_state=RANDOM_SEED, device=_ic['device'])


@dataclass(frozen=True)
class ICLModel:
    """One in-context family: how to build it, and how it is identified."""

    name: str
    make: Callable[[], object]
    #: What the receipt has to carry for the run to be identifiable later.
    package: str


FAMILIES: Tuple[ICLModel, ...] = (
    ICLModel('icl_tabpfn', _tabpfn_regressor, 'tabpfn'),
    ICLModel('icl_tabicl', _tabicl_regressor, 'tabicl'),
)

MODELS: Dict[str, ICLModel] = {family.name: family for family in FAMILIES}


def _package_version(package: str) -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(package)
    except PackageNotFoundError:
        return 'absent'


def cap_context(X_train: pd.DataFrame, y_train: pd.Series,
                entities_train: pd.Series, years_train=None) -> Tuple:
    """Keep the most recent rows when the training window exceeds the cap.

    Recency rather than a random sample: the cap is a constraint a practitioner
    hits too, and the row they drop is the old one. A random subsample would
    also break the correspondence between the context and the training window
    the classical rungs see, in a way that varies with the seed.

    Returns the kept rows and a record of what was dropped, because a silently
    truncated context reads as a full one in the results table.
    """
    cap = SCIENTIFIC_CONFIG['in_context_models']['context_cap_rows']
    if len(X_train) <= cap:
        return (X_train, y_train, entities_train, years_train,
                {'capped': False, 'context_rows': int(len(X_train)),
                 'cap': int(cap)})

    if years_train is None:
        raise ValueError(
            f"the training window has {len(X_train)} rows against a context cap "
            f"of {cap}, and no years were passed to choose by. Taking the last "
            f"rows by position would not take the recent ones: canonical_fold "
            f"sorts by (entity, year), so the tail of the frame is the last "
            f"entities alphabetically and not the last years.")

    # Most recent by year, ties broken by original position so the choice does
    # not depend on how the engine happened to order the frame.
    order = np.lexsort((np.arange(len(X_train)),
                        np.asarray(pd.Series(years_train))))
    keep = np.sort(order[-cap:])

    record = {
        'capped': True,
        'context_rows': int(cap),
        'cap': int(cap),
        'training_rows': int(len(X_train)),
        'rows_dropped': int(len(X_train) - cap),
        'rule': 'most recent rows by year, ties by original position',
    }
    return (X_train.iloc[keep].reset_index(drop=True),
            pd.Series(y_train).iloc[keep].reset_index(drop=True),
            pd.Series(entities_train).iloc[keep].reset_index(drop=True),
            None if years_train is None
            else pd.Series(years_train).iloc[keep].reset_index(drop=True),
            record)


def fit_in_context(X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   entities_train: pd.Series, entities_test: pd.Series,
                   *, model: ICLModel, architecture: str,
                   years_train=None, measure_absorption: bool = False) -> Dict:
    """Predict with an in-context model, in the shape the other models return.

    The entity effect is joined here as it is for every ladder rung. Handing the
    in-context model a different design matrix would make the comparison a
    comparison of feature sets.
    """
    X_train, y_train, entities_train, _years, context_record = cap_context(
        X_train, y_train, entities_train, years_train)

    X_train_augmented, X_test_augmented, means, _global = entity_effect_frames(
        X_train, X_test, y_train, entities_train, entities_test)

    estimator = model.make()
    estimator.fit(X_train_augmented, y_train)
    predictions = np.asarray(estimator.predict(X_test_augmented), dtype=float)

    absorption = None
    if measure_absorption:
        from core.models.absorption import absorption_coefficient
        absorption = absorption_coefficient(
            model.make, X_train_augmented, y_train,
            X_test_augmented, y_test, baseline=predictions)

    return {
        'model_name': model.name,
        'architecture': architecture,
        'mse': mean_squared_error(y_test, predictions),
        'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions),
        'predictions': predictions.tolist(),
        'y_true': list(y_test),
        'entities': [str(entity) for entity in entities_test],
        'country_effects': {str(k): v for k, v in means.items()},
        'features_count': X_train_augmented.shape[1],
        'context': context_record,
        'provenance': {
            'package': model.package,
            'package_version': _package_version(model.package),
            'weights': ('tabpfn v2, ungated' if model.package == 'tabpfn'
                        else 'tabicl default checkpoint'),
        },
        'regularization_applied': 'none; in-context, no parameters fitted',
        'absorption': absorption,
    }
