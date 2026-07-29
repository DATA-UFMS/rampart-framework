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

import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.models.ladder import entity_effect_frames
from core.scientific_config import RANDOM_SEED, SCIENTIFIC_CONFIG


#: Values that mean "off" when a switch is set explicitly. Needed because the
#: switches here used to be read as "is the variable non-empty", under which
#: `RAMPART_CAP_ALL=0` turned the arm **on** -- the reading a person types to
#: disable it. It cost a cloud run: the control arm for the matched-context
#: measurement capped exactly like the treatment arm, and the two logs came back
#: identical in all thirteen models, which is the only reason it was noticed.
_OFF = frozenset({'', '0', 'false', 'no', 'off'})


def switched_on(name: str) -> bool:
    """Whether an arm-selecting environment switch is set to something meaning yes.

    Absent, empty, `0`, `false`, `no` and `off` are all off; anything else is on.
    One reader for every switch, because the alternative is each caller inventing
    its own and the ones that disagree are found by a run that silently measures
    the wrong arm.
    """
    return os.environ.get(name, '').strip().lower() not in _OFF


class ICLUnavailable(ImportError):
    """Raised when an in-context model is asked for and its package is absent."""


class ContextCapped:
    """An in-context estimator that cannot be handed more rows than it accepts.

    The cap used to be the caller's job. It lived in `fit_in_context`, so every
    probe that built an estimator and called `.fit()` itself walked past it; then
    it lived in a helper the probes had to remember to call, and the absorption
    routine -- which appends rows to a frame and refits -- pushed a frame capped
    at exactly the limit twelve rows over it. Three cloud failures, one shape:
    a policy that depends on every caller remembering is a policy that will be
    forgotten, and this repository has now found that shape in the feature list,
    in the aggregation list, in the block length, and here.

    So the limit goes where it cannot be bypassed. Anything handed to `fit` is
    truncated before the wrapped model sees it, whether the frame came from a
    probe, from an injected arm, or from the absorption measurement widening one.
    No reserve to compute, no years to thread through, nothing for the next
    caller to remember.

    **Recency is the tail of the frame.** That is the pre-registered rule
    (pre-spec 4.2p) and it holds because `probe_harness.prepared` sorts by year
    and because the rows an arm or a probe appends are evaluation-window rows --
    newer than anything in training, and so exactly what recency should keep.
    """

    def __init__(self, estimator, cap: int, rule: str = 'recent'):
        self.estimator = estimator
        self.cap = int(cap)
        self.rule = rule
        self.context = {'cap': int(cap), 'capped': False}

    def fit(self, X, y):
        rows = len(X)
        if rows > self.cap:
            if self.rule == 'random':
                keep = np.sort(np.random.default_rng(RANDOM_SEED + rows)
                               .choice(rows, size=self.cap, replace=False))
            else:
                keep = np.arange(rows - self.cap, rows)
            X = X.iloc[keep].reset_index(drop=True)
            y = pd.Series(y).iloc[keep].reset_index(drop=True)
            self.context = {
                'cap': self.cap, 'capped': True, 'offered_rows': int(rows),
                'context_rows': int(self.cap),
                'rows_dropped': int(rows - self.cap),
                'rule': ('uniform random sample, registered sensitivity arm'
                         if self.rule == 'random'
                         else 'most recent rows, frame in chronological order')}
        else:
            self.context = {'cap': self.cap, 'capped': False,
                            'context_rows': int(rows)}
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def __getattr__(self, name):
        """Anything else belongs to the model underneath.

        The wrapper exists to enforce one rule, not to hide the estimator, so
        `n_estimators`, `random_state` and the rest read through. Without this a
        caller has to know it is holding a wrapper, which is the coupling the
        wrapper was introduced to remove.

        Only reached for attributes not found normally, so `estimator`, `cap`,
        `rule` and `context` resolve on the wrapper and cannot recurse.
        """
        return getattr(self.estimator, name)

    def __repr__(self):
        return (f"ContextCapped({self.estimator!r}, cap={self.cap}, "
                f"rule={self.rule!r})")


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
    # The ensemble is off by default: it averages over preprocessing
    # permutations, costs about 6.5x, and the estimand is a difference between
    # arms rather than the level of either. `RAMPART_ICL_ROBUSTNESS` turns it on
    # for the check a reviewer will ask for -- whether averaging shrinks the
    # absorption the whole mechanism claim rests on. A switch rather than a
    # config edit, so the two readings can sit in one table with the receipt
    # saying which is which.
    ensemble = (_ic['tabpfn_n_estimators_robustness']
                if switched_on('RAMPART_ICL_ROBUSTNESS')
                else _ic['tabpfn_n_estimators'])
    return _capped(TabPFNRegressor(n_estimators=ensemble,
                                   random_state=RANDOM_SEED,
                                   device=resolve_device()))


def _tabicl_regressor():
    try:
        from tabicl import TabICLRegressor
    except ImportError as exc:
        raise ICLUnavailable(
            "tabicl is not installed. It is an optional dependency: "
            "pip install 'rampart[icl]'") from exc

    _ic = SCIENTIFIC_CONFIG['in_context_models']
    return _capped(TabICLRegressor(random_state=RANDOM_SEED,
                                   device=resolve_device()))


def _capped(estimator):
    """Wrap so no caller can obtain an uncapped in-context estimator.

    The factory returns the wrapper, not the model, and that is the point: a bare
    estimator handed to something that appends rows is exactly the failure this
    closes.
    """
    _ic = SCIENTIFIC_CONFIG['in_context_models']
    rule = (os.environ.get('RAMPART_CONTEXT_RULE', '').strip()
            or _ic.get('context_rule', 'recent'))
    return ContextCapped(estimator, cap=_ic['context_cap_rows'], rule=rule)


def matched_context(make):
    """Wrap a *classical* factory so it reads the same context an ICL model does.

    The main analysis does not do this, and deliberately: a random forest can read
    the whole training window and an in-context model cannot, and that asymmetry
    is the constraint under study rather than a nuisance. But it leaves the
    absorption column measured at two different n -- ten thousand rows for the
    in-context arms against thirty-eight thousand for the classical ones -- so the
    cross-family comparison in that column is confounded, which the pre-spec had
    to declare as a caveat.

    `RAMPART_CAP_ALL=1` removes the caveat instead of declaring it: every model
    reads the same ten thousand rows and the absorptions become comparable. An
    extra arm, not a change to the default, because equal n answers a different
    question than the deployed configuration does.

    Returns the factory unchanged when the switch is off -- absent, or set to a
    value meaning no -- so the ordinary run is untouched and so a control arm can
    be asked for explicitly rather than by unsetting a variable.
    """
    if not switched_on('RAMPART_CAP_ALL'):
        return make
    return lambda: _capped(make())


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


def resolve_device() -> str:
    """The device the adapters will actually use, named rather than implied.

    `auto` is what the wrappers accept, but `auto` in a receipt says nothing.
    Resolving here means the artifact records `cuda` or `cpu`, which is the
    difference between a one-hour run and an eleven-hour one and belongs in the
    record of what was measured.
    """
    configured = SCIENTIFIC_CONFIG['in_context_models']['device']
    if configured != 'auto':
        return configured
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def _package_version(package: str) -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(package)
    except PackageNotFoundError:
        return 'absent'


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
    # The entity effect is fitted first, on the *whole* training window, and the
    # context is capped after. The order matters and the other one is wrong.
    #
    # Capping first would compute the entity mean from whatever survived the cap.
    # On INEP that is 10,000 rows over 5,564 entities -- under two observations
    # each, against twelve for the classical models, which are handed the full
    # window. The in-context models would then carry a far noisier version of the
    # strongest feature in the design matrix, and would score worse for a reason
    # that has nothing to do with in-context learning. The comparison would be of
    # feature quality wearing the label of a comparison of model families.
    #
    # It never showed up on World Bank because 400 training rows never reach the
    # cap, so this only ever mattered on the panel it was written for.
    #
    # P5 is untouched: the statistic still comes from the training window alone.
    # The cap is a constraint on what the model may read, not on where a
    # statistic may be fitted, and conflating the two is what produced the bug.
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
        'context': estimator.context,
        'provenance': {
            'package': model.package,
            'package_version': _package_version(model.package),
            'weights': ('tabpfn v2, ungated' if model.package == 'tabpfn'
                        else 'tabicl default checkpoint'),
            'ensemble_robustness': switched_on('RAMPART_ICL_ROBUSTNESS'),
            'device': resolve_device(),
        },
        'regularization_applied': 'none; in-context, no parameters fitted',
        'absorption': absorption,
    }
