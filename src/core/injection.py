#!/usr/bin/env python3
"""Deliberate protocol violations, declared before the run and recorded after it.

The experiment this serves measures how much a violation inflates a result, and
whether it inflates more for one model family than another. That requires
committing the violation on purpose, inside the pipeline, so the gates see the
same thing a careless practitioner would produce -- and so the answer to "would
our auditor have caught this?" is observed rather than argued.

Three properties the design has to hold, and the reasons are not stylistic:

**One switch.** A spec activates exactly one class at one dose. The scenario
this module replaces flipped two at once -- naive k-fold *and* an extra leaking
feature -- so the inflation it measured was attributable to neither. Zhang et
al. (2026) establish the one-switch counterfactual as the design; the field
reads a two-switch result as uninterpretable, correctly.

**Silence by default.** With `RAMPART_INJECTION` unset there is no spec, no
waiver and no behavioural difference. A production run cannot be made
experimental by accident, and the absence is what the test suite asserts.

**No hidden waivers.** A gate is never softened by consulting the environment
behind the caller's back. The caller passes the spec explicitly, the gate
records what it let through, and the record reaches the receipt. An arm that
waived a gate is therefore distinguishable from a clean run by its artifacts
alone -- which matters, because those artifacts are what a reader gets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: The variable the orchestrator exports and `run()` copies into subprocesses,
#: the same route RAMPART_RUN_ID and DATASET_NAME take.
ENV_VAR = 'RAMPART_INJECTION'

#: Classes this module can commit. Named for the taxonomy the study extends
#: rather than for what the code does, so the artifact and the paper agree.
#:
#:   C1  estimation over the full panel -- preprocessing statistics fitted on
#:       training and evaluation rows together. Roth measures this as negligible
#:       for classical models, which is why it is the primary comparison: it is
#:       where a model that absorbs preprocessing structure should separate.
#:   C3  memorisation -- evaluation rows pasted into the training frame. Roth's
#:       capacity ladder lives here, which makes it the positive control.
CLASSES = ('C1', 'C3')

#: Gate identifiers a spec may waive. Enumerated rather than free-form: a typo
#: in a waiver would otherwise silently fail to waive, and the arm would abort
#: hours in with a message about a violation it was designed to commit.
GATES = ('L1.1',)


@dataclass(frozen=True)
class InjectionSpec:
    """What violation this arm commits, and what it is allowed to trip."""

    klass: str
    dose: float
    waived: Tuple[str, ...] = field(default_factory=tuple)
    seed: int = 42

    def __post_init__(self):
        if self.klass not in CLASSES:
            raise ValueError(
                f"unknown injection class {self.klass!r}; known: {list(CLASSES)}")
        if not 0.0 < self.dose <= 1.0:
            raise ValueError(
                f"dose must be in (0, 1] and is {self.dose!r}. A dose of zero "
                f"is the clean arm, which is expressed by declaring no "
                f"injection at all rather than by an injection of nothing.")
        unknown = [gate for gate in self.waived if gate not in GATES]
        if unknown:
            raise ValueError(
                f"unknown gate(s) to waive: {unknown}; known: {list(GATES)}")

    def waives(self, gate: str) -> bool:
        return gate in self.waived

    def as_record(self) -> Dict:
        """The form that goes into receipts and the reproducibility snapshot."""
        return {'class': self.klass, 'dose': self.dose,
                'waived_gates': list(self.waived), 'seed': self.seed}


def active() -> Optional[InjectionSpec]:
    """The spec for this run, or None -- which is the ordinary case.

    Parsed on every call rather than cached: the models run as subprocesses and
    each reads the variable once at the point of use, so there is no import
    order in which a cached None outlives the environment that set it.
    """
    raw = os.environ.get(ENV_VAR, '').strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            f"{ENV_VAR} is not valid JSON: {exc}. An unparseable spec must "
            f"stop the run -- treating it as absent would silently produce a "
            f"clean arm under an experimental label.") from exc
    return InjectionSpec(
        klass=payload['class'],
        dose=float(payload['dose']),
        waived=tuple(payload.get('waived_gates', ())),
        seed=int(payload.get('seed', 42)),
    )


def _fold_rng(spec: InjectionSpec, fold_id) -> np.random.Generator:
    """A generator that depends on the fold, so folds differ and the run replays.

    One generator for the whole run would make the sample drawn for fold 3
    depend on how many folds ran before it, and a rerun of one fold would not
    reproduce. Seeding per fold costs nothing and removes that.
    """
    return np.random.default_rng(abs(hash((spec.seed, spec.klass,
                                           spec.dose, str(fold_id)))) % (2 ** 32))


def duplicate_evaluation_rows(
    X_train: pd.DataFrame, y_train: pd.Series,
    entities_train: pd.Series, years_train: pd.Series,
    X_eval: pd.DataFrame, y_eval: pd.Series,
    entities_eval: pd.Series, years_eval: pd.Series,
    *, spec: InjectionSpec, fold_id,
) -> Tuple[Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series], Dict]:
    """C3: paste a fraction of the evaluation rows into the training frame.

    With their true targets, which is the point -- the model is handed the
    answer for rows it will be scored on. This is the violation a practitioner
    commits by shuffling before splitting, and it is invisible to any check that
    reasons about windows rather than rows.

    Returns the widened training split and a record of exactly which rows moved,
    so the arm is auditable after the fact and not only by its label.
    """
    if spec.klass != 'C3':
        raise ValueError(f"duplicate_evaluation_rows is C3; spec is {spec.klass}")

    count = max(1, int(round(spec.dose * len(X_eval))))
    count = min(count, len(X_eval))
    rng = _fold_rng(spec, fold_id)
    picked = np.sort(rng.choice(len(X_eval), size=count, replace=False))

    widened = (
        pd.concat([X_train, X_eval.iloc[picked]], ignore_index=True),
        pd.concat([pd.Series(y_train), pd.Series(y_eval).iloc[picked]],
                  ignore_index=True),
        pd.concat([pd.Series(entities_train), pd.Series(entities_eval).iloc[picked]],
                  ignore_index=True),
        pd.concat([pd.Series(years_train), pd.Series(years_eval).iloc[picked]],
                  ignore_index=True),
    )
    record = {
        'class': 'C3',
        'dose': spec.dose,
        'rows_moved': int(count),
        'evaluation_rows': int(len(X_eval)),
        'training_rows_before': int(len(X_train)),
        'training_rows_after': int(len(widened[0])),
        'keys_moved': [[str(e), int(y)] for e, y in zip(
            pd.Series(entities_eval).iloc[picked],
            pd.Series(years_eval).iloc[picked])],
    }
    return widened, record


def contaminated_fit_frame(
    X_train: pd.DataFrame, *others: pd.DataFrame, spec: InjectionSpec,
) -> Tuple[pd.DataFrame, Dict]:
    """C1: the frame preprocessing statistics are fitted on, widened past training.

    The violation is not that a wrong function is called; it is that the right
    function is handed the wrong window. A practitioner who normalises the panel
    once, before splitting, produces exactly this -- and the P5 receipt, which
    attests that the imputation ran, cannot tell the difference on its own. That
    is why the report below carries the year span it was fitted over: presence
    becomes conformance only when the receipt says *what* was used.

    At dose < 1 a fraction of the evaluation rows joins the fit, which gives the
    dose-response axis something continuous to vary.
    """
    if spec.klass != 'C1':
        raise ValueError(f"contaminated_fit_frame is C1; spec is {spec.klass}")

    extra = []
    for frame in others:
        if frame is None or not len(frame):
            continue
        count = max(1, int(round(spec.dose * len(frame))))
        extra.append(frame.iloc[:min(count, len(frame))])

    widened = pd.concat([X_train, *extra], ignore_index=True) if extra else X_train
    return widened, {
        'class': 'C1',
        'dose': spec.dose,
        'fit_rows_before': int(len(X_train)),
        'fit_rows_after': int(len(widened)),
        'evaluation_rows_in_fit': int(len(widened) - len(X_train)),
    }
