#!/usr/bin/env python3
"""Which models a run fits, declared before it starts.

Two published models -- the hierarchical Ridge and the hierarchical random
forest -- are what the frozen artifact reports, and they run whether or not
anything is declared here. Everything else is opt-in through `RAMPART_MODELS`,
for the same reason injection is opt-in through `RAMPART_INJECTION`: a run that
was not asked to be experimental must not become experimental by accident, and
the clean path has to stay bit-for-bit what it was.

The variable names models, not groups. `RAMPART_MODELS=ladder_knn,icl_tabpfn` is
readable in a receipt six months later in a way that `all` is not, and an
unknown name stops the run rather than quietly fitting fewer models than the
label claims -- a typo that silently dropped a rung would leave a gap in the
trend and no evidence of why.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from core.models.icl import MODELS as ICL_MODELS
from core.models.ladder import RUNGS as LADDER_RUNGS

#: The variable the orchestrator exports and `run()` copies into subprocesses,
#: the same route RAMPART_INJECTION and RAMPART_RUN_ID take.
ENV_VAR = 'RAMPART_MODELS'

#: Fitted always. Named for what they are in the paper, not for the functions.
PUBLISHED = ('simple_hierarchical', 'random_forest_hierarchical')

#: Shorthands, because writing five rung names is how one of them gets dropped.
GROUPS: Dict[str, Tuple[str, ...]] = {
    'ladder': tuple(LADDER_RUNGS),
    'icl': tuple(ICL_MODELS),
}

KNOWN: Tuple[str, ...] = tuple(LADDER_RUNGS) + tuple(ICL_MODELS)


def requested() -> List[str]:
    """The extra models this run fits, in ladder order, or an empty list.

    Read on every call rather than cached: the paradigms run as subprocesses and
    each reads the variable at the point of use, so no import order can let a
    cached empty list outlive the environment that set it.
    """
    raw = os.environ.get(ENV_VAR, '').strip()
    if not raw:
        return []

    asked: List[str] = []
    for token in (part.strip() for part in raw.split(',')):
        if not token:
            continue
        if token in GROUPS:
            asked.extend(GROUPS[token])
        elif token in KNOWN:
            asked.append(token)
        else:
            raise ValueError(
                f"unknown model {token!r} in {ENV_VAR}. Known models: "
                f"{list(KNOWN)}; known groups: {list(GROUPS)}. An unrecognised "
                f"name must stop the run -- ignoring it would fit fewer models "
                f"than the arm's label claims.")

    # Deduplicated, and ordered by the ladder rather than by how the variable
    # was typed, so two spellings of the same request produce the same artifact.
    order = {name: index for index, name in enumerate(KNOWN)}
    return sorted(dict.fromkeys(asked), key=lambda name: order[name])


def models_reported(folds) -> List[str]:
    """Every model the folds actually produced, in the order they appeared.

    The three paradigms each printed their aggregate over a list written out by
    hand -- `['simple_hierarchical', 'random_forest_hierarchical']` -- so a rung
    added to a run was a rung missing from all three summaries, and nothing said
    so: the aggregation would simply be short and read as complete.

    Derived from the results rather than declared, so it cannot fall behind what
    was fitted.
    """
    seen: List[str] = []
    for fold in folds:
        for name in fold.get('models', {}):
            if name not in seen:
                seen.append(name)
    return seen


def fit_requested(X_train, y_train, X_test, y_test,
                  entities_train, entities_test, *, architecture,
                  years_train=None) -> Dict[str, Dict]:
    """Fit every extra model this run asked for, keyed by model name.

    Empty when nothing was asked for, which is the ordinary case and the one
    that has to cost nothing: with no request, neither optional package is
    imported and no estimator is built.
    """
    from core.models.icl import fit_in_context
    from core.models.ladder import fit_rung

    results: Dict[str, Dict] = {}
    for name in requested():
        if name in LADDER_RUNGS:
            results[name] = fit_rung(
                X_train, y_train, X_test, y_test,
                entities_train, entities_test,
                rung=LADDER_RUNGS[name], architecture=architecture)
        else:
            results[name] = fit_in_context(
                X_train, y_train, X_test, y_test,
                entities_train, entities_test,
                model=ICL_MODELS[name], architecture=architecture,
                years_train=years_train)
    return results
