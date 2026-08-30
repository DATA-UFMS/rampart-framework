#!/usr/bin/env python3
"""A capacity ladder: several estimators sharing one hierarchical form.

The comparison this serves asks whether an in-context model absorbs a protocol
violation more than its capacity alone would predict. Answering it against a
single classical comparator gives one step, and a step cannot distinguish "the
in-context model is unusual" from "the in-context model is merely large". Five
rungs give a trend, and the question becomes whether the in-context point falls
off it.

**One form, so that only capacity varies.** Every rung is fitted the same way:
entity means computed on the training window, joined as one extra column, and a
scikit-learn regressor on top. That is the form the published random forest
already used. Were the bottom rung fitted as the published Ridge is -- entity
means plus a shrunken residual model -- the ladder would confound capacity with
hierarchical form, and the trend would not mean what it is read to mean.

**No hyperparameter search.** Not economy. Tuning on the validation window
inside a contaminated arm lets the contamination pick the hyperparameter, and
the difference between arms then mixes two effects: how much the violation
inflates the fit, and how much it moved the selection. Fixed knobs keep the
counterfactual to one switch, which is the design the whole study rests on. The
two published models keep the search they had, because the clean run has to
reproduce the frozen artifact.

**Roth's severities are recorded here and are not the axis. They did not
transfer.** He measures class III inflation rising with capacity -- naive Bayes
0.37, logistic regression 0.44, XGBoost 0.78, random forest 0.90, kNN 1.01,
decision tree 1.11 -- over 2,047 datasets, by AUC, on classifiers. Carried across
to regression on one panel, that order does not predict inflation: measured, the
correlation between rung position and inflation is -0.72, -0.55 and +0.04 across
the three doses, and gradient boosting inflates about five times more than the
random forest ranked above it. Roth himself warns against carrying his numbers
across, and this is what it looks like when one does.

The axis that works is measured rather than borrowed: see
`core.models.absorption`, which asks each rung how much of a single handed
answer it keeps. The severities below stay in the artifact as the external
reference that was tried, because a failed transfer is a result and deleting it
would leave the reader wondering whether it had ever been checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.scientific_config import RANDOM_SEED, SCIENTIFIC_CONFIG

#: The column the entity effect is joined as. Named as the published random
#: forest named it, so the two agree on what the design matrix looks like.
ENTITY_EFFECT_COLUMN = 'country_effect'


@dataclass(frozen=True)
class Rung:
    """One step of the ladder: how to build the estimator, and where it sits."""

    name: str
    make: Callable[[], object]
    #: Roth's class III effect size at 10% duplication. Recorded, not used: it
    #: did not transfer, and the module docstring says how it was checked.
    roth_severity: float
    #: The classifier Roth measured, whose regression analogue this is. Recorded
    #: because the analogy is a judgement and belongs in the artifact.
    roth_analogue: str


def _ridge():
    from sklearn.linear_model import RidgeCV
    _hm = SCIENTIFIC_CONFIG['hierarchical_model']
    return RidgeCV(alphas=np.logspace(_hm['ridge_alpha_log10_start'],
                                      _hm['ridge_alpha_log10_stop'],
                                      _hm['ridge_alpha_count']))


def _gradient_boosting():
    from sklearn.ensemble import GradientBoostingRegressor
    _lc = SCIENTIFIC_CONFIG['capacity_ladder']
    return GradientBoostingRegressor(
        n_estimators=_lc['gb_n_estimators'], max_depth=_lc['gb_max_depth'],
        learning_rate=_lc['gb_learning_rate'], random_state=RANDOM_SEED)


def _random_forest():
    from sklearn.ensemble import RandomForestRegressor
    _hm = SCIENTIFIC_CONFIG['hierarchical_model']
    _lc = SCIENTIFIC_CONFIG['capacity_ladder']
    return RandomForestRegressor(
        n_estimators=_hm['rf_n_estimators'], max_depth=_lc['rf_max_depth'],
        min_samples_split=_hm['rf_min_samples_split'],
        min_samples_leaf=_lc['rf_min_samples_leaf'],
        max_features=_hm['rf_max_features'],
        random_state=RANDOM_SEED, n_jobs=_hm['rf_n_jobs'])


def _knn():
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(
        n_neighbors=SCIENTIFIC_CONFIG['capacity_ladder']['knn_n_neighbors'])


def _decision_tree():
    from sklearn.tree import DecisionTreeRegressor
    # Unbounded depth on purpose: this is the top of the ladder, and the point
    # of the top is that nothing stops it from fitting a duplicated row exactly.
    return DecisionTreeRegressor(
        min_samples_leaf=SCIENTIFIC_CONFIG['capacity_ladder']['dt_min_samples_leaf'],
        random_state=RANDOM_SEED)


#: Declared in the order Roth's severities put them, which is *not* the order
#: their absorption puts them: measured, it runs ridge 0.29, kNN 0.37, forest
#: 0.39, boosting 0.99, tree 1.00, so boosting and the forest swap. The tuple
#: order is kept as the record of what was tried and is never read as a scale --
#: `core.models.absorption` supplies the axis. A tuple so the order is part of the
#: artifact rather than an accident of iteration.
LADDER: Tuple[Rung, ...] = (
    Rung('ladder_ridge', _ridge, 0.44, 'logistic regression'),
    Rung('ladder_gradient_boosting', _gradient_boosting, 0.78, 'XGBoost'),
    Rung('ladder_random_forest', _random_forest, 0.90, 'random forest'),
    Rung('ladder_knn', _knn, 1.01, 'k-nearest neighbours'),
    Rung('ladder_decision_tree', _decision_tree, 1.11, 'decision tree'),
)

RUNGS: Dict[str, Rung] = {rung.name: rung for rung in LADDER}


def neural_rung() -> Rung:
    """A neural rung, deliberately OUTSIDE the LADDER tuple.

    The five classical rungs are the roster of the published record and of
    every golden test; the interference audit adds this one behind an
    explicit request (RAMPART_MODELS) so no existing run or test changes.
    Multilayer perceptron with standardised inputs (the panels' features
    span orders of magnitude), fixed seed, no early stopping (its internal
    validation split would add a second RNG consumer). Its clean fit on the
    large panel is poor under the temporal shift, like every other rung's
    there; the audit measures bias against each model's own clean fit, so a
    weak baseline is a property being measured, not a defect of the probe.
    RAMPART_NEURAL_SEED (integer, default RANDOM_SEED) overrides the MLP's
    random_state; it exists for the F1.1 multi-seed fleet.
    """
    def _mlp():
        import os
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        raw = os.environ.get('RAMPART_NEURAL_SEED')
        if raw is None:
            seed = RANDOM_SEED
        else:
            # Set-but-blank (a shell export bug) must stop the shard, not
            # quietly run the default seed: a multi-seed fleet that silently
            # collapses onto seed 42 burns the whole fleet's compute.
            try:
                seed = int(raw)
            except ValueError as exc:
                raise RuntimeError(
                    f"RAMPART_NEURAL_SEED is set but is not an integer: "
                    f"{raw!r}") from exc
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32,), max_iter=300,
                         batch_size=1024, random_state=seed,
                         early_stopping=False))
    return Rung('ladder_mlp', _mlp, float('nan'),
                'not in Roth: added for the interference audit')


def xgboost_rung() -> Rung:
    """An XGBoost rung, deliberately OUTSIDE the LADDER tuple.

    Like the neural rung, it exists only behind an explicit RAMPART_MODELS
    request (F1.3 fleet: does the global-learner pattern survive a change of
    boosting implementation?), so no existing run or test changes. The import
    is inside `make`, so a machine without the package fails only when the
    rung is actually requested.
    """
    def _xgb():
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "ladder_xgboost was requested via RAMPART_MODELS but the "
                "'xgboost' package is not installed in this environment"
            ) from exc
        # Modest fixed hyperparameters, no search (module docstring says why):
        # 300 trees at depth 3 with learning rate 0.05, squared-error
        # objective to match the ladder's loss, fixed seed, one thread,
        # verbosity off.
        # tree_method pinned: the auto default changed at xgboost 2.0 and
        # the fleet must be reproducible across image versions.
        return XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            objective='reg:squarederror', random_state=RANDOM_SEED,
            tree_method='hist', n_jobs=1, verbosity=0)
    return Rung('ladder_xgboost', _xgb, 0.78,
                'XGBoost: the one system Roth measured directly')


def lightgbm_rung() -> Rung:
    """A LightGBM rung, deliberately OUTSIDE the LADDER tuple.

    Same contract as `xgboost_rung`: opt-in only via RAMPART_MODELS (F1.3
    fleet), import inside `make` so a missing package fails only when the
    rung is actually requested.
    """
    def _lgbm():
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "ladder_lightgbm was requested via RAMPART_MODELS but the "
                "'lightgbm' package is not installed in this environment"
            ) from exc
        # Modest fixed hyperparameters, no search: 300 trees, learning rate
        # 0.05, the library's default leaf-wise growth (num_leaves=31),
        # squared-error objective ('regression') to match the ladder's loss,
        # fixed seed, one thread, verbosity off.
        # deterministic + force_row_wise pinned: the row/col-wise auto choice
        # is timing-based and the two paths differ in FP summation order.
        return LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            objective='regression', random_state=RANDOM_SEED,
            deterministic=True, force_row_wise=True,
            n_jobs=1, verbose=-1)
    return Rung('ladder_lightgbm', _lgbm, float('nan'),
                'not in Roth: added for the F1.3 boosting fleet')


def entity_effect_frames(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, entities_train: pd.Series, entities_test: pd.Series,
    *, contaminate_with: Optional[Tuple[pd.Series, pd.Series]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], float]:
    """Join the training-window entity mean as one column, on both frames.

    Fitted on the training window and applied outward, which is P5: an entity
    absent from training gets the global training mean rather than its own,
    because its own would have to be read from the window being predicted.

    Raw means, not shrunken. The published random forest computed them this way
    and the ladder has to agree with it, or its random-forest rung would not be
    the same model the paper reports.

    `contaminate_with` takes (targets, entities) from outside the training window
    and folds them into the means. That is a deliberate violation and the reason
    it exists is that this column is a **target encoding**: the mean of the
    outcome per category, which is the one estimation leakage Roth measures as
    large (d_z = +0.46) and the one his mechanism-first taxonomy has to carve out,
    because its mechanism is Class I while its magnitude is Class II. Contaminating
    it moves no row into the training frame -- only label information, attenuated
    by however many training years each entity has. It is the handle for testing
    whether severity follows label information rather than mechanism.
    """
    if contaminate_with is not None:
        extra_targets, extra_entities = contaminate_with
        y_train = pd.concat([pd.Series(y_train), pd.Series(extra_targets)],
                            ignore_index=True)
        entities_train = pd.concat([pd.Series(entities_train),
                                    pd.Series(extra_entities)],
                                   ignore_index=True)
        # entities_train is now longer than X_train, which is correct and is the
        # whole point: the statistic saw rows the design matrix does not contain.
        # Only the means below read it.

    global_mean = float(y_train.mean())
    means = {entity: float(y_train[entities_train == entity].mean())
             for entity in entities_train.unique()}

    train_augmented = X_train.copy()
    test_augmented = X_test.copy()
    train_augmented[ENTITY_EFFECT_COLUMN] = [
        means.get(entity, global_mean)
        for entity in entities_train.iloc[:len(X_train)]]
    test_augmented[ENTITY_EFFECT_COLUMN] = [
        means.get(entity, global_mean) for entity in entities_test]
    return train_augmented, test_augmented, means, global_mean


def fit_rung(X_train: pd.DataFrame, y_train: pd.Series,
             X_test: pd.DataFrame, y_test: pd.Series,
             entities_train: pd.Series, entities_test: pd.Series,
             *, rung: Rung, architecture: str,
             measure_absorption: bool = False) -> Dict:
    """Fit one rung and score it, in the shape the other models return.

    `measure_absorption` costs a handful of extra fits and buys the axis this
    ladder is read along -- see `core.models.absorption`. The caller decides,
    because it is only meaningful on an uncontaminated frame and the caller is
    the one that knows whether this arm is one.
    """
    X_train_augmented, X_test_augmented, means, global_mean = \
        entity_effect_frames(X_train, X_test, y_train,
                             entities_train, entities_test)

    estimator = rung.make()
    estimator.fit(X_train_augmented, y_train)
    predictions = np.asarray(estimator.predict(X_test_augmented), dtype=float)

    absorption = None
    if measure_absorption:
        from core.models.absorption import absorption_coefficient
        absorption = absorption_coefficient(
            rung.make, X_train_augmented, y_train,
            X_test_augmented, y_test, baseline=predictions)

    return {
        'model_name': rung.name,
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
        'ladder_position': {
            'roth_severity': rung.roth_severity,
            'roth_analogue': rung.roth_analogue,
            'rank': LADDER.index(rung) + 1,
            'rungs': len(LADDER),
        },
        'regularization_applied': (
            f"fixed hyperparameters, no validation-set search "
            f"({type(estimator).__name__})"),
        'absorption': absorption,
    }
