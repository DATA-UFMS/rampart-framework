#!/usr/bin/env python3
"""
Centralised validation module for ML architectures.

Centralises all temporal validation, data integrity and scientific metric
logic, eliminating duplication across architectures.

Anti-leakage protocol (P1-P5):
    P1 — Temporal ordering: train_end < val_start < val_end < test_start.
    P2 — Minimum gap: N years between splits (default 2), configurable via
         temporal_gap_years. Optional embargo for sub-annual data.
    P3 — Feature separation: exclusion list (target-derived features,
         metadata) + proxy detection (|r| with the target above
         proxy_correlation_threshold, measured on the full panel) + rejection
         of joint reconstruction (least-squares R2 of the target on the
         selected features above identity_r2_threshold, measured on the
         training window).
    P4 — Selection scope: feature selection restricted to the training
         period of the first fold (Kapoor & Narayanan, 2023).
    P5 — Preprocessing scope: scaling and imputation fitted
         exclusively on training data (Kaufman et al. 2012).

HPO: grid search on the validation set; final model retrained
on the full training set. Prevents leakage from optimising on the test set (Kapoor & Narayanan, 2023).

Enforcement: P1/P2 violations raise ValueError via enforce_walk_forward().

Mapping to the Kapoor & Narayanan (2023) taxonomy, eight types:
    L1.2 preprocessing over train+test .............. P5
    L1.3 feature selection over train+test .......... P4
    L1.4 duplicates in the dataset .................. canonical_fold
    L2   illegitimate feature ....................... P3 (tracking, not discharge)
    L3.1 temporal leakage ........................... P1
    L3.2 dependency between train and test .......... P2 partially mitigates

The P2 gap does not come from K&N: their taxonomy does not mention gaps. It
follows the literature on blocked cross-validation with a buffer (Roberts
et al., 2017), which is the reference K&N themselves cite when dealing with
L3.2, with the embargo variant of López de Prado (2018).

L2 stands as tracking and not as discharge: K&N deliberately do not subdivide
this category because "judging whether the use of a given feature is legitimate
requires domain knowledge". A correlation threshold detects the detectable
subset -- the strongly associated proxy -- and does not reach a feature that is
illegitimate for being a consequence of the outcome rather than a cause of it.

Two types are NOT covered by P1-P5, and that is declared here rather than left
implicit in the absence:

    L3.2 dependency between train and test. The same country is in train and
         in test; the split is temporal, not by entity. K&N say this is
         leakage "unless the scientific claim is about a distribution with the
         same dependency structure". For panel forecasting the structure
         matches -- it is the same country, years ahead -- but the argument is
         the author's and not the code's. Note the deliberate asymmetry: the
         inner CV groups by country (GroupKFold) because there entity leakage
         inflates hyperparameter selection; the outer split does not group
         because grouping would turn the forecasting claim into a different one.

    L3.3 sampling bias in the test set. Covered by half: the minimum
         geographic coverage per fold addresses the spatial bias, which is
         K&N's own example. The other half is created by this pipeline -- rows
         with no observed target are removed, and target absence is not
         random. Their info sheet asks exactly this (Q18-19: "describe how the
         rows included in the analysis were selected"). The evidence is in
         target_coverage.json; the argument is the author's.
P3/P4 violations raise ValueError in run_feature_selection().
P5 is enforced by contract (docstring + unit tests).
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime


class AntiLeakageViolation(ValueError):
    """A property of the anti-leakage protocol was violated.

    Distinguished from operational failures because it is not recoverable: the
    experiment is invalid, and continuing would produce measurements of a
    pipeline that does not hold the guarantees the results are reported under.
    Subclasses ValueError so existing handlers and tests continue to match.
    """


try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


def materialise_pandas(data: Any, columns: List[str]) -> pd.DataFrame:
    """Materialise the given columns of any supported frame as pandas."""
    if _HAS_POLARS and isinstance(data, pl.LazyFrame):
        return data.select(columns).collect().to_pandas()
    if _HAS_POLARS and isinstance(data, pl.DataFrame):
        return data.select(columns).to_pandas()
    if hasattr(data, 'compute'):  # Dask
        return data[columns].compute()
    if isinstance(data, pd.DataFrame):
        return data[columns].copy()
    raise TypeError(f"Unsupported data type for materialisation: {type(data)}")


def linear_reconstruction_r2(
    data: Any, features: List[str], target_column: str
) -> Optional[float]:
    """R2 of an ordinary least squares fit of the target on `features`.

    None when the fit is not determined: no features, too few complete rows for
    the number of predictors, or a constant target.
    """
    if not features:
        return None

    frame = materialise_pandas(data, list(features) + [target_column]).dropna()
    if len(frame) <= len(features) + 1:
        return None

    X = frame[list(features)].to_numpy(dtype=float)
    y = frame[target_column].to_numpy(dtype=float)
    design = np.column_stack([X, np.ones(len(X))])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    total = ((y - y.mean()) ** 2).sum()
    if total <= 0:
        return None
    return float(1.0 - (residual ** 2).sum() / total)



def canonical_fold(X, y, entities, years, *, paradigm: str,
                   return_years: bool = False):
    """Check the fold each paradigm materialised, and index it positionally.

    Every paradigm applies the same policy -- drop rows with no target, order by
    entity then year -- in its own idiom: an ORDER BY in the SQL view, a Polars
    sort, a pandas sort after compute. Three implementations of one policy are
    three chances to disagree, and a disagreement here is not a small one: the
    models would be fitted on the same rows in different orders, which
    falsifies the bitwise claim for a reason that has nothing to do with the
    paradigms.

    The policy stays inside each engine, because performing it is part of what
    the benchmark measures. What moves here is the verification.

    Three things are checked, each of which has a distinct failure behind it:

      * lengths agree, and no target is missing. A filter applied in one
        paradigm and not another changes n, and n reaches the reported degrees
        of freedom.
      * (entity, year) pairs are unique. A join that multiplies rows produces
        exactly this, and nothing downstream would notice: the fit succeeds and
        the latency simply grows. This is L1.4 in Kapoor & Narayanan (2023) --
        duplicates in the dataset -- whose info sheet asks whether duplicates
        exist and how they are handled. Here the answer is derived rather than
        asserted: the run halts if any survive.
      * the order is non-decreasing by (entity, year) under Python comparison.
        The engines order under their own rules -- a database collation, a Rust
        string comparison -- and only agreement between them makes the
        comparison meaningful.

    The returned objects carry a positional index. Downstream alignment is
    positional, and a label index that survives this far is a hazard rather
    than information.
    """
    frames = {'X': X, 'y': y, 'entities': entities, 'years': years}
    lengths = {name: len(value) for name, value in frames.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"{paradigm}: the materialised fold has inconsistent lengths "
            f"{lengths}."
        )

    if len(X) == 0:
        raise ValueError(
            f"{paradigm}: the materialised fold is empty. There is nothing to "
            f"fit, and an empty fold reaches the reported n as a zero."
        )

    target = pd.Series(y).reset_index(drop=True)
    if bool(target.isna().any()):
        missing = int(target.isna().sum())
        raise ValueError(
            f"{paradigm}: {missing} of {len(target)} rows carry no target. "
            f"Rows without a target are dropped upstream in every paradigm; "
            f"their presence here means one of them stopped doing it."
        )

    entity_values = pd.Series(entities).reset_index(drop=True).to_numpy()
    year_values = pd.Series(years).reset_index(drop=True).to_numpy()

    duplicated = pd.MultiIndex.from_arrays(
        [entity_values, year_values]).duplicated()
    if duplicated.any():
        raise ValueError(
            f"{paradigm}: {int(duplicated.sum())} duplicated (entity, year) "
            f"pairs in the materialised fold. One row per entity and year is "
            f"the panel's shape; duplicates come from a join that multiplied "
            f"rows, and nothing downstream would notice."
        )

    order = np.lexsort((year_values, entity_values))
    if not np.array_equal(order, np.arange(len(order))):
        first = int(np.flatnonzero(order != np.arange(len(order)))[0])
        raise ValueError(
            f"{paradigm}: the materialised fold is not ordered by (entity, "
            f"year); the first row out of place is at position {first} "
            f"({entity_values[first]!r}, {year_values[first]!r}). The "
            f"paradigms must present the same rows in the same order, or the "
            f"models are fitted on different matrices."
        )

    ordered = (pd.DataFrame(X).reset_index(drop=True),
               target,
               pd.Series(entities).reset_index(drop=True))
    if not return_years:
        return ordered
    # Opt-in, and the default is off: fifty-odd call sites unpack three values,
    # and widening the tuple for all of them to serve one caller would be the
    # kind of change that breaks a paradigm quietly. The year is needed to key a
    # row -- an entity alone repeats across splits by design, which is the panel
    # structure and not a defect.
    return ordered + (pd.Series(years).reset_index(drop=True),)



def assert_splits_disjoint(splits, *, paradigm: str, injection=None) -> Dict:
    """No row of an evaluation split may also be a row of the training split.

    P1 and P2 impose discipline on the *windows*: the test period comes after
    the training period, with a gap. Nothing until now imposed it on the
    *rows*. Verified before this function existed: pasting rows from the test
    window into the training frame passes `canonical_fold`, passes the temporal
    gate, and passes every P1-P5 check, because the duplicate test is per split
    and the temporal gate reads the fold configuration rather than the data.

    That is memorisation leakage -- Class III in Roth's taxonomy, and the class
    where severity is known to scale with model capacity. A model that can
    memorise a row is handed the answer for a row it will be scored on, and no
    part of this framework noticed.

    The check is on (entity, year), because entity alone recurs across splits by
    construction: the same country appears in training and in test, and that is
    the panel structure rather than a defect. It is what K&N call L1.1, an
    evaluation set that is not independent, enforced at the granularity where
    the violation actually happens.

    Args:
        splits: mapping of split name to its (entities, years) pair. The
            training split must be present under the key 'train'.
        paradigm: named in the diagnostic, because the three materialise folds
            independently and a defect in one is not a defect in the others.
        injection: the declared violation for this arm, when there is one. An
            experiment that commits this violation on purpose has to be able to
            run; what it must not do is run indistinguishably from a clean one.
            So the waiver is passed in by the caller rather than read from the
            environment here, and the overlap it lets through comes back in the
            return value, bound for the receipt.

    Returns:
        A record of what was checked and what was waived. Empty of waivers on
        every production path, which is what makes a waived arm visible.
    """
    if 'train' not in splits:
        raise ValueError(
            f"{paradigm}: assert_splits_disjoint needs the training split to "
            f"compare the others against; got {sorted(splits)}")

    def keys(pair):
        entities, years = pair
        return set(zip((str(e) for e in pd.Series(entities)),
                       (int(y) for y in pd.Series(years))))

    training = keys(splits['train'])
    waived = []
    for name, pair in splits.items():
        if name == 'train':
            continue
        shared = training & keys(pair)
        if not shared:
            continue
        if injection is not None and injection.waives('L1.1'):
            waived.append({'split': name, 'overlapping_rows': len(shared),
                           'declared_by': injection.as_record()})
            continue
        sample = sorted(shared)[:5]
        raise AntiLeakageViolation(
            f"Anti-leakage violation (L1.1 evaluation independence): "
            f"{paradigm} has {len(shared)} row(s) present in both the "
            f"training split and '{name}'. The model would be scored on "
            f"rows it was fitted on. First few (entity, year): {sample}"
        )

    return {'gate': 'L1.1', 'splits_checked': sorted(splits), 'waived': waived}


def assert_lag_columns(present, paradigm: str, lag_orders, *,
                       target_stem: str) -> None:
    """The autoregressive columns exist, in every paradigm.

    Two of the three built them inside a try/except that printed a warning and
    returned the frame without them -- one of those catching bare Exception.
    A paradigm missing its lags trains on a different feature set from the
    other two, so the bitwise claim fails for a reason that has nothing to do
    with the paradigms, and the only trace is a line of stdout in a run that
    takes hours.

    Lags are not optional. Where the entity's past target was never observed
    the join yields NULL, which is the honest value and is handled downstream;
    a missing *column* is a different thing entirely.

    The stem is a required keyword rather than a default: this module holds the
    checks that are meant to outlive the study they were written for, and the
    name of one study's dependent variable was hardcoded here. A default would
    have kept it, one import away.
    """
    expected = {f'{target_stem}_lag_{order}' for order in lag_orders}
    missing = sorted(expected - set(present))
    if missing:
        raise ValueError(
            f"{paradigm}: the target's lag columns were not created "
            f"{missing}. Without them this paradigm trains on a feature set "
            f"different from the other two, and the comparison stops being "
            f"between paradigms."
        )


def redundant_features(data, features, target_column, tolerance):
    """Features that the others determine exactly.

    The same question the joint-reconstruction check asks about the target,
    asked about each feature: can the rest reproduce it to numerical
    precision? If so the design matrix is rank deficient and the feature count
    overstates the information in it.

    This is not leakage. Ridge absorbs it through regularisation and a forest
    never notices, so the predictions stand. What does not stand is the
    reported dimensionality, and any reading of a coefficient.

    The shape is not hypothetical here: INEP's rendimento rates are approval,
    failure and abandonment per level, and the three sum to a hundred by
    construction. Pairwise collinearity filtering does not have to catch it --
    with comparable variances each pair correlates around -0.5, under the
    ceiling, and all three survive while any two determine the third.
    """
    redundant = {}
    for feature in features:
        others = [other for other in features if other != feature]
        if not others:
            continue
        explained = linear_reconstruction_r2(data, others, feature)
        if explained is not None and explained > 1.0 - tolerance:
            redundant[feature] = float(explained)
    return redundant


#: Outcome of one check inside the audit. `RAN` and `NOT_APPLICABLE` are both
#: acceptable; `INDETERMINATE` is not, and used to be silent. A check whose
#: statistic could not be computed -- too few complete rows after listwise
#: deletion, an empty feature subset -- returned None and the audit moved on,
#: leaving a report indistinguishable from one where the check had passed.
RAN = 'ran'
NOT_APPLICABLE = 'not_applicable'
INDETERMINATE = 'indeterminate'


def audit_feature_set(
    X_train: pd.DataFrame, y_train: Any, *,
    autoregressive: Sequence[str],
    unaudited_by_selection: Sequence[str],
    config: Dict,
) -> Dict:
    """Apply the P3 checks to the matrix the model is about to fit.

    Feature selection audits what it produced. Columns appended afterwards --
    autoregressive lags of the target, added by the models -- never passed
    through it, so the set the models consume was never the set the gate saw.

    The arguments are the fitted design matrix and its target, not a panel and a
    list of names, and that is the point. Taking a frame plus column names left
    the *scope* to the caller, and the three paradigms chose differently: two
    audited the whole panel at setup, one audited each fold's training window.
    Same threshold, three frames, so the same check could abort exactly one
    paradigm in a study whose claim is that the three are equivalent -- the
    failure `canonical_fold` exists to prevent, reappearing inside a gate. With
    the matrix itself there is one thing the audit can mean.

    Auditing before scaling is deliberate and harmless: Pearson correlation and
    R2 are invariant under the affine transform a standard scaler applies.

    Three checks, and they are not interchangeable:

      * Do any columns that selection never saw behave as proxies? Only those
        columns are eligible: selection already applied this ceiling to what it
        chose, over the full panel, and aborts there. Re-measuring the same
        columns on a narrower frame cannot detect a proxy selection missed -- it
        can only disagree with itself through sampling noise. Judged at
        `proxy_correlation_threshold`.

      * Do the *non-autoregressive* columns jointly determine the target? That
        is the leakage question -- an additive identity that pairwise
        correlation cannot see, such as rates that sum to a constant. Selection
        asks this too, but only of the first fold's window, so later windows are
        genuinely uncovered. Judged at `identity_r2_threshold`.

      * Does the *whole set*, lags included, reproduce the target exactly? A
        genuine lag never does: y_t is not an exact linear function of y_{t-2}
        and y_{t-3}. An R2 at machine precision means a column labelled as
        lagged is carrying the contemporaneous value -- an off-by-one join, or a
        lag of zero. Judged against `target_reproduction_tolerance`, which is
        not a modelling choice but a numerical one. Selection never sees the
        lags, so this is the one check here that is nobody else's.

    Applying the 0.95 ceiling to the whole set conflated the last two. On an
    annual panel pooled across entities, a lag carries the entity's level and
    the pooled R2 is high by construction, so the check would abort a valid run
    for exhibiting the autocorrelation the task exists to exploit.

    Autoregressive columns are named rather than inferred. The exemption used to
    key off the substring `_lag_`, which would have silently excused any feature
    whose name happened to contain it, in a module meant to outlive the study it
    was written for.
    """
    # One accepted frame type, refused loudly rather than handled. The audit
    # previously took whatever each paradigm held -- a Polars frame, a Dask
    # collection, a pandas panel -- and the invariance of its verdict across
    # those was something a test had to assert. It is now structural: every
    # paradigm materialises the fold through canonical_fold before fitting, so
    # the matrix reaching here is pandas or the caller has skipped that step.
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            f"audit_feature_set expects the materialised design matrix, and "
            f"received {type(X_train).__name__}. Every paradigm passes the "
            f"fold through canonical_fold before fitting; a frame of another "
            f"type here means the audit is not seeing what the model fits."
        )

    proxy_threshold = float(config.get('proxy_correlation_threshold', 0.80))
    identity_threshold = float(config.get('identity_r2_threshold', 0.95))
    reproduction_tolerance = float(
        config.get('target_reproduction_tolerance', 1e-9))

    features = list(X_train.columns)
    autoregressive = [f for f in autoregressive if f in features]
    eligible = [f for f in unaudited_by_selection
                if f in features and f not in autoregressive]
    exogenous = [f for f in features if f not in autoregressive]

    #: A name the design matrix cannot already carry.
    target = '__rampart_target__'
    if target in features:
        raise ValueError(f"the design matrix carries the reserved name {target}")
    frame = X_train.copy()
    frame[target] = np.asarray(y_train, dtype=float)

    status: Dict[str, str] = {}

    correlations = {}
    for feature in features:
        pair = frame[[feature, target]].dropna()
        if len(pair) < 3 or pair[feature].nunique() < 2:
            continue
        correlations[feature] = float(abs(pair.corr().iloc[0, 1]))

    if not eligible:
        # The expected state: everything the model loads either came through
        # selection or is a declared lag. Recorded rather than skipped, so the
        # report says the domain was empty instead of saying nothing.
        status['proxy_ceiling'] = NOT_APPLICABLE
    elif any(f not in correlations for f in eligible):
        status['proxy_ceiling'] = INDETERMINATE
    else:
        status['proxy_ceiling'] = RAN
        proxies = {f: correlations[f] for f in eligible
                   if correlations[f] > proxy_threshold}
        if proxies:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P3 proxy detection): columns that "
                f"feature selection never audited show |correlation| > "
                f"{proxy_threshold} with the target "
                f"(Kapoor & Narayanan, 2023): {proxies}"
            )

    identity_r2 = linear_reconstruction_r2(frame, exogenous, target)
    if not exogenous:
        # Never benign: a model training on the target's own past alone.
        raise AntiLeakageViolation(
            f"Anti-leakage violation (P3 joint reconstruction): the design "
            f"matrix carries no non-autoregressive column, so the model would "
            f"train on the target's own past alone and the check has nothing "
            f"to evaluate. Columns present: {sorted(features)}"
        )
    if identity_r2 is None:
        status['joint_reconstruction'] = INDETERMINATE
    else:
        status['joint_reconstruction'] = RAN
        if identity_r2 > identity_threshold:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P3 joint reconstruction): the "
                f"non-autoregressive features explain the target with R2 = "
                f"{identity_r2:.4f} > {identity_threshold}, indicating the "
                f"target is an algebraic function of them: {sorted(exogenous)}"
            )

    reproduction_r2 = linear_reconstruction_r2(frame, features, target)
    if reproduction_r2 is None:
        status['target_reproduction'] = INDETERMINATE
    else:
        status['target_reproduction'] = RAN
        if reproduction_r2 > 1.0 - reproduction_tolerance:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P3 target reproduction): R2 = "
                f"{reproduction_r2:.12f} reproduces the target to numerical "
                f"precision. No genuine lag does this; a column labelled as "
                f"lagged is carrying the contemporaneous value: "
                f"{sorted(features)}"
            )

    # Design matrix rank. Reported, never halts: an exact dependency between
    # features does not inflate performance, and aborting here would kill a
    # valid run over a property regularisation absorbs.
    #
    # Two measures, because they answer different things. The rank says how many
    # independent directions there actually are; the redundant list says which
    # columns the others determine, to name the culprits. Neither follows from
    # the other: three rates summing to a hundred have one dependency -- rank
    # three counting the intercept, deficiency one -- and yet each of the three
    # is individually determined by the other two.
    redundant = redundant_features(frame, features, target,
                                   reproduction_tolerance)
    design = X_train.dropna()
    if len(design) > len(features):
        with_intercept = np.column_stack(
            [design.to_numpy(dtype=float), np.ones(len(design))])
        design_rank = int(np.linalg.matrix_rank(with_intercept))
        deficiency = (len(features) + 1) - design_rank
    else:
        design_rank = None
        deficiency = None

    return {
        # Sorted, so two paradigms auditing the same set produce comparable
        # reports; the order the model actually saw is kept beside it.
        'features_audited': sorted(features),
        'feature_order': features,
        'target_column': getattr(y_train, 'name', None),
        'audited_rows': int(len(X_train)),
        # Which checks actually executed. The gate reads this: a report where a
        # check came out indeterminate is not a report that the check passed.
        'checks': status,
        'unaudited_by_selection': sorted(eligible),
        'redundant_features': redundant,
        'design_rank': design_rank,
        'rank_deficiency': deficiency,
        'proxy_correlation_threshold': proxy_threshold,
        'identity_r2_threshold': identity_threshold,
        # Over the non-autoregressive features: the leakage question.
        'joint_reconstruction_r2': identity_r2,
        # Over the whole set: only exact reproduction is a defect here.
        'full_set_reconstruction_r2': reproduction_r2,
        'target_reproduction_tolerance': reproduction_tolerance,
        'autoregressive_exemptions': {
            f: correlations[f] for f in autoregressive if f in correlations
        },
        # Reported so the receipt carries the strongest non-autoregressive
        # association even when the ceiling had no eligible column to judge.
        'max_nonautoregressive_correlation': (
            max((correlations[f] for f in exogenous if f in correlations),
                default=None)),
    }


class TemporalValidator:
    """
    Temporal validator for leakage prevention in time series.

    Implements validation of temporal splits with mandatory gaps
    and a configurable embargo to ensure scientific validity in educational
    dropout forecasting.

    The protocol combines two complementary mechanisms:
      - **Temporal gap**: minimum period between consecutive splits,
        preventing future information from influencing training.
      - **Embargo**: an increment required of the gap, not an exclusion of
        observations. López de Prado (2018) describes the embargo as the
        removal of the training observations adjacent to the boundary of each
        split; here the validator only checks that the declared gap covers the
        declared embargo, and fails the fold when it does not. Nothing is
        removed -- what removes observations is the gap, in fold generation.

        The distinction matters because the two formulations differ when the
        gap is not uniform. With a constant gap of two years and one point per
        entity/year, requiring gap >= embargo and excluding `embargo` adjacent
        observations select the same training set, and that is why the check
        suffices on this panel.

    Note on purging (López de Prado 2018):
        Purging removes training observations whose labels overlap
        temporally with the test period. In data with annual
        granularity (one point per country/year), there is no label overlap
        between splits — each observation is a discrete point. Therefore,
        purging is unnecessary in this context. The temporal gap of N
        years already subsumes the effect of the embargo for annual data,
        since there are no intermediate sub-annual observations to exclude.
        The embargo_years parameter exists for use in adaptations of the
        framework to higher-frequency data (monthly, daily).
    """

    def __init__(self, min_gap_years: int = 2, embargo_years: int = 0):
        """
        Initialise the temporal validator.

        Args:
            min_gap_years: Minimum gap in years between splits (default: 2).
                Controls the mandatory temporal separation between periods.
            embargo_years: Additional embargo period in years (default: 0).
                When > 0, observations in the interval [train_end+1,
                train_end+embargo] are excluded from training, even if
                they are already outside the training split. Prevents leakage
                from residual autocorrelation in data with temporal
                dependency (lagged features, moving averages).
        """
        self.min_gap_years = min_gap_years
        self.embargo_years = embargo_years
    
    def validate_fold_integrity(self, fold: Dict) -> Tuple[bool, List[str]]:
        """
        Validate the full integrity of a temporal fold.
        
        Args:
            fold: Dictionary with the fold configuration
            
        Returns:
            Tuple (is_valid, list_of_errors)
        """
        errors = []
        
        # Check mandatory fields
        required_fields = [
            'train_start', 'train_end', 'val_start', 'val_end',
            'test_start', 'test_end'
        ]
        
        for field in required_fields:
            if field not in fold:
                errors.append(f"Missing mandatory field: {field}")
        
        if errors:
            return False, errors
        
        # Check chronological order
        if fold['train_start'] > fold['train_end']:
            errors.append(f"Train: start ({fold['train_start']}) > end ({fold['train_end']})")
        
        if fold['val_start'] > fold['val_end']:
            errors.append(f"Val: start ({fold['val_start']}) > end ({fold['val_end']})")
        
        if fold['test_start'] > fold['test_end']:
            errors.append(f"Test: start ({fold['test_start']}) > end ({fold['test_end']})")
        
        # Check temporal sequence
        if fold['train_end'] >= fold['val_start']:
            errors.append(f"Train-val overlap: train_end={fold['train_end']}, val_start={fold['val_start']}")
        
        if fold['val_end'] >= fold['test_start']:
            errors.append(f"Val-test overlap: val_end={fold['val_end']}, test_start={fold['test_start']}")
        
        # Check minimum gaps
        train_val_gap = fold['val_start'] - fold['train_end'] - 1
        val_test_gap = fold['test_start'] - fold['val_end'] - 1

        if train_val_gap < self.min_gap_years:
            errors.append(f"Insufficient train-val gap: {train_val_gap} < {self.min_gap_years}")

        if val_test_gap < self.min_gap_years:
            errors.append(f"Insufficient val-test gap: {val_test_gap} < {self.min_gap_years}")

        # Check the embargo: the effective gap must also cover the embargo
        if self.embargo_years > 0:
            effective_gap_tv = train_val_gap - self.embargo_years
            effective_gap_vt = val_test_gap - self.embargo_years
            if effective_gap_tv < 0:
                errors.append(
                    f"Train-val embargo violated: gap={train_val_gap} < "
                    f"embargo={self.embargo_years}"
                )
            if effective_gap_vt < 0:
                errors.append(
                    f"Val-test embargo violated: gap={val_test_gap} < "
                    f"embargo={self.embargo_years}"
                )

        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_walk_forward(self, folds: List[Dict]) -> Tuple[bool, Dict]:
        """
        Validate the walk-forward structure of multiple folds.
        
        Args:
            folds: List of folds to validate
            
        Returns:
            Tuple (is_valid, detailed_report)
        """
        report = {
            'total_folds': len(folds),
            'valid_folds': 0,
            'invalid_folds': 0,
            'fold_errors': {},
            'walk_forward_valid': True,
            'expanding_window': True
        }
        
        for i, fold in enumerate(folds):
            is_valid, errors = self.validate_fold_integrity(fold)
            
            if is_valid:
                report['valid_folds'] += 1
            else:
                report['invalid_folds'] += 1
                report['fold_errors'][f'fold_{i}'] = errors
        
        # Check whether it is an expanding walk-forward
        if len(folds) > 1:
            for i in range(1, len(folds)):
                # Train must expand or stay the same
                if folds[i]['train_end'] < folds[i-1]['train_end']:
                    report['expanding_window'] = False
                    report['walk_forward_valid'] = False
                    break
        
        # A structural walk-forward violation is recorded above and must
        # reach the verdict; counting invalid folds alone would let an
        # invalid fold sequence pass with every fold individually valid.
        report['all_valid'] = (
            report['invalid_folds'] == 0 and report['walk_forward_valid']
        )

        return report['all_valid'], report

    def enforce_walk_forward(self, folds: List[Dict]) -> None:
        """
        Validate the walk-forward structure and halt execution on violation.

        Raises:
            ValueError: If any fold violates temporal integrity
        """
        # An empty set satisfies "no invalid fold" vacuously, and the
        # pipeline used to record "0 folds -- temporal integrity verified".
        # Zero folds means the models had nothing to train on, or that the
        # artifact is broken; in neither case is there any integrity to
        # attest to.
        if not folds:
            raise AntiLeakageViolation(
                "Anti-leakage violation: the fold configuration is empty. "
                "There is no temporal integrity to attest to, and the models "
                "had nothing to train on."
            )

        all_valid, report = self.validate_walk_forward(folds)
        if not all_valid:
            errors = report.get('fold_errors', {})
            raise AntiLeakageViolation(
                f"Anti-leakage violation: {report['invalid_folds']} of "
                f"{report['total_folds']} folds failed temporal integrity. "
                f"Errors: {errors}"
            )
    


class DataIntegrityValidator:
    """
    Data integrity validator for ML.
    
    Checks the quality, completeness and consistency of the data
    before model training.
    """
    
    def validate_target_distribution(self, target_values: np.ndarray,
                                    expected_range: Tuple[float, float] = (0, 100),
                                    name: str = "target") -> Dict:
        """
        Validate the distribution of the target variable.
        
        Args:
            target_values: Target values
            expected_range: Expected range (min, max)
            name: Variable name for the report
            
        Returns:
            Dictionary with the distribution analysis
        """
        # Remove NaN for the analysis
        clean_values = target_values[~np.isnan(target_values)]
        
        validation = {
            'variable': name,
            'total_observations': len(target_values),
            'valid_observations': len(clean_values),
            'missing_count': len(target_values) - len(clean_values),
            'missing_rate': (len(target_values) - len(clean_values)) / len(target_values) * 100
        }
        
        if len(clean_values) > 0:
            validation.update({
                'mean': float(np.mean(clean_values)),
                'std': float(np.std(clean_values)),
                'min': float(np.min(clean_values)),
                'max': float(np.max(clean_values)),
                'median': float(np.median(clean_values)),
                'q25': float(np.percentile(clean_values, 25)),
                'q75': float(np.percentile(clean_values, 75))
            })
            
            # Check the range
            out_of_range = np.sum((clean_values < expected_range[0]) | 
                                 (clean_values > expected_range[1]))
            validation['out_of_range_count'] = int(out_of_range)
            validation['out_of_range_rate'] = float(out_of_range / len(clean_values) * 100)
            
            negative_count = np.sum(clean_values < 0)
            validation['negative_values'] = int(negative_count)
            
            # Alerts
            validation['warnings'] = []
            
            if validation['missing_rate'] > 20:
                validation['warnings'].append(f"High missing rate: {validation['missing_rate']:.1f}%")
            
            if validation['out_of_range_rate'] > 5:
                validation['warnings'].append(f"Values out of range: {validation['out_of_range_rate']:.1f}%")
            
            if negative_count > 0:
                validation['warnings'].append(f"Negative values detected: {negative_count}")
            
            if validation['std'] < 1:
                validation['warnings'].append(f"Low variability: std={validation['std']:.2f}")
        else:
            validation['warnings'] = ["No valid data for analysis"]
        
        validation['is_valid'] = len(validation.get('warnings', [])) == 0
        
        return validation
    
    def validate_dataframe(self, df: pd.DataFrame,
                          target_col: str = None,
                          check_completeness: bool = True) -> Tuple[bool, Dict]:
        """
        Validate the full integrity of a DataFrame.
        
        Args:
            df: DataFrame to validate
            target_col: Name of the target column (optional)
            check_completeness: Whether to check completeness
            
        Returns:
            Tuple (is_valid, validation_report)
        """
        validation_report = {
            'is_valid': True,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_data': {},
            'warnings': [],
            'errors': []
        }
        
        if df.empty:
            validation_report['is_valid'] = False
            validation_report['errors'].append("DataFrame is empty")
            return False, validation_report
        
        missing_counts = df.isnull().sum()
        missing_rates = (missing_counts / len(df)) * 100
        
        for col in df.columns:
            if missing_counts[col] > 0:
                validation_report['missing_data'][col] = {
                    'count': int(missing_counts[col]),
                    'rate': float(missing_rates[col])
                }
                
                # If completeness is required
                if check_completeness and missing_rates[col] > 50:
                    validation_report['warnings'].append(
                        f"Column '{col}' has {missing_rates[col]:.1f}% missing data"
                    )
        
        # Validate the target if specified
        if target_col and target_col in df.columns:
            target_validation = self.validate_target_distribution(
                df[target_col].values,
                name=target_col
            )
            validation_report['target_validation'] = target_validation
            
            if not target_validation['is_valid']:
                validation_report['warnings'].extend(target_validation.get('warnings', []))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].var() == 0:
                validation_report['warnings'].append(f"Column '{col}' has zero variance")
        
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        for col, count in inf_counts.items():
            if count > 0:
                validation_report['warnings'].append(f"Column '{col}' has {count} infinite values")
                validation_report['is_valid'] = False
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation_report['warnings'].append(f"DataFrame has {duplicates} duplicated rows")
        
        # Determine final validity
        if validation_report['errors']:
            validation_report['is_valid'] = False
        
        # Heuristic: more than MAX_TOLERABLE_WARNINGS indicates a degraded dataset
        MAX_TOLERABLE_WARNINGS = 5
        if len(validation_report['warnings']) > MAX_TOLERABLE_WARNINGS:
            validation_report['is_valid'] = False
            validation_report['errors'].append(
                f"Number of warnings ({len(validation_report['warnings'])}) "
                f"exceeds the tolerable limit ({MAX_TOLERABLE_WARNINGS})"
            )
        
        return validation_report['is_valid'], validation_report
    


def scale_from_training_window(train: pd.DataFrame, *apply_to: pd.DataFrame,
                               fit_on: Optional[pd.DataFrame] = None
                               ) -> Tuple[List[pd.DataFrame], Dict]:
    """Standardise on statistics fitted in the training window, and say so.

    The other half of P5, and on these panels the half that does the work.
    Measured on the World Bank collection: zero missing cells, so the imputation
    this sits next to has nothing to fill, while the scaler touches every row of
    every column. The protocol's receipt covered the statistic with no work and
    left uncovered the one that does all of it.

    Was written out three times, identically, one per paradigm -- the shape this
    repository keeps finding, where a policy in several places has already
    diverged or is waiting to. Here it had not diverged yet; consolidating is
    what stops it from being a matter of luck.

    `fit_on` widens the frame the statistics come from, and exists so that an
    experiment can commit the violation deliberately. Left at None, which is
    every production path, the statistics come from `train` and the behaviour is
    what it was.
    """
    from sklearn.preprocessing import StandardScaler

    source = train if fit_on is None else fit_on
    scaler = StandardScaler()
    scaler.fit(source)

    def applied(frame):
        return pd.DataFrame(scaler.transform(frame), columns=frame.columns,
                            index=frame.index)

    report = {
        'fitted_on_rows': int(len(source)),
        # The span the statistics came from. A frame widened past the split
        # shows here as a range reaching into the evaluation years, which is the
        # only way the receipt can tell a clean fit from a contaminated one.
        'fitted_on_year_range': (
            [int(source['year'].min()), int(source['year'].max())]
            if 'year' in getattr(source, 'columns', []) else None),
        'fitted_beyond_training_window': fit_on is not None,
    }
    return [applied(train)] + [applied(frame) for frame in apply_to], report


def impute_from_training_window(train: pd.DataFrame, *apply_to: pd.DataFrame,
                               strategy: str = 'median',
                               fit_on: Optional[pd.DataFrame] = None
                               ) -> Tuple[List[pd.DataFrame], Dict]:
    """Fill missing feature values with statistics fitted on the training window.

    P5 (preprocessing scope; Kaufman et al., 2012) requires that any fitted
    statistic come from training data alone. The collection stage previously
    imputed with the mean of stratum peers in the same year and with the mean of
    the whole panel across all years -- statistics computed over validation and
    test periods and written into training cells, before folds existed, where the
    P1-P5 gates could not reach.

    Forward fill within an entity needs no fitted statistic, so it stays in
    collection and is P5-safe by construction. Everything that needs a statistic
    happens here, once, for every paradigm: three implementations of this would be
    three chances for the paradigms to preprocess differently, and the equivalence
    claim assumes they do not.

    A column with no observed value in the training window raises. It cannot occur
    while the invariants hold: the training window is expansive (train_start is
    fixed at the start year), and feature selection runs on the first fold's
    training window under P4, so a feature that was selected has observations in
    that window and therefore in every later one. If it occurs, an invariant broke.

    The three alternatives are all worse. Filling a constant fabricates a value the
    training window never observed, and makes the feature constant in training and
    variable in test -- a distribution shift introduced by preprocessing itself.
    Dropping the column changes the feature set between folds and possibly between
    paradigms, which breaks both cross-fold comparability and the equivalence
    claim. Leaving the value missing defers the failure to RidgeCV, since
    StandardScaler propagates NaN silently rather than rejecting it.

    Args:
        train: the fold's training frame; the only source of statistics
        apply_to: further frames (validation, test) receiving the same values
        strategy: 'median' (default, resistant to outliers) or 'mean'

    Returns:
        ([train, *apply_to] filled, report) with the fitted value per column and
        the columns left untouched.
    """
    if strategy not in ('median', 'mean'):
        raise ValueError(f"unsupported strategy: {strategy}")

    # Where the statistics come from. `train` on every production path; a
    # widened frame only when an arm declares the violation, so that the report
    # below records a fit window reaching into the evaluation years.
    source = train if fit_on is None else fit_on

    fitted: Dict[str, float] = {}
    unobserved: List[str] = []
    for column in train.columns:
        observed = source[column].dropna() if column in source else train[column].dropna()
        if observed.empty:
            unobserved.append(column)
            continue
        fitted[column] = float(observed.median() if strategy == 'median'
                               else observed.mean())

    if unobserved:
        raise ValueError(
            f"Features with no observation in the training window: "
            f"{sorted(unobserved)}. With an expansive window and selection "
            f"under P4 on the first fold's window, this cannot occur: a "
            f"selected feature has data there and therefore in every later "
            f"window. Filling with a constant would fabricate a value the "
            f"training window never observed."
        )

    # Counted per split, because how much of each window is fabricated is what
    # a reader needs and the report carried only the fitted values. The extent
    # of fold-level imputation appeared in no artifact at all.
    split_names = ['train'] + [f'apply_{index}'
                               for index in range(len(apply_to))]
    filled = []
    filled_cells = {}
    for name, frame in zip(split_names, (train, *apply_to)):
        out = frame.copy()
        per_column = {}
        for column, value in fitted.items():
            if column not in out.columns:
                continue
            missing = int(out[column].isna().sum())
            if missing:
                per_column[column] = missing
            out[column] = out[column].fillna(value)
        filled_cells[name] = {
            'rows': int(len(out)),
            'by_column': per_column,
            'total': int(sum(per_column.values())),
        }
        filled.append(out)

    report = {
        'strategy': strategy,
        'fitted_on_rows': int(len(source)),
        # What the statistics were fitted over, and not merely how many rows.
        # The receipt attests that the imputation ran; without the span it
        # cannot attest that it ran on the training window, which is the whole
        # of P5. A frame widened past the split shows up here as a range that
        # reaches into the evaluation years.
        'fitted_on_year_range': (
            [int(source['year'].min()), int(source['year'].max())]
            if 'year' in getattr(source, 'columns', []) else None),
        'fitted_beyond_training_window': fit_on is not None,
        'filled_cells': filled_cells,
        'values': fitted,
        # Kept, always empty: the condition raises above. The key remains so
        # that an old artifact and a new one are comparable.
        'columns_without_training_observation': [],
    }
    return filled, report
