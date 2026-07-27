#!/usr/bin/env python3
"""
Abstract base class for ML architectures.

This module defines the common structure for all ML architectures,
ensuring methodological consistency and easing maintenance without duplication.
It preserves the logic of each architecture.
"""

from abc import ABC, abstractmethod
import os
import json
import math
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.scientific_config import SCIENTIFIC_CONFIG, setup_reproducibility
from core.validation import AntiLeakageViolation

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


class BaseArchitectureML(ABC):
    """
    Abstract base class for Machine Learning architectures.

    Defines the common structure and the methods shared across different
    architectures (Data Lake, Data Warehouse), ensuring methodological
    consistency and eliminating code duplication.

    Anti-leakage protocol (P1-P5):
        P1 — Temporal ordering: train < val < test strictly.
        P2 — Minimum gap: N years between consecutive splits (default 2).
        P3 — Feature separation: exclusion of target-derived columns
             and proxy detection (|correlation| > threshold).
        P4 — Temporal scope of selection: feature selection restricted
             to the training period of the first fold (Kapoor & Narayanan, 2023).
        P5 — Preprocessing scope: statistical transformations
             (scaling, imputation) fitted exclusively on the training data
             (Kaufman et al. 2012).

    HPO strategy:
        Hyperparameters are selected via grid search on the validation
        set, never on the test set. The final model is retrained on the
        full training window with the selected hyperparameters. This
        prevents leakage from optimization on the test set (Kapoor & Narayanan, 2023).

    Attributes:
        architecture_name: Architecture name (task_graph, sql_engine)
        output_base: Base directory for outputs
        prep_dir: Preparation directory
        target_column: Name of the created target column
        source_column: Source column used to build the target
    """

    _registry: Dict[str, type] = {}

    # Stem of the target-derived names: each paradigm's target
    # (TARGET_STEM_<paradigm>) and its lags (TARGET_STEM_lag_<k>).
    TARGET_STEM = 'dropout_rate'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register only fully concrete classes (with no abstract methods left).
        # CPython implementation detail: ABCMeta.__new__ fires __init_subclass__
        # (via type.__new__) before computing __abstractmethods__. For that reason,
        # we recompute the pending abstracts by walking the MRO manually.
        # Assumption: every override of an abstract method appears in the __dict__
        # of some class in the MRO. Dynamic attribute protocols (__getattr__)
        # are not supported.
        pending: set = set()
        for klass in reversed(cls.__mro__):
            for name, val in klass.__dict__.items():
                if getattr(val, '__isabstractmethod__', False):
                    pending.add(name)
                elif name in pending:
                    pending.discard(name)
        if pending:
            return
        # Opt-in registration: only classes that explicitly define PARADIGM_META
        # in their own __dict__ are treated as paradigms. Concrete subclasses
        # without PARADIGM_META (helpers, test stubs) are silently ignored.
        if 'PARADIGM_META' not in cls.__dict__:
            return
        meta = cls.PARADIGM_META
        if not meta.get('name'):
            raise TypeError(
                f"{cls.__name__} defines PARADIGM_META but is missing "
                f"the mandatory key 'name'."
            )
        existing = BaseArchitectureML._registry.get(meta['name'])
        if existing is not None:
            same_source = (
                existing.__qualname__ == cls.__qualname__
                and existing.__name__ == cls.__name__
            )
            if not same_source:
                raise TypeError(
                    f"The paradigm name '{meta['name']}' is already registered by "
                    f"{existing.__name__}. {cls.__name__} cannot reuse it."
                )
        BaseArchitectureML._registry[meta['name']] = cls

    @classmethod
    def get_registered_paradigms(cls) -> Dict[str, type]:
        """Return every registered concrete paradigm class."""
        return dict(cls._registry)

    #: Fragments that give away a collection-metadata column instead of an
    #: indicator of the phenomenon. Metadata as a predictor turns the sampling
    #: process into a feature: a completeness score, for example, correlates
    #: with statistical capacity, which correlates with the target -- the model
    #: would learn to predict from how well the data were collected. This
    #: detects, it does not filter: the exclusion still belongs to the
    #: configuration, and what changes is that forgetting it no longer goes
    #: unnoticed.
    METADATA_NAME_FRAGMENTS = ('timestamp', 'batch_id', 'completeness',
                               'synthetic', 'data_source', 'partition',
                               'processing_method', '_flag', 'etl_', 'ingest')

    #: Target lag orders that the three paradigms build. Declared here because
    #: the fold metadata needs to know the most recent value a model consults
    #: at prediction time, and because three independent implementations
    #: building different lags would break Δ=0 with nothing flagging it. A test
    #: checks all three against this list.
    TARGET_LAG_ORDERS = (2, 3)

    def __init__(self, architecture_name: str, output_base_path: str,
                 dataset_config=None):
        """
        Initialize the base architecture.

        Args:
            architecture_name: Architecture identifier
            output_base_path: Base path for outputs
            dataset_config: DatasetConfig (default: worldbank)
        """
        self.architecture_name = architecture_name
        self.output_base = output_base_path
        self.prep_dir = f"{self.output_base}/prep"

        # Centralized scientific configuration
        self.config = SCIENTIFIC_CONFIG
        setup_reproducibility()

        # Dataset config (lazy import, detected via env var for subprocesses)
        if dataset_config is None:
            dataset_name = os.environ.get('DATASET_NAME', 'worldbank')
            if dataset_name == 'inep_censo':
                from datasets.inep_censo import InepCensoDatasetConfig
                dataset_config = InepCensoDatasetConfig()
            else:
                from datasets.worldbank import WorldBankDatasetConfig
                dataset_config = WorldBankDatasetConfig()
        self.dataset_config = dataset_config

        # Target configuration (derived from the dataset)
        self.target_column = f"{self.TARGET_STEM}_{architecture_name}"
        self.source_column = dataset_config.target_source_column
        
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """Create the required directory structure."""
        os.makedirs(self.prep_dir, exist_ok=True)
        os.makedirs(f"{self.prep_dir}/folds", exist_ok=True)
        
    @abstractmethod
    def setup_environment(self) -> None:
        """
        Configure the architecture-specific environment.

        Abstract method that each architecture must implement in order
        to configure its own environment (Dask, SQL, etc.).
        """
        pass
    
    @abstractmethod
    def load_data(self) -> Any:
        """
        Load the architecture-specific data.

        Returns:
            Data loaded in the architecture's own format
            (dd.DataFrame for Dask, SQL connection for DW, etc.)
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> None:
        """
        Validate data integrity.

        Args:
            data: Data to validate, in the architecture's format

        Raises:
            ValueError: When validation fails
        """
        pass
    
    @abstractmethod
    def create_target_implementation(self, data: Any) -> Any:
        """
        Architecture-specific implementation that creates the target variable.

        Args:
            data: Input data

        Returns:
            Data with the target variable created
        """
        pass
    
    def create_target(self, data: Any) -> Any:
        """
        Create the target variable with the shared scientific validation.

        This method implements the common target-creation logic
        (dropout_rate = 100 - completion_rate) with scientific validations
        that are identical across all architectures.

        Args:
            data: Input data in the architecture's format

        Returns:
            Data with the target variable created and validated

        Simple inversion --- raw_data_collector guarantees the range [0,100].
        """
        print(f"\nCreating target {self.architecture_name}: {self.source_column} -> {self.target_column}")
        
        data_with_target = self.create_target_implementation(data)
        
        self._save_target_statistics(data_with_target)
        
        return data_with_target
    
    @abstractmethod
    def _compute_target_statistics(self, data: Any) -> Dict[str, float]:
        """
        Compute the architecture-specific target statistics.

        Args:
            data: Data with the target already created

        Returns:
            Dictionary with statistics (mean, std, min, max, etc.)
        """
        pass
    
    def _save_target_statistics(self, data: Any) -> None:
        """
        Save the target statistics in a standardized way.

        Args:
            data: Data with the target, used to compute the statistics
        """
        stats = self._compute_target_statistics(data)
        
        stats.update({
            'architecture': self.architecture_name,
            'target_variable': self.target_column,
            'source_column': self.source_column,
            'creation_timestamp': datetime.now().isoformat()
        })
        
        expected_range = list(self.dataset_config.target_expected_range)
        if stats['mean'] < expected_range[0] or stats['mean'] > expected_range[1]:
            print(f"   Warning: Mean dropout ({stats['mean']:.2f}%) "
                  f"outside the expected range {expected_range}")

        if stats['valid_count'] < self.dataset_config.min_valid_count:
            print(f"   Warning: Too few valid records ({stats['valid_count']}) "
                  f"for ML")
        
        stats_path = f"{self.prep_dir}/target_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"   Statistics saved: {stats_path}")
    
    def create_temporal_folds(self, data: Any = None) -> List[Dict]:
        """
        Create scientific temporal folds with a walk-forward methodology.

        Implements a temporal validation structure that is identical for all
        architectures, ensuring scientific comparability.

        Args:
            data: Optional data used to validate the folds

        Returns:
            List of fold configurations with complete metadata

        Applies 2-year temporal gaps between train/val and val/test
        to prevent temporal leakage.
        """
        print("\nCreating temporal folds...")
        gap = int(self.config.get('temporal_gap_years', 2))
        embargo = int(self.config.get('embargo_years', 0))
        print(f"Methodology: Automatic walk-forward with gaps of {gap} years"
              + (f" and an embargo of {embargo} years" if embargo > 0 else ""))
        folds = self._generate_walkforward_folds_auto()

        # Enforce anti-leakage: stop on violation
        from core.validation import TemporalValidator
        validator = TemporalValidator(min_gap_years=gap, embargo_years=embargo)
        validator.enforce_walk_forward(folds)

        if data is not None:
            if hasattr(data, 'reset_index'):
                data = data.reset_index(drop=True)
            self._validate_temporal_folds(data, folds)

        return folds

    def _generate_walkforward_folds_auto(self) -> List[Dict]:
        """
        Generate expanding walk-forward folds automatically, respecting gaps and windows.

        Parameters are read from SCIENTIFIC_CONFIG (with safe defaults):
          - temporal_range_start / end
          - folds_min_train_years
          - folds_val_len_years
          - folds_test_len_years
          - temporal_gap_years
          - folds_step_years
          - folds_max (optional)
        """
        cfg = self.config
        # Override the temporal range and walk-forward from dataset_config when available
        ds = self.dataset_config
        start_year = int(ds.temporal_range[0]) if ds else int(cfg.get('temporal_range_start', 2000))
        end_year = int(ds.temporal_range[1]) if ds else int(cfg.get('temporal_range_end', 2023))
        wf = ds.walk_forward_config if ds else {}
        min_train = int(wf.get('min_train', cfg.get('folds_min_train_years', 8)))
        val_len = int(wf.get('val_len', cfg.get('folds_val_len_years', 2)))
        test_len = int(wf.get('test_len', cfg.get('folds_test_len_years', 2)))
        gap = int(cfg.get('temporal_gap_years', 2))
        step = int(wf.get('step', cfg.get('folds_step_years', 1)))
        max_folds = cfg.get('folds_max', None)
        try:
            max_folds = int(max_folds) if max_folds is not None else None
        except Exception:
            max_folds = None

        # Compute the start bounds for the test window
        # Derivation: val_start >= start_year + min_train + gap
        # test_start = val_start + val_len + gap
        # => test_start_min = start_year + min_train + val_len + 2*gap
        test_start_min = start_year + min_train + val_len + 2 * gap
        test_start_max = end_year - test_len + 1

        folds: List[Dict] = []
        fold_id = 0
        for test_start in range(test_start_min, test_start_max + 1, step):
            test_end = test_start + test_len - 1
            # Derive val_end and val_start from the gap
            val_end = test_start - gap - 1
            val_start = val_end - val_len + 1
            # Derive train_end and train_start
            train_end = val_start - gap - 1
            train_start = start_year

            # Validity checks
            if train_end < train_start:
                continue
            train_len = train_end - train_start + 1
            if train_len < min_train:
                continue
            if not (train_start <= train_end < val_start <= val_end < test_start <= test_end <= end_year):
                continue

            train_val_gap = val_start - train_end - 1
            val_test_gap = test_start - val_end - 1
            if train_val_gap < gap or val_test_gap < gap:
                continue

            fold = {
                'fold_id': fold_id,
                'architecture': self.architecture_name,
                'methodology': 'walk_forward_with_gaps_auto',
                'train_start': int(train_start), 'train_end': int(train_end),
                'train_gap_start': int(train_end + 1), 'train_gap_end': int(val_start - 1),
                'val_start': int(val_start), 'val_end': int(val_end),
                'val_gap_start': int(val_end + 1), 'val_gap_end': int(test_start - 1),
                'test_start': int(test_start), 'test_end': int(test_end),
                'total_train_years': int(train_len),
                'total_val_years': int(val_len),
                'total_test_years': int(test_len),
                'train_val_gap': int(train_val_gap),
                'val_test_gap': int(val_test_gap),
                # Separation between the last observation that enters *parameter
                # estimation* and the first observation evaluated.
                #
                # Recorded because it is larger than the declared gap, and by
                # decision: the model evaluated on the test set is fitted on the
                # training window only, and validation serves exclusively to
                # select hyperparameters. Refitting on train+validation would use
                # 25% more years and bring the origin closer, but would reduce
                # this separation to the minimum declared in P2 -- it would trade
                # safety margin in the anti-leakage guarantee for statistical
                # efficiency in a device whose accuracy is not the object of
                # study.
                #
                # It is not the information horizon. At prediction time the
                # model reads target lags, and the naive baseline reads the
                # history up to t minus the gap: for a test row at test_start,
                # the most recent value consulted is from test_start - min(lags).
                # The two numbers answer different questions, and the previous
                # field, on its own, was read as if it covered both.
                'fit_to_test_gap': int(test_start - train_end - 1),
                'information_horizon_years': int(min(self.TARGET_LAG_ORDERS)),
                'fit_window': 'train_only',
                'description': f'Walk-forward auto (gap={gap}y, val={val_len}y, test={test_len}y)',
                'forecast_horizon': '1-2 years ahead'
            }
            folds.append(fold)
            fold_id += 1
            if max_folds is not None and len(folds) >= max_folds:
                break

        if not folds:
            raise ValueError(
                f"No fold could be generated with the current parameters. "
                f"Adjust temporal_range_start/end, folds_min_train_years or the gaps. "
                f"Parameters: start={start_year}, end={end_year}, min_train={min_train}, "
                f"val_len={val_len}, test_len={test_len}, gap={gap}"
            )

        print(f"   Auto-generated folds: {len(folds)} (gap={gap}, val={val_len}, test={test_len})")
        return folds
    
    @abstractmethod
    def _validate_temporal_folds(self, data: Any, folds: List[Dict]) -> None:
        """
        Validate the scientific structure of the folds.

        Args:
            data: Data used for validation
            folds: List of folds to validate
        """
        pass
    
    @abstractmethod
    def save_folds(self, data: Any, folds: List[Dict]) -> None:
        """
        Save the folds in the architecture's own format.

        Args:
            data: Processed data
            folds: List of fold configurations
        """
        pass
    
    def _filter_by_year(self, data: Any, max_year: int) -> Any:
        """Filter data down to year <= max_year. Supports pandas, Dask and Polars."""
        if _HAS_POLARS and isinstance(data, pl.DataFrame):
            return data.filter(pl.col('year') <= max_year)
        elif hasattr(data, 'compute'):  # Dask DataFrame
            return data[data['year'] <= max_year]
        elif isinstance(data, pd.DataFrame):
            return data[data['year'] <= max_year]
        else:
            raise TypeError(f"Unsupported data type for temporal filtering: {type(data)}")

    @staticmethod
    def _count_rows(data: Any) -> int:
        """Count the rows of a DataFrame (pandas or Dask)."""
        if hasattr(data, 'compute'):  # Dask
            return len(data)
        return len(data)

    def _materialise_pandas(self, data: Any, columns: List[str]) -> pd.DataFrame:
        """Materialise the given columns as a pandas frame.

        Delegates to core.validation, which the model-level P3 audit also uses.
        There were two copies of this dispatch and they had already diverged:
        this one did not handle a Polars LazyFrame, so the same check raised on
        input the model-level one accepts.
        """
        from core.validation import materialise_pandas

        return materialise_pandas(data, columns)

    def _linear_reconstruction_r2(self, data: Any,
                                  features: List[str]) -> Optional[float]:
        """R2 of an ordinary least squares fit of the target on `features`.

        Delegates for the same reason: the setup-level and model-level joint
        reconstruction checks must answer identically on identical input.
        """
        from core.validation import linear_reconstruction_r2

        return linear_reconstruction_r2(data, features, self.target_column)

    def release_resources(self) -> None:
        """Release whatever this paradigm keeps open between runs.

        The benchmark re-executes each phase `warmup + n` times in the same
        process. A resource that survives one repetition is measured by the
        next: sql_engine used to leave one DuckDB connection open per
        repetition, twelve by the end, each with its own buffer pool -- so its
        later repetitions were measuring under conditions the other two never
        faced.

        The default is empty, and deliberately so: Polars keeps nothing, and
        the collections Dask persists are local and die with the scope. The
        contract exists so that release is symmetric across paradigms, not so
        that each one invents its own.
        """

    @staticmethod
    def reported_statistic(value) -> Optional[float]:
        """An undefined statistic comes out as null, not as zero.

        With a single observation the standard deviation does not exist; with
        the whole column missing, neither does the mean. DuckDB returns NULL,
        Polars returns None, pandas returns NaN -- and two of the three
        paradigms converted that into 0.0, which is a claim about the data:
        "there is no variation". The third wrote NaN, which is not even valid
        JSON under a strict parser. The three disagreed on the same degenerate
        input, in a published artifact.
        """
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def get_excluded_features(self) -> List[str]:
        """
        Return the list of features to exclude (leakage/metadata).

        The list is harmonized across all architectures to guarantee a fair
        scientific comparison.

        Returns:
            List of column names to exclude
        """
        base_excluded = list(self.dataset_config.excluded_columns)
        if self.target_column not in base_excluded:
            base_excluded.append(self.target_column)
        return base_excluded
    
    @abstractmethod
    def compute_feature_correlations(self, data: Any, features: List[str]) -> Dict[str, float]:
        """
        Compute the correlations between features and target.

        Args:
            data: Data containing the features
            features: List of features to analyze

        Returns:
            Dictionary of absolute correlations
        """
        pass
    
    def select_features_with_bounds(
            self, correlations: Dict[str, float]) -> Tuple[List[str], Dict]:
        """Candidates whose marginal association with the target is relevant and
        not suspicious, together with the bounds that decided it.

        Floor and ceiling answer different questions, and the previous version
        relaxed them together.

        The **floor** is relevance: below it a feature contributes noise. It is
        a modelling choice, and loosening it when the pool is thin is
        legitimate.

        The **ceiling** is validity: above it a feature is suspected of being
        the target under another name (Kapoor & Narayanan, 2023). Loosening it
        does not buy a better model, it buys a contaminated one. The previous
        relaxation swapped the band for a loose floor, so the branch that real
        runs take -- the pool is small -- admitted a feature with |r| = 0.99.
        What barred it was the proxy audit, downstream, and relying on that is
        relying on the second line because the first was removed.

        The comparison is on **absolute value**. The previous one was signed,
        so every negatively associated feature was discarded -- and in this
        domain those are the protective factors (GDP per capita, completion
        rate, enrolment) against dropout. Neither RidgeCV nor RandomForest
        cares about the direction of a marginal association: the coefficient
        absorbs it, and the tree does not see it. Discarding them threw away
        genuine signal and biased the set towards a single sign of association.

        Failing to reach the minimum number of features does not stop the run:
        the minimum is a pragmatic floor, and proceeding with four features
        that pass both criteria is defensible. Reaching zero does stop it,
        because then there is no model.
        """
        config = self.config
        ceiling = float(config['proxy_correlation_threshold'])
        floor = float(config['feature_selection_min_abs_correlation'])
        relaxed = float(config['feature_selection_relaxed_min_abs_correlation'])
        minimum = int(config['feature_selection_min_features'])

        # An undefined correlation is a feature that is constant over the
        # training window: with no variation there is no association to
        # measure. It is discarded -- which is right -- but it used to be
        # discarded silently, and anyone counting the candidates against the
        # selected ones would find an unexplained disappearance.
        undefined = sorted(feature for feature, correlation
                           in correlations.items()
                           if not np.isfinite(float(correlation)))

        def within(lower: float) -> List[str]:
            return sorted(
                feature for feature, correlation in correlations.items()
                if np.isfinite(float(correlation))
                and lower <= abs(float(correlation)) <= ceiling
            )

        selected = within(floor)
        effective_floor = floor
        print(f"   Features with |r| in [{floor}, {ceiling}]: {len(selected)}")

        if len(selected) < minimum and relaxed < floor:
            widened = within(relaxed)
            # Only the floor comes down, so the relaxed set contains the strict
            # one by construction -- no feature the ceiling barred can come back.
            if len(widened) > len(selected):
                selected, effective_floor = widened, relaxed
                print(f"   Floor lowered to {relaxed}: {len(selected)} features")

        if not selected:
            raise ValueError(
                f"No candidate with |r| in [{relaxed}, {ceiling}] against the "
                f"target on the training window"
                + (f" ({len(undefined)} with an undefined correlation: "
                   f"{undefined})" if undefined else "") + f". Below the floor the association is "
                f"conventionally negligible; above the ceiling the feature is "
                f"suspected of being the target under another name. Without "
                f"features there is no model, and proceeding would produce an "
                f"empty artifact."
            )

        bounds = {
            'abs_correlation_floor': effective_floor,
            'abs_correlation_ceiling': ceiling,
            'floor_was_relaxed': effective_floor != floor,
            'min_features_target': minimum,
            'features_selected': len(selected),
            'below_min_features': len(selected) < minimum,
            'undefined_correlation': undefined,
        }
        if bounds['below_min_features']:
            print(f"   [WARNING] {len(selected)} features, below the target of "
                  f"{minimum}. Recorded; the ceiling is not loosened to "
                  f"reach it.")
        return selected, bounds

    def select_features_by_correlation(
            self, correlations: Dict[str, float]) -> List[str]:
        """The list only. See select_features_with_bounds for the criterion."""
        return self.select_features_with_bounds(correlations)[0]

    @abstractmethod
    def apply_collinearity_filter(self, data: Any, features: List[str],
                                  threshold: float = 0.8) -> List[str]:
        """
        Remove multicollinearity via greedy pairwise-correlation filtering.

        For each candidate feature, computes the maximum absolute correlation
        with the features already selected. Rejects it if max |r| >= threshold.

        Args:
            data: Data to analyze
            features: Candidate features
            threshold: Pairwise correlation threshold for removal

        Returns:
            List of features after multicollinearity removal
        """
        pass
    
    def _first_fold_train_end(self) -> int:
        """
        Compute train_end of the first fold from the scientific config.

        Used to restrict feature selection to the training period, preventing
        temporal leakage (Kapoor & Narayanan, 2023): feature selection using
        data that belongs to validation/test.
        """
        cfg = self.config
        ds = self.dataset_config
        start_year = int(ds.temporal_range[0]) if ds else int(cfg.get('temporal_range_start', 2000))
        wf = ds.walk_forward_config if ds else {}
        min_train = int(wf.get('min_train', cfg.get('folds_min_train_years', 8)))
        val_len = int(wf.get('val_len', cfg.get('folds_val_len_years', 2)))
        gap = int(cfg.get('temporal_gap_years', 2))
        test_start_min = start_year + min_train + val_len + 2 * gap
        val_end = test_start_min - gap - 1
        val_start = val_end - val_len + 1
        train_end = val_start - gap - 1
        return train_end

    def run_feature_selection(self, data: Any) -> Dict:
        """
        Run the complete feature selection pipeline.

        Standardized pipeline with anti-leakage enforcement:
        1. Removes leakage/metadata features (P3)
        2. Restricts the data to the training period of the first fold (P4)
        3. Selects by moderate correlation with the target
        4. Removes multicollinearity via pairwise filtering
        5. Detects target proxy features (extended P3)

        Preprocessing (scaling, imputation) happens in prepare_features()
        and in the models, with P5 enforcement (preprocessing scope).

        P4 (Kapoor & Narayanan, 2023; Kaufman et al., 2012):
        Correlations are computed using only data up to train_end of the
        first fold, preventing information from the validation or test
        periods from influencing feature selection.

        Args:
            data: Data for the selection

        Returns:
            Dictionary with statistics and the selected features
        """
        print(f"\nFeature selection {self.architecture_name}...")

        exclude_cols = self.get_excluded_features()
        feature_cols = self.get_numeric_features(data)

        # P3: the target and the column it derives from cannot be candidates.
        # Checked over the pool, and not only over the final selection: the
        # proxy audit runs after selection, so a candidate that the correlation
        # ceiling discards never gets audited at all. That is how the target's
        # source column, with correlation -1.0, made it through the gate.
        # Redundant with the candidate policy by decision: a regression in that
        # policy shows up here as a halt, not as contamination.
        forbidden = {self.target_column, self.source_column} & set(feature_cols)
        if forbidden:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P3 data separation): candidate pool "
                f"contains the target or the column it is derived from: "
                f"{sorted(forbidden)}"
            )

        print(f"   {len(feature_cols)} candidates ({len(exclude_cols)} excluded)")

        # P4: Restrict to the training period to avoid leakage in the selection
        # (Kapoor & Narayanan, 2023; Kaufman et al., 2012)
        train_end = self._first_fold_train_end()
        data_train_only = self._filter_by_year(data, max_year=train_end)
        n_total = self._count_rows(data)
        n_train = self._count_rows(data_train_only)
        print(f"   P4: Correlations restricted to the training period "
              f"(≤{train_end}): {n_train}/{n_total} observations")

        # Correlation with the target (using training data only)
        correlations = self.compute_feature_correlations(data_train_only, feature_cols)
        selected_by_corr, selection_bounds = \
            self.select_features_with_bounds(correlations)

        # Pairwise collinearity filtering (using training data)
        final_features = self.apply_collinearity_filter(
            data_train_only, selected_by_corr,
            float(self.config['collinearity_threshold']))

        # P3: Enforce that no excluded/target-derived feature is in the selection
        leaked = set(final_features) & set(exclude_cols)
        if leaked:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P3 data separation): "
                f"excluded features found in final selection: {leaked}"
            )

        # P3 extended: proxy detection (Kapoor & Narayanan, 2023; Kaufman et
        # al., 2012).
        #
        # Selection and auditing serve different purposes and use different
        # data. Selection reads only the P4 window, because choosing features by
        # their agreement with future target values is look-ahead bias. The
        # audit below reads the full panel: a feature whose correlation clears
        # the threshold only outside the first training window is still a proxy,
        # and restricting the audit to that window is what let one through here.
        #
        # The audit may only abort, never filter. Aborting reports that the
        # design is invalid without letting full-panel information reach the
        # model; silently dropping the feature would.
        PROXY_THRESHOLD = float(self.config.get('proxy_correlation_threshold', 0.80))
        audit_correlations = self.compute_feature_correlations(data, final_features)
        proxies = {
            feat: corr for feat, corr in audit_correlations.items()
            if feat in final_features and abs(corr) > PROXY_THRESHOLD
        }
        if proxies:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P3 proxy detection): "
                f"features with |correlation| > {PROXY_THRESHOLD} with target "
                f"over the full panel suggest proxy leakage "
                f"(Kapoor & Narayanan, 2023): {proxies}"
            )

        # P3 extended: joint reconstruction of the target.
        #
        # Pairwise correlation cannot see an additive identity. Where the target
        # partitions into several features -- rates that sum to a constant, for
        # instance -- each one correlates weakly while together they determine
        # the target exactly. Fitted on the training window, so an exact
        # identity is detected without consulting the evaluation periods.
        IDENTITY_THRESHOLD = float(self.config.get('identity_r2_threshold', 0.95))
        identity_r2 = self._linear_reconstruction_r2(data_train_only, final_features)
        if identity_r2 is not None and identity_r2 > IDENTITY_THRESHOLD:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P3 joint reconstruction): selected "
                f"features explain the target with R2 = {identity_r2:.4f} > "
                f"{IDENTITY_THRESHOLD} on the training window, indicating the "
                f"target is an algebraic function of the feature set: "
                f"{sorted(final_features)}"
            )

        # Selection statistics
        selection_stats = {
            'architecture': self.architecture_name,
            'total_features_analyzed': len(feature_cols),
            'features_selected': len(final_features),
            'selection_method': 'correlation_pairwise_filter',
            # The bounds that decided, not just their outcome. A reader who
            # sees four features needs to know whether the floor was lowered
            # to get there, and that the ceiling was not.
            'selection_bounds': selection_bounds,
            'temporal_scope': f'train_only (≤{train_end})',
            'proxy_threshold': PROXY_THRESHOLD,
            'selected_features': final_features,
            'target_correlations': {
                feat: float(correlations.get(feat, 0))
                for feat in final_features
            },
            'selection_timestamp': datetime.now().isoformat(),
            # This is the artifact that decides what the model trains on, and
            # was the one file in the setup path no gate checked. A file from
            # another run would feed the models a different feature set, and the
            # three would agree with each other by reading the same stale file
            # -- so not even the equivalence gate would notice.
            'run_id': os.environ.get('RAMPART_RUN_ID'),
        }
        
        selection_path = f"{self.prep_dir}/feature_selection_{self.architecture_name}.json"
        with open(selection_path, 'w') as f:
            json.dump(selection_stats, f, indent=2)
        
        print(f"   Selected features: {len(final_features)} -> {selection_path}")
        
        return selection_stats
    
    @abstractmethod
    def discover_numeric_columns(self, data: Any) -> List[str]:
        """
        List the numeric-typed columns present in the data.

        Discovery only: each paradigm inspects the schema by its own means
        (catalog metadata, dtype inference). The policy of which columns are
        legitimate candidates does not belong here — it lives in
        candidate_exclusions(), which is the same for all paradigms.

        Args:
            data: Input data

        Returns:
            List of numeric column names, in any order
        """
        pass

    def candidate_exclusions(self) -> Tuple[set, str]:
        """
        Names and prefix that disqualify a column as a candidate.

        Derived from the configuration, not enumerated. An enumerated list ages
        silently: that is how the target's source column (correlation -1.0 with
        the target) got into one paradigm's pool and survived the P3 gate, only
        to be discarded by the selection's correlation ceiling.

        The target-derived prefix covers, in one go, this paradigm's target,
        the other paradigms' targets and the target lags — none of them is a
        candidate for selection.

        Returns:
            (names to exclude, prefix to exclude)
        """
        excluded = set(self.get_excluded_features())
        excluded.add(self.source_column)
        for attr in ('entity_column', 'entity_name_column',
                     'year_column', 'stratification_column'):
            name = getattr(self.dataset_config, attr, None)
            if name:
                excluded.add(name)
        return excluded, f'{self.TARGET_STEM}_'

    def get_numeric_features(self, data: Any) -> List[str]:
        """
        Pool of candidates for feature selection.

        Identical across paradigms by construction: a divergent pool would make
        the cross-paradigm comparison start from different search spaces.

        Args:
            data: Input data

        Returns:
            Sorted list of candidates
        """
        excluded, derived_prefix = self.candidate_exclusions()

        # A declared feature that matched the derived prefix would be discarded
        # silently, changing the result without warning.
        declared = set(getattr(self.dataset_config, 'feature_columns', None) or ())
        shadowed = {c for c in declared if c.startswith(derived_prefix)} - excluded
        if shadowed:
            raise ValueError(
                f"{self.architecture_name}: declared features collide with the "
                f"prefix reserved for target-derived columns "
                f"('{derived_prefix}'): {sorted(shadowed)}. Rename them or "
                f"change TARGET_STEM; leaving them would drop them silently."
            )

        candidates = sorted(
            col for col in self.discover_numeric_columns(data)
            if col not in excluded and not col.startswith(derived_prefix)
        )

        metadata = [column for column in candidates
                    if any(fragment in column.lower()
                           for fragment in self.METADATA_NAME_FRAGMENTS)]
        if metadata:
            raise ValueError(
                f"{self.architecture_name}: collection metadata columns "
                f"survived into the candidate pool: {sorted(metadata)}. "
                f"Using them as predictors turns the sampling process into a "
                f"feature. Add them to the dataset's excluded_columns."
            )

        if len(candidates) < 5:
            print(f"  [WARN] Few candidates ({len(candidates)}) may "
                  f"limit predictive capacity")
        if len(candidates) > 100:
            print(f"  [WARN] Many candidates ({len(candidates)}) require "
                  f"careful selection (curse of dimensionality)")

        return candidates


    @abstractmethod
    def prepare_features(self, data: Any, selected_features: List[str]) -> Any:
        """
        Prepare the final features for ML.

        P5 (preprocessing scope — Kaufman et al. 2012):
        Implementations must ensure that any statistical transformation
        (scaling, imputation, encoding) is fitted exclusively on the
        training data. Statistics derived from the full set (including
        validation/test) constitute preprocessing leakage, even when the
        temporal separation of the folds is correct.

        Pattern required in the subclasses:
          - scaler.fit(X_train) → scaler.transform(X_val), scaler.transform(X_test)
          - fillna(reference_data.median()) where reference_data = train_data
          - NEVER use data.median() or scaler.fit(X_full)

        Args:
            data: Full data
            selected_features: List of selected features

        Returns:
            Prepared data with the final features
        """
        pass
    
    @staticmethod
    def _convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: BaseArchitectureML._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BaseArchitectureML._convert_numpy_types(v) for v in obj]
        else:
            return obj

    def save_fold_metadata(self, fold: Dict, fold_dir: str) -> None:
        """
        Save a fold's metadata in a standardized way.

        Args:
            fold: Fold configuration
            fold_dir: Fold directory
        """
        fold_metadata = {
            **self._convert_numpy_types(fold),
            'data_source': self.architecture_name,
            'target_variable': self.target_column,
            'temporal_boundaries_preserved': True,
            'gaps_applied_effectively': True,
            'saved_timestamp': datetime.now().isoformat()
        }
        
        with open(f'{fold_dir}/metadata.json', 'w') as f:
            json.dump(fold_metadata, f, indent=2)
    
    def save_master_config(self, folds: List[Dict], total_observations: int,
                          total_entities: int, year_range: Tuple[int, int]) -> str:
        """
        Save the master fold configuration.

        Args:
            folds: List of folds
            total_observations: Total number of observations
            total_entities: Total number of geographic entities (countries, municipalities)
            year_range: Tuple (min_year, max_year)

        Returns:
            Path of the saved configuration file
        """
        folds_config = {
            'architecture': self.architecture_name,
            'creation_timestamp': datetime.now().isoformat(),
            'run_id': os.environ.get('RAMPART_RUN_ID'),
            'total_observations': int(total_observations),
            'total_entities': int(total_entities),
            'year_range': [int(year_range[0]), int(year_range[1])],
            'target_variable': self.target_column,
            'folds': self._convert_numpy_types(folds)
        }
        
        folds_path = f"{self.prep_dir}/temporal_folds_{self.architecture_name}.json"
        with open(folds_path, 'w') as f:
            json.dump(folds_config, f, indent=2)
        
        return folds_path
    
    def run_setup(self) -> Dict:
        """
        Run the architecture's complete setup pipeline.

        Standardized pipeline:
        1. Environment setup
        2. Data loading
        3. Data validation
        4. Target creation
        5. Feature selection
        6. Feature preparation
        7. Temporal fold creation
        8. Saving of folds and configurations

        Returns:
            Dictionary with the setup results
        """
        print(f"Running setup {self.architecture_name}")
        
        try:
            # Architecture-specific environment setup
            self.setup_environment()
            
            # Load data
            data = self.load_data()
            
            # Validate data
            self.validate_data(data)
            
            # Create target
            data_with_target = self.create_target(data)
            
            selection_stats = self.run_feature_selection(data_with_target)
            
            data_processed = self.prepare_features(
                data_with_target, 
                selection_stats['selected_features']
            )
            
            # Create temporal folds
            folds = self.create_temporal_folds(data_processed)
            
            # Save folds
            self.save_folds(data_processed, folds)
            
            print(f"\nSetup {self.architecture_name} completed")
            
            return {
                'architecture': self.architecture_name,
                'status': 'success',
                'setup_timestamp': datetime.now().isoformat(),
                'features_selected': len(selection_stats['selected_features']),
                'folds_created': len(folds)
            }
            
        except AntiLeakageViolation:
            # Never reported as a recoverable failure. A violation means the
            # experiment does not hold the guarantees its results would be
            # reported under, so it must reach the caller and stop the run.
            print(f"\nAnti-leakage violation in {self.architecture_name}")
            raise

        except Exception as e:
            print(f"\nError in setup {self.architecture_name}: {e}")

            return {
                'architecture': self.architecture_name,
                'status': 'failed',
                'error': str(e),
                'setup_timestamp': datetime.now().isoformat()
            }
    
    def validate_temporal_integrity_years(self, train_years: Tuple[int, int],
                                   val_years: Tuple[int, int],
                                   test_years: Tuple[int, int]) -> bool:
        """
        Validate the temporal integrity of the splits to prevent leakage.

        Checks whether:
        1. The periods are in correct chronological order
        2. There are adequate temporal gaps between splits
        3. There is no overlap between periods

        Args:
            train_years: Tuple (start_year, end_year) of the training window
            val_years: Tuple (start_year, end_year) of the validation window
            test_years: Tuple (start_year, end_year) of the test window

        Returns:
            True if the temporal integrity is preserved, False otherwise

        Requires a minimum gap of 2 years between splits to prevent
        temporal leakage in educational data.
        """
        # P1: Check temporal ordering
        # val/test may span 1 year (start == end), hence <=
        if not (train_years[1] < val_years[0] <= val_years[1] < test_years[0]):
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P1 temporal ordering): "
                f"Train: {train_years}, Val: {val_years}, Test: {test_years}"
            )

        # P2: Check gaps
        train_val_gap = val_years[0] - train_years[1] - 1
        val_test_gap = test_years[0] - val_years[1] - 1

        MIN_GAP = int(self.config.get('temporal_gap_years', 2))

        if train_val_gap < MIN_GAP:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P2 gap sufficiency): "
                f"train-val gap={train_val_gap} < {MIN_GAP}"
            )

        if val_test_gap < MIN_GAP:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P2 gap sufficiency): "
                f"val-test gap={val_test_gap} < {MIN_GAP}"
            )

        print(f"   Temporal integrity OK (gaps: train-val={train_val_gap}yr, val-test={val_test_gap}yr)")

        return True
