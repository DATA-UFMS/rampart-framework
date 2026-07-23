#!/usr/bin/env python3
"""The three paradigms hand the models the same fold.

Each applies the same policy in its own idiom -- drop rows with no target,
order by entity then year -- through an ORDER BY in the SQL view, a Polars
sort, a pandas sort after compute. Performing it is part of what the benchmark
measures, so it stays inside each engine. Verifying it does not: three
implementations of one policy are three chances to disagree, and a
disagreement here falsifies the bitwise claim for a reason that has nothing to
do with the paradigms.

The concrete defect behind this file: the Dask model reordered with
``X.loc[sort_idx]``. Each Dask partition carries its own index, so after
``compute`` the labels repeat across partitions, and label-based selection
returns every row matching each label. Measured here: sixty-four rows in, two
hundred and fifty-six out with four partitions. The fit succeeds on the
quadrupled fold, the hierarchical stage's latency grows accordingly, and
nothing downstream looks.
"""

import contextlib
import hashlib
import importlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.paradigm_registry import discover_paradigms
from core.validation import canonical_fold

ENTITIES = ['ARG', 'BRA', 'CHL', 'URY']
YEARS = list(range(2000, 2016))
FEATURES = ['gini', 'internet']


def _panel(target_column='dropout_rate'):
    rng = np.random.default_rng(11)
    return pd.DataFrame([
        {'country_code': entity, 'year': year,
         'gini': rng.normal(40, 5), 'internet': rng.normal(50, 8),
         target_column: rng.normal(10, 2)}
        for entity in ENTITIES for year in YEARS
    ])


def _digest(frame):
    values = pd.util.hash_pandas_object(
        pd.DataFrame(frame).reset_index(drop=True), index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


class TestCanonicalFold:
    """The shared verification, in isolation."""

    @staticmethod
    def _parts(n=6):
        entities = pd.Series(np.repeat(['AAA', 'BBB'], n // 2))
        years = pd.Series(np.tile(np.arange(2000, 2000 + n // 2), 2))
        X = pd.DataFrame({'f': np.arange(n, dtype=float)})
        y = pd.Series(np.arange(n, dtype=float))
        return X, y, entities, years

    def test_a_canonical_fold_passes(self):
        X, y, entities, years = self._parts()
        out_X, out_y, out_entities = canonical_fold(X, y, entities, years,
                                                    paradigm='probe')
        assert len(out_X) == len(out_y) == len(out_entities) == 6

    def test_the_returned_index_is_positional(self):
        """Downstream alignment is positional; a surviving label is a hazard."""
        X, y, entities, years = self._parts()
        for part in (X, y, entities):
            part.index = [9, 8, 7, 6, 5, 4]
        out_X, out_y, out_entities = canonical_fold(X, y, entities, years,
                                                    paradigm='probe')
        for out in (out_X, out_y, out_entities):
            assert list(out.index) == list(range(6))

    def test_a_length_mismatch_halts(self):
        X, y, entities, years = self._parts()
        with pytest.raises(ValueError, match='inconsistent lengths'):
            canonical_fold(X, y.iloc[:-1], entities, years, paradigm='probe')

    def test_an_empty_fold_halts(self):
        X, y, entities, years = self._parts()
        with pytest.raises(ValueError, match='empty'):
            canonical_fold(X.iloc[:0], y.iloc[:0], entities.iloc[:0],
                           years.iloc[:0], paradigm='probe')

    def test_a_missing_target_halts(self):
        X, y, entities, years = self._parts()
        y = y.copy()
        y.iloc[2] = np.nan
        with pytest.raises(ValueError, match='no target'):
            canonical_fold(X, y, entities, years, paradigm='probe')

    def test_a_duplicated_entity_year_halts(self):
        """The signature of a join that multiplied rows."""
        X, y, entities, years = self._parts()
        years = years.copy()
        years.iloc[1] = years.iloc[0]
        with pytest.raises(ValueError, match='duplicated'):
            canonical_fold(X, y, entities, years, paradigm='probe')

    def test_an_unordered_fold_halts(self):
        # Pairs stay unique, so this reaches the ordering check rather than
        # tripping the duplicate one first.
        X, y, entities, years = self._parts()
        entities = pd.Series(['BBB'] * 3 + ['AAA'] * 3)
        assert not pd.MultiIndex.from_arrays(
            [entities, years]).duplicated().any()
        with pytest.raises(ValueError, match='not ordered'):
            canonical_fold(X, y, entities, years, paradigm='probe')

    def test_the_message_names_the_paradigm(self):
        X, y, entities, years = self._parts()
        with pytest.raises(ValueError, match='task_graph'):
            canonical_fold(X, y.iloc[:-1], entities, years,
                           paradigm='task_graph')

    def test_ordering_within_an_entity_is_checked(self):
        """Grouping by entity is not enough; the years must ascend too."""
        X, y, entities, years = self._parts()
        years = pd.Series([2002, 2001, 2000, 2000, 2001, 2002])
        with pytest.raises(ValueError, match='not ordered'):
            canonical_fold(X, y, entities, years, paradigm='probe')


def _materialise(paradigm, frame):
    """Run that paradigm's own _prepare_data over a shared panel."""
    module = importlib.import_module(
        f'architectures_ml.{paradigm}.models.hierarchical_model')
    cls = next(getattr(module, name) for name in dir(module)
               if isinstance(getattr(module, name), type)
               and hasattr(getattr(module, name), '_prepare_data'))
    instance = cls.__new__(cls)
    instance.target_col = 'dropout_rate'
    instance.available_features = list(FEATURES)

    if paradigm == 'sql_engine':
        # The view supplies the canonical order; nothing else does it for this
        # paradigm, which is why the verification exists.
        data = frame.sort_values(['country_code', 'year']).reset_index(
            drop=True)
        argument = (data, list(FEATURES))
    elif paradigm == 'task_graph':
        import dask.dataframe as dd
        argument = (dd.from_pandas(frame, npartitions=3),)
    else:
        import polars as pl
        argument = (pl.from_pandas(frame).lazy(),)

    with contextlib.redirect_stdout(io.StringIO()):
        warnings.filterwarnings('ignore')
        return instance._prepare_data(*argument)


class TestTheParadigmsAgree:

    @pytest.fixture(scope='class')
    def materialised(self):
        # Shuffled on the way in: the two paradigms that sort must recover the
        # canonical order, and the one that relies on its view is handed it.
        shuffled = _panel().sample(frac=1.0, random_state=7).reset_index(
            drop=True)
        return {paradigm: _materialise(paradigm, shuffled)
                for paradigm in sorted(discover_paradigms())}

    def test_every_paradigm_returns_the_full_panel(self, materialised):
        for paradigm, (X, y, entities) in materialised.items():
            assert len(X) == len(ENTITIES) * len(YEARS), paradigm
            assert len(y) == len(entities) == len(X), paradigm

    def test_the_feature_matrices_are_bitwise_identical(self, materialised):
        digests = {paradigm: _digest(X)
                   for paradigm, (X, _, _) in materialised.items()}
        assert len(set(digests.values())) == 1, digests

    def test_the_targets_are_bitwise_identical(self, materialised):
        digests = {paradigm: _digest(pd.DataFrame({'y': y.to_numpy()}))
                   for paradigm, (_, y, _) in materialised.items()}
        assert len(set(digests.values())) == 1, digests

    def test_the_entity_vectors_are_bitwise_identical(self, materialised):
        digests = {paradigm: _digest(pd.DataFrame({'e': e.to_numpy()}))
                   for paradigm, (_, _, e) in materialised.items()}
        assert len(set(digests.values())) == 1, digests

    def test_the_shuffle_actually_disturbed_the_order(self):
        """Otherwise agreement would follow from nothing having been shuffled."""
        panel = _panel()
        shuffled = panel.sample(frac=1.0, random_state=7).reset_index(
            drop=True)
        assert not shuffled[['country_code', 'year']].equals(
            panel[['country_code', 'year']])

    def test_the_columns_are_in_the_declared_order(self, materialised):
        for paradigm, (X, _, _) in materialised.items():
            assert list(pd.DataFrame(X).columns) == FEATURES, paradigm


class TestDuplicateLabelsDoNotMultiplyRows:
    """The regression, reproduced through the production path."""

    @staticmethod
    def _partitioned_with_repeating_labels(partitions):
        import dask.dataframe as dd
        shuffled = _panel().sample(frac=1.0, random_state=7).reset_index(
            drop=True)
        frame = dd.from_pandas(shuffled, npartitions=partitions)
        # Each partition back to 0..k-1: what a parquet read produces, and
        # what makes the labels repeat across partitions.
        return frame.map_partitions(lambda part: part.reset_index(drop=True))

    @pytest.mark.parametrize('partitions', [1, 2, 4, 8])
    def test_the_row_count_does_not_depend_on_the_partitioning(self,
                                                               partitions):
        X, y, entities = _materialise_dask(
            self._partitioned_with_repeating_labels(partitions))
        assert len(X) == len(ENTITIES) * len(YEARS), (
            f'{partitions} partitions produced {len(X)} rows'
        )
        assert len(y) == len(entities) == len(X)

    def test_the_content_does_not_depend_on_the_partitioning(self):
        digests = {partitions: _digest(_materialise_dask(
            self._partitioned_with_repeating_labels(partitions))[0])
            for partitions in (1, 2, 4, 8)}
        assert len(set(digests.values())) == 1, digests

    def test_the_labels_really_do_repeat(self):
        """Without this the tests above could be exercising a unique index."""
        frame = self._partitioned_with_repeating_labels(4).compute()
        assert frame.index.duplicated().any()

    def test_no_paradigm_reorders_by_label(self):
        """`.loc` with a computed index is the shape of the defect.

        Checked over the syntax tree rather than the text, so the comment
        explaining the removed pattern does not count as the pattern.
        """
        import ast as ast_module
        for paradigm in sorted(discover_paradigms()):
            path = (_SRC / 'architectures_ml' / paradigm / 'models'
                    / 'hierarchical_model.py')
            tree = ast_module.parse(path.read_text())
            prepare = next(
                (node for node in ast_module.walk(tree)
                 if isinstance(node, ast_module.FunctionDef)
                 and node.name == '_prepare_data'), None)
            if prepare is None:
                continue
            for node in ast_module.walk(prepare):
                if (isinstance(node, ast_module.Subscript)
                        and isinstance(node.value, ast_module.Attribute)
                        and node.value.attr == 'loc'):
                    raise AssertionError(
                        f'{paradigm}._prepare_data selects by label at line '
                        f'{node.lineno}; with repeating labels that multiplies '
                        f'rows instead of reordering them'
                    )


def _materialise_dask(frame):
    module = importlib.import_module(
        'architectures_ml.task_graph.models.hierarchical_model')
    cls = next(getattr(module, name) for name in dir(module)
               if isinstance(getattr(module, name), type)
               and hasattr(getattr(module, name), '_prepare_data'))
    instance = cls.__new__(cls)
    instance.target_col = 'dropout_rate'
    instance.available_features = list(FEATURES)
    with contextlib.redirect_stdout(io.StringIO()):
        warnings.filterwarnings('ignore')
        return instance._prepare_data(frame)


class TestEveryParadigmIsVerified:

    @pytest.mark.parametrize('paradigm', sorted(discover_paradigms()))
    def test_prepare_data_ends_in_the_shared_check(self, paradigm):
        source = (_SRC / 'architectures_ml' / paradigm / 'models'
                  / 'hierarchical_model.py').read_text()
        assert 'canonical_fold(' in source, (
            f'{paradigm} hands the models a fold nothing checked'
        )

    @pytest.mark.parametrize('paradigm', sorted(discover_paradigms()))
    def test_the_check_is_told_which_paradigm_it_is(self, paradigm):
        source = (_SRC / 'architectures_ml' / paradigm / 'models'
                  / 'hierarchical_model.py').read_text()
        assert f"PARADIGM = '{paradigm}'" in source
        assert 'paradigm=PARADIGM' in source


class TestTheLagColumnsAreNotOptional:
    """Two paradigms built them inside a try/except that only warned.

    A paradigm returning the frame without its lag columns trains on a
    different feature set from the other two, so the bitwise claim fails for a
    reason that has nothing to do with the paradigms -- and the only trace was
    a line of stdout in a run that takes hours. One of the two caught bare
    Exception.

    Where the entity's past target was never observed the join yields NULL,
    which is the honest value and is handled downstream. A missing *column* is
    a different thing.
    """

    def test_the_check_names_the_missing_columns(self):
        from core.base_architecture import BaseArchitectureML
        from core.validation import assert_lag_columns

        with pytest.raises(ValueError, match='dropout_rate_lag_3'):
            assert_lag_columns(['country_code', 'year',
                                'dropout_rate_lag_2'], 'task_graph',
                               BaseArchitectureML.TARGET_LAG_ORDERS)

    def test_a_complete_set_passes(self):
        from core.base_architecture import BaseArchitectureML
        from core.validation import assert_lag_columns

        columns = ['country_code', 'year'] + [
            f'dropout_rate_lag_{order}'
            for order in BaseArchitectureML.TARGET_LAG_ORDERS]
        assert_lag_columns(columns, 'sql_engine',
                           BaseArchitectureML.TARGET_LAG_ORDERS)

    def test_it_names_the_paradigm(self):
        from core.base_architecture import BaseArchitectureML
        from core.validation import assert_lag_columns
        with pytest.raises(ValueError, match='dataframe_lib'):
            assert_lag_columns([], 'dataframe_lib',
                               BaseArchitectureML.TARGET_LAG_ORDERS)

    @pytest.mark.parametrize('paradigm', ['task_graph', 'dataframe_lib'])
    def test_no_paradigm_swallows_a_lag_failure(self, paradigm):
        source = (_SRC / 'architectures_ml' / paradigm / 'setup.py').read_text()
        assert '[WARN] Falha ao criar dropout_rate_lag_2' not in source, (
            f'{paradigm} still continues past a failed lag join'
        )
        assert 'falha ao criar as defasagens do alvo' in source

    @pytest.mark.parametrize('paradigm', ['task_graph', 'dataframe_lib'])
    def test_each_paradigm_checks_afterwards(self, paradigm):
        """The raise covers a thrown error; the check covers a silent absence."""
        source = (_SRC / 'architectures_ml' / paradigm / 'setup.py').read_text()
        assert f"assert_lag_columns(" in source

    def test_the_sql_view_selects_them(self):
        """The third paradigm builds them in SQL, where absence is an error."""
        source = (_SRC / 'architectures_ml' / 'sql_engine' / 'setup.py').read_text()
        from core.base_architecture import BaseArchitectureML
        for order in BaseArchitectureML.TARGET_LAG_ORDERS:
            assert f'dropout_rate_lag_{order}' in source
