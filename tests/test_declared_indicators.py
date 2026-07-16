#!/usr/bin/env python3
"""The collected panel must match the declared indicator set.

Twenty-three indicators were declared and twenty-two collected. GE.EST does not
exist in the World Bank Indicators API -- it answers "The indicator was not found.
It may have been deleted or archived" -- so the collector printed a warning and
carried on, and the published panel silently held one indicator fewer than the code
declares. Nothing in the artifacts records the difference.

A declared indicator that cannot be collected is a failure. Accepting its absence
has to be an explicit decision so that it is on the record.
"""

import inspect
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.indicators import ALL_INDICATORS


class TestDeclarationIsCollectible:

    def test_the_removed_indicator_is_gone(self):
        """GE.EST is a governance series, absent from the indicators API."""
        assert 'GE.EST' not in ALL_INDICATORS
        assert 'government_effectiveness' not in ALL_INDICATORS.values()

    def test_declaration_is_not_empty(self):
        assert len(ALL_INDICATORS) >= 20

    def test_codes_and_names_are_one_to_one(self):
        """A duplicated column name would silently drop a series."""
        names = list(ALL_INDICATORS.values())
        assert len(set(names)) == len(names), 'duplicate column names'

    def test_codes_look_like_world_bank_series(self):
        for code in ALL_INDICATORS:
            assert code.replace('.', '').replace('_', '').isalnum(), code


class TestMissingIndicatorAborts:

    def test_default_is_to_abort(self):
        import collection.raw_data_collector as collector

        classes = [obj for _, obj in inspect.getmembers(collector, inspect.isclass)
                   if 'allow_missing_indicators' in
                   inspect.signature(obj.__init__).parameters]
        assert classes, 'no collector accepts allow_missing_indicators'
        for cls in classes:
            default = inspect.signature(
                cls.__init__).parameters['allow_missing_indicators'].default
            assert default is False, (
                f'{cls.__name__} defaults to accepting missing indicators, so a '
                f'panel narrower than the declaration passes unnoticed'
            )

    def test_failure_path_raises_rather_than_warning(self):
        source = (_SRC / 'collection' / 'raw_data_collector.py').read_text()
        block = source[source.index('failed_indicators and not'):]
        block = block[:block.index('return final_df')]
        assert 'raise RuntimeError' in block

    def test_the_error_names_the_missing_indicators(self):
        source = (_SRC / 'collection' / 'raw_data_collector.py').read_text()
        block = source[source.index('failed_indicators and not'):]
        block = block[:block.index('return final_df')]
        assert '{failed_indicators}' in block
        assert 'indicators.py' in block, (
            'the error should point at the declaration to fix'
        )

    def test_allowance_is_still_reported(self):
        """An accepted absence must remain visible, not become silent."""
        source = (_SRC / 'collection' / 'raw_data_collector.py').read_text()
        assert 'Ausências aceitas explicitamente' in source
