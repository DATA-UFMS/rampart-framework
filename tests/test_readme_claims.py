#!/usr/bin/env python3
"""Numbers the README states about this repository must be true of it.

The test count appeared twice and disagreed with itself -- 79 in one place, 84
in another -- against a suite of over a thousand. A number nobody recomputes
rots quietly, and a reader who checks one and finds it wrong has no reason to
trust the ones they cannot check.

The panel size was stated as the analysed n. 768 is 32 countries by 24 years:
the complete grid, before rows without an observed target are removed. The
count that survives is in the collection artifact, and the README now points
there instead of asserting a figure it cannot know.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

README = (_ROOT / 'README.md').read_text()


@pytest.fixture(scope='module')
def collected():
    """How many tests the suite actually has, by collecting it."""
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-q', '--collect-only',
         '-p', 'no:warnings', '-p', 'no:cacheprovider'],
        cwd=str(_ROOT), capture_output=True, text=True)
    match = re.search(r'(\d+) tests? collected', result.stdout)
    assert match, result.stdout[-500:]
    return int(match.group(1))


class TestTheTestCount:

    def test_every_stated_count_matches(self, collected):
        stated = [int(value) for value in
                  re.findall(r'(\d[\d.]*)\s+testes', README)]
        assert stated, 'the README no longer states a test count'
        for value in stated:
            assert value == collected, (
                f'README says {value} tests, the suite collects {collected}'
            )

    def test_the_counts_agree_with_each_other(self):
        """They disagreed: 79 in one section, 84 in another."""
        stated = {int(value) for value in
                  re.findall(r'(\d[\d.]*)\s+testes', README)}
        assert len(stated) == 1, f'README states several counts: {stated}'

    def test_it_is_stated_more_than_once(self, collected):
        """Both places must be kept honest, not only the first."""
        assert len(re.findall(r'(\d[\d.]*)\s+testes', README)) >= 2


class TestThePanelSize:

    def test_the_grid_is_not_presented_as_the_analysed_n(self):
        assert '768 obs' not in README, (
            '768 is 32 countries by 24 years, the complete grid before rows '
            'without an observed target are removed'
        )

    def test_the_grid_arithmetic_holds(self):
        """Whatever it is called, the number must be the product it claims."""
        match = re.search(r'(\d+) países × (\d+) anos, painel completo de '
                          r'(\d+)', README)
        assert match, 'the panel claim is no longer stated in a checkable form'
        countries, years, total = (int(value) for value in match.groups())
        assert countries * years == total

    def test_the_country_count_matches_the_configuration(self):
        from core.config import LATIN_AMERICA_COUNTRIES
        match = re.search(r'(\d+) países ×', README)
        assert int(match.group(1)) == len(LATIN_AMERICA_COUNTRIES)

    def test_the_year_span_matches_the_configuration(self):
        from core.scientific_config import SCIENTIFIC_CONFIG
        start = SCIENTIFIC_CONFIG['temporal_range_start']
        end = SCIENTIFIC_CONFIG['temporal_range_end']
        match = re.search(r'× (\d+) anos', README)
        assert int(match.group(1)) == end - start + 1

    def test_the_reader_is_pointed_at_the_artifact(self):
        assert 'target_coverage.json' in README


class TestTheCoreBudgetClaim:

    def test_no_machine_below_the_budget_is_documented(self):
        from core.scientific_config import SCIENTIFIC_CONFIG
        minimum = (SCIENTIFIC_CONFIG['engine_threads']
                   + SCIENTIFIC_CONFIG['blas_threads'] - 1)
        for count in re.findall(r'(\d+)\s*vCPU', README):
            assert int(count) >= minimum
