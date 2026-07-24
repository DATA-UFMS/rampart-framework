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

#: Both user-facing documents make the same claims about the machine and
#: the protocol. The vCPU figure was corrected in one and left stale in
#: the other, because the check only looked at the first.
DOCUMENTS = {name: (_ROOT / name).read_text()
             for name in ('README.md', 'USAGE_GUIDE.md')}


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
    """Both documents, not only the first: the fix landed in one of them."""

    @pytest.mark.parametrize('name', sorted(DOCUMENTS))
    def test_no_machine_below_the_budget_is_documented(self, name):
        from core.scientific_config import SCIENTIFIC_CONFIG
        minimum = (SCIENTIFIC_CONFIG['engine_threads']
                   + SCIENTIFIC_CONFIG['blas_threads'] - 1)
        for count in re.findall(r'(\d+)\s*vCPU', DOCUMENTS[name]):
            assert int(count) >= minimum, (
                f'{name} documents a {count}-vCPU machine while pipeline.py '
                f'requires {minimum} and refuses to start below it'
            )

    @pytest.mark.parametrize('name', sorted(DOCUMENTS))
    def test_the_minimum_is_stated(self, name):
        from core.scientific_config import SCIENTIFIC_CONFIG
        minimum = (SCIENTIFIC_CONFIG['engine_threads']
                   + SCIENTIFIC_CONFIG['blas_threads'] - 1)
        assert str(minimum) in DOCUMENTS[name], (
            f'{name} does not state the core budget the pipeline enforces'
        )


class TestTheProtocolNumbers:
    """The repetition count is the protocol's n; a stale example misleads."""

    @pytest.mark.parametrize('name', sorted(DOCUMENTS))
    def test_the_stated_total_matches_the_configuration(self, name):
        from core.config import BENCHMARK_CONFIG
        total = (BENCHMARK_CONFIG['repetitions']
                 + BENCHMARK_CONFIG['warmup_runs'])
        for stated in re.findall(r'`warmup \+ n` vezes \((\d+) por padrão\)',
                                 DOCUMENTS[name]):
            assert int(stated) == total, (
                f'{name} says {stated} passes, the configuration gives {total}'
            )

    @pytest.mark.parametrize('name', sorted(DOCUMENTS))
    def test_the_documented_flags_match_the_defaults(self, name):
        """An example that contradicts the default is a second source for n."""
        from core.config import BENCHMARK_CONFIG
        pairs = re.findall(r'--repetitions (\d+) --warmup (\d+)',
                           DOCUMENTS[name])
        canonical = (str(BENCHMARK_CONFIG['repetitions']),
                     str(BENCHMARK_CONFIG['warmup_runs']))
        assert any(pair == canonical for pair in pairs) or not pairs, (
            f'{name} shows {pairs} and none reproduces the configured '
            f'{canonical}'
        )


class TestTheExtensionExample:
    """Following it produced a paradigm that cannot load, and must not.

    It listed `get_numeric_features` among the methods to implement. That one
    is not abstract, and a test forbids overriding it: the candidate pool has
    to be identical across paradigms, or the comparison starts from different
    search spaces. The method a new engine does implement is
    `discover_numeric_columns`.

    The metadata block showed three keys of the fifteen discovery requires, so
    a paradigm written from the example would not be found at all.
    """

    @staticmethod
    def _example():
        block = README[README.index('# src/architectures_ml/meu_paradigma'):]
        return block[:block.index('```')]

    def test_every_abstract_method_is_listed(self):
        from core.base_architecture import BaseArchitectureML
        example = self._example()
        for name in sorted(BaseArchitectureML.__abstractmethods__):
            assert name in example, (
                f'{name} is abstract and the example does not mention it'
            )

    def test_the_stated_count_matches(self):
        from core.base_architecture import BaseArchitectureML
        match = re.search(r'Métodos abstratos a implementar \((\d+)\)',
                          self._example())
        assert match
        assert int(match.group(1)) == len(
            BaseArchitectureML.__abstractmethods__)

    def test_no_non_abstract_method_is_listed_as_required(self):
        from core.base_architecture import BaseArchitectureML
        example = self._example()
        listed = set(re.findall(r'^    #   (\w+)$', example, re.M))
        assert listed == set(BaseArchitectureML.__abstractmethods__), (
            f'the example asks for methods that are not abstract: '
            f'{sorted(listed - set(BaseArchitectureML.__abstractmethods__))}'
        )

    def test_the_pool_policy_method_is_not_offered(self):
        """The specific one that broke the suite for whoever followed it."""
        example = self._example()
        assert '#   get_numeric_features' not in example
        assert 'get_numeric_features' in README, (
            'the README should say why it is absent, not merely omit it'
        )

    def test_every_metadata_key_appears(self):
        from core.paradigm_registry import discover_paradigms
        example = self._example()
        reference = next(iter(discover_paradigms().values()))
        for key in sorted(reference):
            assert f"'{key}'" in example, (
                f'{key} is in every paradigm\'s metadata and the example omits '
                f'it; a paradigm written from this would not be discovered'
            )

    def test_the_example_names_the_replacement(self):
        assert 'discover_numeric_columns' in self._example() or \
            'discover_numeric_columns' in README
