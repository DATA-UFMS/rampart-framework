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


class TestNoResultIsTranscribed:
    """Speedup factors stated in prose come from a run nobody can point at.

    The README asserted ~9.6x end-to-end and ~2.0-2.2x on the large panel.
    Nothing in the repository produces those numbers: the statistics directory
    they would come from does not exist here, and the run that produced them
    predates the corrections to the bootstrap, the multiple-comparison
    procedure and the latency labels.

    A derived table carries its own provenance -- commit, timestamp, core
    budget -- in its caption. A number retyped into prose carries none, and
    survives the run that invalidates it.
    """

    #: Where a reader is sent instead.
    ARTIFACTS = ('architectural_latency_percentiles.json',
                 'derive_paper_tables.py')

    @pytest.mark.parametrize('name', sorted(DOCUMENTS))
    def test_no_speedup_factor_is_stated_in_prose(self, name):
        prose = '\n'.join(line for line in DOCUMENTS[name].splitlines()
                          if not line.strip().startswith(('|', '```', '#')))
        factors = re.findall(r'~?\d+[,.]\d+\s*×', prose)
        assert not factors, (
            f'{name} states {factors}; those come from a run whose artifacts '
            f'are not in the repository, and the derived table is where a '
            f'factor carries its provenance'
        )

    def test_the_reader_is_sent_to_the_artifact(self):
        for name in self.ARTIFACTS:
            assert name in README, name

    def test_the_qualitative_finding_survives(self):
        """Removing the numbers must not remove the claim."""
        assert 'crossover' in README
        assert 'in-process' in README

    def test_the_provenance_requirement_is_stated(self):
        assert 'orçamento de núcleos' in README and 'commit' in README


class TestTheArtifactTable:
    """Every path the guide lists must be one the pipeline writes.

    And the reverse matters more: the guide listed six artifacts and omitted
    the ones that make the imputation auditable, so the evidence existed and
    was not discoverable. A reviewer reads this table to know what to open.
    """

    GUIDE = (_ROOT / 'USAGE_GUIDE.md').read_text()

    #: Written by the pipeline and worth opening. Absence from the guide is
    #: the defect: the artifact exists, and nobody is told.
    EXPECTED = (
        'temporal_folds_', 'target_statistics.json',
        'equivalence_estimation.json', 'architectural_benchmark_results.csv',
        'architectural_benchmark_resource_log.jsonl',
        'architectural_scorecard.tex', 'predictions_',
        'target_coverage.json', 'fold_imputation_', 'used_features_fold_',
        'scientific_config_snapshot.json',
    )

    @staticmethod
    def _sources():
        paths = list((_ROOT / 'src').rglob('*.py')) + [_ROOT / 'pipeline.py']
        return '\n'.join(path.read_text() for path in paths)

    @pytest.mark.parametrize('stem', EXPECTED)
    def test_the_guide_lists_it(self, stem):
        assert stem in self.GUIDE, (
            f'{stem} is written by the pipeline and the guide does not '
            f'mention it'
        )

    @pytest.mark.parametrize('stem', EXPECTED)
    def test_the_pipeline_writes_it(self, stem):
        assert stem in self._sources(), (
            f'the guide sends the reader to {stem}, which nothing produces'
        )

    def test_the_three_imputation_artifacts_are_distinguished(self):
        """Each answers a different question; one is not a summary of another."""
        for stem in ('target_coverage.json', 'fold_imputation_'):
            assert stem in self.GUIDE
        assert 'sem limite de alcance' in self.GUIDE, (
            'the guide should say which of the two is unbounded'
        )


class TestTheProtocolTable:
    """The highest-stakes prose in the repository: what the gates enforce.

    Each row names the mechanism, so the claim can be checked instead of
    trusted. P5 said "contract plus unit tests" while the code had come to
    raise as well -- understating is safer than overstating, but neither is
    checkable while the row names nothing.
    """

    #: (protocol, symbol that must exist and must raise)
    ROWS = [
        ('P1', 'enforce_walk_forward'),
        ('P2', 'enforce_walk_forward'),
        ('P3', 'audit_feature_set'),
        ('P4', '_first_fold_train_end'),
        ('P5', 'impute_from_training_window'),
    ]

    @pytest.mark.parametrize('protocol,symbol', ROWS)
    def test_the_row_names_its_mechanism(self, protocol, symbol):
        row = next((line for line in README.splitlines()
                    if line.startswith(f'| {protocol} ')), None)
        assert row, f'{protocol} is not in the table'
        assert symbol in row, f'{protocol} names no mechanism: {row}'

    @staticmethod
    def _resolve(symbol):
        """Module-level function, or a method of one of the two gate classes."""
        from core import validation
        from core.base_architecture import BaseArchitectureML
        for holder in (validation, validation.TemporalValidator,
                       BaseArchitectureML):
            found = getattr(holder, symbol, None)
            if found is not None:
                return found
        return None

    @pytest.mark.parametrize('protocol,symbol', ROWS)
    def test_the_mechanism_exists(self, protocol, symbol):
        assert self._resolve(symbol) is not None, (
            f'{protocol} points at {symbol}, which does not exist'
        )

    @pytest.mark.parametrize('protocol,symbol', [r for r in ROWS
                                                 if r[0] != 'P4'])
    def test_the_mechanism_can_raise(self, protocol, symbol):
        """A row promising runtime enforcement must point at code that raises.

        Caught one of these: the P2 row named validate_fold_integrity, which
        returns a verdict and a list of errors. The raise lives one level up,
        in enforce_walk_forward -- so the row pointed at the part that decides
        rather than the part that stops the run.
        """
        import ast as ast_module
        import inspect
        import textwrap
        source = textwrap.dedent(inspect.getsource(self._resolve(symbol)))
        tree = ast_module.parse(source)
        assert any(isinstance(node, ast_module.Raise)
                   for node in ast_module.walk(tree)), (
            f'{protocol} points at {symbol}, which returns rather than raising'
        )

    def test_every_protocol_appears(self):
        for protocol, _ in self.ROWS:
            assert f'| {protocol} ' in README

    def test_the_newly_enforced_cases_are_stated(self):
        """Each was a case that passed silently until it was found."""
        for phrase in ('vazio', 'diferem entre paradigmas',
                       'nenhuma observação'):
            assert phrase in README, phrase
