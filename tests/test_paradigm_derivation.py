#!/usr/bin/env python3
"""Analysis modules derive the paradigms instead of naming them.

The three paradigm names appeared in eight analysis modules, and the pairing was
written out four times under the abbreviations dl/dw/pl -- which encoded the
pre-rename names (data_lake, data_warehouse, polars) and named nothing after the
rename. A fourth paradigm entered the comparison only after each module was
edited by hand, which is the opposite of what an extensible framework claims.

The latency loop is exercised end to end here because that is where a NameError
survived: the loop referenced label_a and label_b, which the surrounding scope
did not define, and no test reached it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.paradigm_registry import (baseline_results_paths, discover_paradigms,
                                    paradigm_pairs)

ANALYSIS_MODULES = [
    'statistical_validation.significance_tests',
    'statistical_validation.effect_analysis',
    'statistical_validation.equivalence_estimation',
    'statistical_validation.bootstrap_sensitivity',
]


class TestPairsComeFromTheRegistry:

    def test_every_combination_appears_once(self):
        names = sorted(discover_paradigms())
        pairs = paradigm_pairs()
        assert len(pairs) == len(names) * (len(names) - 1) // 2
        assert len(set(pairs)) == len(pairs)

    def test_pairs_are_ordered_within_and_across(self):
        """Ordering fixes the sign of every effect, so it must be stable."""
        pairs = paradigm_pairs()
        assert all(a < b for a, b in pairs)
        assert pairs == sorted(pairs)

    def test_ordering_survives_a_permuted_registry(self, monkeypatch):
        """The registry's insertion order is alphabetical only by accident.

        If discovery order reached the pairing, adding a directory could reorder
        the pairs, silently flipping the sign of every effect involving them.
        """
        import core.paradigm_registry as registry

        real = registry.discover_paradigms()
        permuted = dict(reversed(list(real.items())))
        assert list(permuted) != list(real), 'need at least two paradigms'
        monkeypatch.setattr(registry, 'discover_paradigms',
                            lambda **kw: permuted)
        assert registry.paradigm_pairs() == sorted(registry.paradigm_pairs())
        assert all(a < b for a, b in registry.paradigm_pairs())

    def test_no_pair_repeats_a_paradigm(self):
        assert all(a != b for a, b in paradigm_pairs())

    @pytest.mark.parametrize('module_name', ANALYSIS_MODULES)
    def test_module_uses_the_derived_pairs(self, module_name):
        import importlib
        module = importlib.import_module(module_name)
        declared = [getattr(module, name) for name in
                    ('ALL_PAIRS', 'PREDICTIVE_PAIRS', 'LATENCY_PAIRS',
                     'PAIR_CONFIGS') if hasattr(module, name)]
        assert declared, f'{module_name} declares no pair list'
        for pairs in declared:
            assert list(pairs) == paradigm_pairs(), (
                f'{module_name} does not use the registry pairing'
            )


class TestAbbreviationsAreGone:

    @pytest.mark.parametrize('path', sorted(_SRC.rglob('*.py')),
                             ids=lambda p: str(p.relative_to(_SRC)))
    def test_no_module_uses_dl_dw_pl(self, path):
        source = path.read_text()
        for token in ("'dl'", '"dl"', "'dw'", '"dw"', "'pl'", '"pl"'):
            assert token not in source, (
                f"{path.relative_to(_ROOT)} still uses {token}, which named a "
                f"paradigm that no longer exists under that name"
            )


class TestResultPathsAreDeclared:

    def test_every_paradigm_declares_its_baseline_results(self):
        paths = baseline_results_paths()
        assert set(paths) == set(discover_paradigms())

    def test_paths_differ_per_paradigm(self):
        paths = baseline_results_paths()
        assert len(set(paths.values())) == len(paths)

    def test_a_paradigm_without_the_declaration_raises(self, monkeypatch):
        """Guessing the layout is what made a fourth paradigm impossible."""
        import core.paradigm_registry as registry

        monkeypatch.setattr(registry, 'discover_paradigms',
                            lambda **kw: {'toy': {'name': 'toy'}})
        with pytest.raises(KeyError, match='baseline_results_json'):
            registry.baseline_results_paths()


class TestLatencyLoopRuns:
    """End to end over the latency branch, where a NameError went unnoticed."""

    @pytest.fixture
    def benchmark_csv(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(23)
        names = sorted(discover_paradigms())
        base = {name: 1.0 + 0.3 * i for i, name in enumerate(names)}
        rows = []
        for run_id in range(10):
            for phase in ('processing', 'baseline'):
                for name in names:
                    rows.append({
                        'run_id': run_id, 'phase': phase,
                        'architecture': name,
                        'duration_s': base[name] * (1 + 0.05 * rng.normal()),
                    })
        outputs = tmp_path / 'outputs' / 'worldbank' / 'benchmarks'
        outputs.mkdir(parents=True)
        csv = outputs / 'architectural_benchmark_results.csv'
        pd.DataFrame(rows).to_csv(csv, index=False)
        return csv

    @pytest.fixture
    def latency_results(self, benchmark_csv, monkeypatch):
        from statistical_validation import equivalence_estimation as ee

        monkeypatch.setattr(ee, '_load_benchmark_csv',
                            lambda: pd.read_csv(benchmark_csv))

        class Args:
            bootstrap = 500
            seed = 42
            latency_delta = 0.05
            latency_delta_profile = None

        return ee._analyze_latency(Args())

    def test_every_phase_produces_results(self, latency_results):
        assert latency_results, 'no phase produced a result'

    def test_every_pair_is_covered(self, latency_results):
        expected = {f'{a}_vs_{b}' for a, b in paradigm_pairs()}
        for phase, per_pair in latency_results.items():
            assert set(per_pair) == expected, (
                f'{phase} covers {set(per_pair)}, expected {expected}'
            )

    def test_each_record_names_the_favoured_side(self, latency_results):
        for phase, per_pair in latency_results.items():
            for pair_key, record in per_pair.items():
                if record.get('status') == 'insufficient_data':
                    continue
                assert 'advantage' in record, (
                    f'{phase}/{pair_key} lacks the advantage field'
                )

    def test_results_are_serialisable(self, latency_results):
        """The artifact is JSON; a numpy scalar would break the write."""
        json.dumps(latency_results, default=str)
