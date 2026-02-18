#!/usr/bin/env python3
"""
Testes unitários para a lógica central do framework.

Estes testes validam algoritmos fundamentais SEM depender de
saídas de pipeline pré-geradas ou dados externos. Executam com
apenas numpy e pandas como dependências.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. Transformação log simétrica
# ---------------------------------------------------------------------------

def _symmetric_log(x):
    """Implementação de referência: sign(x) * ln(|x| + 1)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log(np.abs(x) + 1)


class TestSymmetricLogTransform:
    """Testes para T(x) = sign(x) * ln(|x| + 1)."""

    def test_zero(self):
        assert _symmetric_log(0.0) == 0.0

    def test_positive_matches_log1p(self):
        for val in [1.0, 10.0, 1000.0, 1e6]:
            np.testing.assert_allclose(
                _symmetric_log(val), np.log1p(val), rtol=1e-12
            )

    def test_negative_mirrors_positive(self):
        for val in [1.0, 50.0, 1e4]:
            np.testing.assert_allclose(
                _symmetric_log(-val), -_symmetric_log(val), rtol=1e-12
            )

    def test_monotonic_increasing(self):
        xs = np.linspace(-100, 100, 500)
        ys = _symmetric_log(xs)
        assert np.all(np.diff(ys) >= 0), "Transformação deve ser monotonicamente crescente"

    def test_vectorized(self):
        arr = np.array([-10, -1, 0, 1, 10])
        result = _symmetric_log(arr)
        assert result.shape == (5,)
        assert result[2] == 0.0
        assert result[0] < 0 < result[4]

    def test_sql_equivalence(self):
        """Verifica que implementação numpy corresponde à fórmula SQL SIGN(x)*LN(ABS(x)+1)."""
        xs = np.array([-1e6, -100, -1, 0, 1, 100, 1e6])
        sql_style = np.sign(xs) * np.log(np.abs(xs) + 1)
        np.testing.assert_array_equal(_symmetric_log(xs), sql_style)


# ---------------------------------------------------------------------------
# 2. Geração de folds walk-forward
# ---------------------------------------------------------------------------

def _generate_folds(start_year, end_year, min_train, val_len, test_len, gap, step=1):
    """
    Gerador de folds standalone extraído de BaseArchitectureML._generate_walkforward_folds_auto.
    Retorna uma lista de dicionários de fold.
    """
    test_start_min = start_year + min_train + val_len + 2 * gap
    test_start_max = end_year - test_len + 1

    folds = []
    fold_id = 0
    for test_start in range(test_start_min, test_start_max + 1, step):
        test_end = test_start + test_len - 1
        val_end = test_start - gap - 1
        val_start = val_end - val_len + 1
        train_end = val_start - gap - 1
        train_start = start_year

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

        folds.append({
            'fold_id': fold_id,
            'train_start': train_start, 'train_end': train_end,
            'val_start': val_start, 'val_end': val_end,
            'test_start': test_start, 'test_end': test_end,
            'train_val_gap': train_val_gap, 'val_test_gap': val_test_gap,
        })
        fold_id += 1
    return folds


class TestWalkForwardFolds:
    """Testes para geração de folds temporais walk-forward."""

    @pytest.fixture
    def default_folds(self):
        return _generate_folds(
            start_year=2000, end_year=2023,
            min_train=8, val_len=2, test_len=2, gap=2,
        )

    def test_nine_folds(self, default_folds):
        assert len(default_folds) == 9

    def test_no_temporal_overlap(self, default_folds):
        for f in default_folds:
            assert f['train_end'] < f['val_start'], f"Fold {f['fold_id']}: train overlaps val"
            assert f['val_end'] < f['test_start'], f"Fold {f['fold_id']}: val overlaps test"

    def test_minimum_gap_respected(self, default_folds):
        for f in default_folds:
            assert f['train_val_gap'] >= 2, f"Fold {f['fold_id']}: train-val gap < 2"
            assert f['val_test_gap'] >= 2, f"Fold {f['fold_id']}: val-test gap < 2"

    def test_expanding_train(self, default_folds):
        """Janelas de treino devem crescer monotonicamente (walk-forward expansível)."""
        train_sizes = [f['train_end'] - f['train_start'] + 1 for f in default_folds]
        assert train_sizes == sorted(train_sizes)
        assert train_sizes[0] >= 8

    def test_first_fold_starts_at_2000(self, default_folds):
        assert default_folds[0]['train_start'] == 2000

    def test_last_fold_ends_at_2023(self, default_folds):
        assert default_folds[-1]['test_end'] == 2023

    def test_val_and_test_length(self, default_folds):
        for f in default_folds:
            val_len = f['val_end'] - f['val_start'] + 1
            test_len = f['test_end'] - f['test_start'] + 1
            assert val_len == 2, f"Fold {f['fold_id']}: val_len={val_len}"
            assert test_len == 2, f"Fold {f['fold_id']}: test_len={test_len}"

    def test_no_folds_for_impossible_params(self):
        folds = _generate_folds(
            start_year=2000, end_year=2005,
            min_train=8, val_len=2, test_len=2, gap=2,
        )
        assert len(folds) == 0

    def test_custom_step(self):
        folds_step1 = _generate_folds(2000, 2023, 8, 2, 2, 2, step=1)
        folds_step2 = _generate_folds(2000, 2023, 8, 2, 2, 2, step=2)
        assert len(folds_step2) < len(folds_step1)
        assert len(folds_step2) == 5  # 9 folds / step 2 = 5


# ---------------------------------------------------------------------------
# 3. Filtro greedy de colinearidade pairwise
# ---------------------------------------------------------------------------

def _greedy_collinearity_filter(corr_matrix, features, threshold=0.8):
    """
    Filtro de colinearidade standalone extraído de setup.py.
    Retorna a lista filtrada de features.
    """
    selected = []
    for feature in features:
        if feature not in corr_matrix.columns:
            continue
        if not selected:
            selected.append(feature)
        else:
            max_corr = 0.0
            for sel_feat in selected:
                if sel_feat in corr_matrix.columns:
                    corr_val = abs(corr_matrix.loc[feature, sel_feat])
                    if corr_val > max_corr:
                        max_corr = corr_val
            if max_corr < threshold:
                selected.append(feature)
    return selected


class TestCollinearityFilter:
    """Testes para filtro greedy de correlação pairwise."""

    def test_uncorrelated_features_all_kept(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.randn(100),
        })
        corr = df.corr().abs()
        result = _greedy_collinearity_filter(corr, ['a', 'b', 'c'], threshold=0.8)
        assert result == ['a', 'b', 'c']

    def test_perfectly_correlated_pair(self):
        np.random.seed(42)
        x = np.random.randn(100)
        df = pd.DataFrame({'a': x, 'b': x * 1.0 + 0.001 * np.random.randn(100), 'c': np.random.randn(100)})
        corr = df.corr().abs()
        result = _greedy_collinearity_filter(corr, ['a', 'b', 'c'], threshold=0.8)
        assert 'a' in result
        assert 'b' not in result  # b correlaciona com a em ~1.0
        assert 'c' in result

    def test_single_feature(self):
        corr = pd.DataFrame({'a': [1.0]}, index=['a'])
        assert _greedy_collinearity_filter(corr, ['a']) == ['a']

    def test_empty_features(self):
        corr = pd.DataFrame()
        assert _greedy_collinearity_filter(corr, []) == []

    def test_order_matters(self):
        """Primeira feature é sempre mantida; segunda é rejeitada se correlacionada."""
        np.random.seed(42)
        x = np.random.randn(100)
        df = pd.DataFrame({'a': x, 'b': x + 0.001 * np.random.randn(100)})
        corr = df.corr().abs()
        result_ab = _greedy_collinearity_filter(corr, ['a', 'b'], threshold=0.8)
        result_ba = _greedy_collinearity_filter(corr, ['b', 'a'], threshold=0.8)
        assert result_ab == ['a']
        assert result_ba == ['b']

    def test_threshold_boundary(self):
        """Feature exatamente no threshold deve ser rejeitada (< estrito)."""
        corr = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=['a', 'b'], columns=['a', 'b']
        )
        result = _greedy_collinearity_filter(corr, ['a', 'b'], threshold=0.8)
        assert result == ['a']

    def test_chain_correlation(self):
        """a-b corr=0.9, b-c corr=0.9, mas a-c corr=0.3 => mantém a,c."""
        corr = pd.DataFrame(
            [[1.0, 0.9, 0.3],
             [0.9, 1.0, 0.9],
             [0.3, 0.9, 1.0]],
            index=['a', 'b', 'c'], columns=['a', 'b', 'c']
        )
        result = _greedy_collinearity_filter(corr, ['a', 'b', 'c'], threshold=0.8)
        assert result == ['a', 'c']


# ---------------------------------------------------------------------------
# 4. Validação da configuração científica
# ---------------------------------------------------------------------------

class TestScientificConfig:
    """Testes para verificar que a configuração científica tem as chaves requeridas e valores válidos."""

    @pytest.fixture
    def config(self):
        from core.scientific_config import SCIENTIFIC_CONFIG
        return SCIENTIFIC_CONFIG

    def test_required_keys_exist(self, config):
        required = [
            'random_seed', 'collinearity_threshold', 'temporal_gap_years',
            'feature_transform', 'bootstrap_iters',
            'temporal_range_start', 'temporal_range_end',
            'folds_min_train_years', 'folds_val_len_years', 'folds_test_len_years',
            'sesoi_r2', 'sesoi_wape', 'sesoi_mase',
        ]
        for key in required:
            assert key in config, f"Missing key: {key}"

    def test_seed_is_int(self, config):
        assert isinstance(config['random_seed'], int)

    def test_collinearity_threshold_range(self, config):
        t = config['collinearity_threshold']
        assert 0.5 <= t <= 1.0, f"Threshold {t} outside reasonable range [0.5, 1.0]"

    def test_temporal_range_valid(self, config):
        assert config['temporal_range_start'] < config['temporal_range_end']
        span = config['temporal_range_end'] - config['temporal_range_start'] + 1
        min_required = (config['folds_min_train_years']
                       + config['folds_val_len_years']
                       + config['folds_test_len_years']
                       + 2 * config['temporal_gap_years'])
        assert span >= min_required, f"Temporal span {span} < min required {min_required}"

    def test_transform_is_symmetric_log(self, config):
        assert config['feature_transform'] == 'symmetric_log'

    def test_sesoi_positive(self, config):
        for key in ['sesoi_r2', 'sesoi_wape', 'sesoi_mase']:
            assert config[key] > 0, f"{key} must be positive"


# ---------------------------------------------------------------------------
# 5. Completude dos estratos de países
# ---------------------------------------------------------------------------

class TestCountryStrata:
    """Testes para a configuração de estratificação geográfica."""

    @pytest.fixture(autouse=True)
    def _load_strata(self):
        from core.config import COUNTRY_STRATA
        self.STRATA = COUNTRY_STRATA

    def test_total_32_countries(self):
        all_countries = [c for s in self.STRATA.values() for c in s]
        assert len(all_countries) == 32

    def test_no_duplicate_countries(self):
        all_countries = [c for s in self.STRATA.values() for c in s]
        assert len(all_countries) == len(set(all_countries))

    def test_all_iso2_codes(self):
        for stratum, countries in self.STRATA.items():
            for code in countries:
                assert len(code) == 2 and code.isalpha() and code.isupper(), \
                    f"Invalid ISO-2 code '{code}' in {stratum}"


# ---------------------------------------------------------------------------
# 6. Validação de integridade temporal
# ---------------------------------------------------------------------------

class TestTemporalIntegrity:
    """Testes para lógica de validação temporal anti-leak."""

    @staticmethod
    def _validate(train_years, val_years, test_years, min_gap=2):
        """Validação standalone extraída de BaseArchitectureML."""
        if not (train_years[1] < val_years[0] < val_years[1] < test_years[0]):
            return False
        if (val_years[0] - train_years[1] - 1) < min_gap:
            return False
        if (test_years[0] - val_years[1] - 1) < min_gap:
            return False
        return True

    def test_valid_split(self):
        assert self._validate((2000, 2007), (2010, 2011), (2014, 2015))

    def test_insufficient_train_val_gap(self):
        assert not self._validate((2000, 2008), (2010, 2011), (2014, 2015))

    def test_insufficient_val_test_gap(self):
        assert not self._validate((2000, 2007), (2010, 2011), (2013, 2014))

    def test_overlapping_train_val(self):
        assert not self._validate((2000, 2010), (2010, 2011), (2014, 2015))

    def test_reversed_order(self):
        assert not self._validate((2014, 2015), (2010, 2011), (2000, 2007))

    def test_all_default_folds_valid(self):
        folds = _generate_folds(2000, 2023, 8, 2, 2, 2)
        for f in folds:
            assert self._validate(
                (f['train_start'], f['train_end']),
                (f['val_start'], f['val_end']),
                (f['test_start'], f['test_end']),
            ), f"Fold {f['fold_id']} fails temporal integrity"


# ---------------------------------------------------------------------------
# 7. Testes de importação (via conftest.py PYTHONPATH)
# ---------------------------------------------------------------------------

class TestImports:
    """Verifica que os módulos core são importáveis via configuração de path do conftest.py."""

    def test_import_scientific_config(self):
        from core.scientific_config import SCIENTIFIC_CONFIG, RANDOM_SEED
        assert RANDOM_SEED == 42
        assert isinstance(SCIENTIFIC_CONFIG, dict)

    def test_import_config(self):
        from core.config import COUNTRY_STRATA, LATIN_AMERICA_COUNTRIES
        assert len(LATIN_AMERICA_COUNTRIES) == 32
        assert 'large_economies' in COUNTRY_STRATA

    def test_import_get_project_root(self):
        from core.config import get_project_root
        root = get_project_root()
        assert root.endswith("dw-vs-dl-dropout-prediction-latam")
