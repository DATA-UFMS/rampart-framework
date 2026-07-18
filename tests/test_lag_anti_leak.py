#!/usr/bin/env python3
"""Lags do target e gaps temporais por fold.

Estes testes exigiam artefatos pré-gerados em disco e portanto **pulavam** em
toda execução, inclusive no CI: um teste anti-leakage que pula não protege nada.

As duas propriedades que verificam são puras -- um lag de k anos só pode existir
onde há observação em t-k, e um fold só é válido se respeita os gaps -- então não
precisam do pipeline, apenas de um painel em memória.

O painel sintético tem lacunas deliberadas: uma entidade com série completa, uma
com um ano faltando no meio, e uma que começa depois do início. Sem lacunas, um
teste de lag passa trivialmente.
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

from core.scientific_config import SCIENTIFIC_CONFIG
from core.validation import TemporalValidator

TARGET = 'dropout_rate'
LAG = 2


def _panel_with_gaps() -> pd.DataFrame:
    rows = []
    for year in range(2000, 2011):            # série completa
        rows.append(('AAA', year))
    for year in range(2000, 2011):            # 2005 ausente
        if year != 2005:
            rows.append(('BBB', year))
    for year in range(2004, 2011):            # começa depois
        rows.append(('CCC', year))
    frame = pd.DataFrame(rows, columns=['country_code', 'year'])
    frame[TARGET] = np.random.default_rng(3).uniform(1.0, 20.0, len(frame))
    return frame


def _with_lag(frame: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Constrói o lag por junção temporal, como os paradigmas fazem."""
    previous = frame[['country_code', 'year', TARGET]].copy()
    previous['year'] = previous['year'] + lag
    previous = previous.rename(columns={TARGET: f'{TARGET}_lag_{lag}'})
    return frame.merge(previous, on=['country_code', 'year'], how='left')


def _orphan_lags(panel: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Lags preenchidos que não têm observação de origem em t-k."""
    reference = source[['country_code', 'year', TARGET]].copy()
    reference['year'] = reference['year'] + LAG
    reference = reference.rename(columns={TARGET: 'source'})
    checked = panel.merge(reference, on=['country_code', 'year'], how='left')
    filled = checked[checked[f'{TARGET}_lag_{LAG}'].notna()]
    return filled[filled['source'].isna()]


class TestLagHasASourceObservation:

    def test_no_lag_value_lacks_its_source(self):
        source = _panel_with_gaps()
        assert _orphan_lags(_with_lag(source, LAG), source).empty

    def test_the_fixture_actually_exercises_the_property(self):
        """Sem lacunas o teste acima passaria trivialmente."""
        panel = _with_lag(_panel_with_gaps(), LAG)
        assert panel[f'{TARGET}_lag_{LAG}'].isna().sum() > 0

    def test_the_absent_lags_are_exactly_the_expected_ones(self):
        panel = _with_lag(_panel_with_gaps(), LAG)
        absent = {(r.country_code, r.year) for r in
                  panel[panel[f'{TARGET}_lag_{LAG}'].isna()].itertuples()}
        assert ('AAA', 2000) in absent and ('AAA', 2001) in absent
        assert ('BBB', 2007) in absent, 'a lacuna de 2005 deveria aparecer em 2007'
        assert ('CCC', 2004) in absent and ('CCC', 2005) in absent
        assert ('AAA', 2002) not in absent

    def test_a_forged_lag_is_detected(self):
        """A checagem só vale se reprovar um lag sem origem."""
        source = _panel_with_gaps()
        panel = _with_lag(source, LAG)
        panel.loc[(panel['country_code'] == 'BBB') &
                  (panel['year'] == 2007), f'{TARGET}_lag_{LAG}'] = 9.99
        assert not _orphan_lags(panel, source).empty


class TestFoldGaps:

    @staticmethod
    def _folds(gap: int):
        folds = []
        for index in range(3):
            train_end = 2007 + index
            val_start = train_end + gap + 1
            val_end = val_start + 1
            test_start = val_end + gap + 1
            folds.append({
                'fold_id': index,
                'train_start': 2000, 'train_end': train_end,
                'train_gap_start': train_end + 1, 'train_gap_end': val_start - 1,
                'val_start': val_start, 'val_end': val_end,
                'val_gap_start': val_end + 1, 'val_gap_end': test_start - 1,
                'test_start': test_start, 'test_end': test_start + 1,
            })
        return folds

    def test_declared_gaps_are_respected(self):
        gap = SCIENTIFIC_CONFIG['temporal_gap_years']
        for fold in self._folds(gap):
            assert fold['val_start'] - fold['train_end'] - 1 >= gap
            assert fold['test_start'] - fold['val_end'] - 1 >= gap

    def test_a_narrow_gap_violates_the_requirement(self):
        """O teste anterior seria vácuo se nada reprovasse um gap estreito."""
        gap = SCIENTIFIC_CONFIG['temporal_gap_years']
        narrow = self._folds(gap - 1)
        assert [f for f in narrow
                if f['val_start'] - f['train_end'] - 1 < gap]

    def test_the_validator_accepts_conforming_folds(self):
        """Liga a propriedade ao componente que a impõe, não só à aritmética."""
        gap = SCIENTIFIC_CONFIG['temporal_gap_years']
        TemporalValidator(min_gap_years=gap,
                          embargo_years=0).enforce_walk_forward(self._folds(gap))

    def test_the_validator_rejects_a_narrow_gap(self):
        gap = SCIENTIFIC_CONFIG['temporal_gap_years']
        validator = TemporalValidator(min_gap_years=gap, embargo_years=0)
        with pytest.raises(Exception):
            validator.enforce_walk_forward(self._folds(gap - 1))

    def test_folds_serialise_as_the_pipeline_writes_them(self):
        json.dumps({'folds': self._folds(
            SCIENTIFIC_CONFIG['temporal_gap_years'])})
